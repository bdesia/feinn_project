
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import SinusoidalEmbedding, GridEmbedding2D
from neuralop.losses import LpLoss

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

class RVEDataset(Dataset):
    """
    Optimized Dataset for Dual-Encoder FNO Training on RVE Data.
    - Lazy loading HDF5 with swmr=True for multiprocessing safety.
    - Channel-first (CHW) layout consistent with NeuralOperator library expectations.
    - Uses UnitGaussianNormalizers with precomputed stats
    """

    def __init__(self, h5_path: str | Path, split: str = 'train', normalize: bool = True):
        self.h5_path = Path(h5_path)
        self.split = split
        self.normalize = normalize
        self.archive = None  # Initialized in __getitem__ to avoid pickling errors

        self._check_split()

        # Initial read for metadata and precomputed statistics
        with h5py.File(self.h5_path, 'r') as f:
            self.N = f[split]['x_local'].shape[0]
            stats = f['stats']

            # Normalizers - mean/std are per-channel (shape (C,) or broadcastable)
            self.x_normalizer = UnitGaussianNormalizer(
                mean=torch.from_numpy(stats['mean_x_local'][:]).float(),   # shape (phase + nstatev,)
                std=torch.from_numpy(stats['std_x_local'][:]).float()
            )
            self.y_normalizer = UnitGaussianNormalizer(
                mean=torch.from_numpy(stats['mean_y_local'][:]).float(),   # shape (3 stresses + nstatev,)
                std=torch.from_numpy(stats['std_y_local'][:]).float()
            )
            self.global_normalizer = UnitGaussianNormalizer(
                mean=torch.from_numpy(stats['mean_x_global'][:]).float(),  # shape (3 strain + nprops,)
                std=torch.from_numpy(stats['std_x_global'][:]).float()
            )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Open file once per worker process (when DataLoader num_workers > 0)
        if self.archive is None:
            self.archive = h5py.File(self.h5_path, 'r', swmr=True)      # swmr: single writer multiple readers

        data = self.archive[self.split]

        # Load from HDF5 (stored as HWC)
        x_local = torch.from_numpy(data['x_local'][idx]).float()   # (H, W, phase + nstatev + 3 stresses + 3 dstrain)
        x_global = torch.from_numpy(data['x_global'][idx]).float() # (3 strain + nprops,)
        y_local  = torch.from_numpy(data['y_local'][idx]).float()  # (H, W, 3 stresses + nstatev)

        if self.normalize:
            # Apply NeuralOperator normalization (broadcasts across H, W)
            x_local = self.x_normalizer.transform(x_local)
            x_global = self.global_normalizer.transform(x_global)
            y_local = self.y_normalizer.transform(y_local)

        # Permute to Channel-First (CHW) for FNO/Fourier operations
        x_local = x_local.permute(2, 0, 1).contiguous()   # (channels, H, W)
        y_local = y_local.permute(2, 0, 1).contiguous()   # (channels, H, W)

        return x_local, x_global[:3], y_local

    def get_normalizers(self):
        """Returns all three normalizers for training/inference workflow"""
        return self.x_normalizer, self.global_normalizer, self.y_normalizer

    def __del__(self):
        """Ensure HDF5 file handles are closed on object destruction"""
        if self.archive is not None:
            self.archive.close()
    
    def _check_split(self):
        """Validate that the specified split exists in the HDF5 file."""
        with h5py.File(self.h5_path, 'r') as f:
            if self.split not in f:
                raise ValueError(f"Split '{self.split}' not found in {self.h5_path}. Available splits: train, val, test.")

class DualEncoderFNO(nn.Module):
    """
    Dual-Encoder Fourier Neural Operator for RVE Analysis with FiLM Conditioning
    - Spatial Branch: Lifts local microstructural features (with optional positional grid) to latent space.
    - Global Branch: Sinusoidal embedding of macroscopic parameters (essential high-frequency encoding).
    - FiLM Conditioning: Feature-wise Linear Modulation using the sinusoidal embedding 
    - FNO Core: Multi-layer Fourier Neural Operator.
    - Output Projection: Maps back to target fields (stresses, damage, plastic strain, etc.).


    Parameters
    ----------
    in_channels : int
        Number of channels in input function. Determined by the problem.
    out_channels : int
        Number of channels in output function. Determined by the problem.
    n_macro: int
        Number of pixel-independent variables. Determined by the problem.
    n_modes : int
        Number of modes to keep in Fourier Layer, equally along each dimension.
        n_modes must be larger enough but smaller than max_resolution//2 (Nyquist frequency)
    hidden_channels : int
        Width of the FNO (i.e. number of channels).
        This significantly affects the number of parameters of the FNO.
        Good starting point can be 64, and then increased if more expressivity is needed.
        Update lifting_channel_ratio and projection_channel_ratio accordingly since they are proportional to hidden_channels.
    n_layers : int, optional
        Number of Fourier Layers. Default: 4

    Other Parameters
    ----------------
    lifting_channel_ratio : Number, optional
        Ratio of lifting channels to hidden_channels.
        The number of lifting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    projection_channel_ratio : Number, optional
        Ratio of projection channels to hidden_channels.
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use. Default: nn.GELU()

    TO COMPLETE

    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 3,
                 n_macro: int = 7,
                 n_modes: int = 16,
                 hidden_channels: int = 64,         # Fourier layer width
                 n_layers: int = 4,
                 lifting_channel_ratio: int = 2,
                 projection_channel_ratio: int = 2,
                 macro_embed_nfreq: int = 128,
                 use_positional_grid: bool = True,
                 non_linearity: nn.Module = nn.GELU(),
                 **fno_blocks_kwargs
                 ):
        super().__init__()

        self.n_dim = 2      # 2D case

        self.n_modes = n_modes
        self.hidden_channels = hidden_channels          # hidden channels
        self.n_layers = n_layers

        self.use_positional_grid = use_positional_grid
        self.n_macro = n_macro
        self.macro_embed_nfreq = macro_embed_nfreq

        # init lifting and projection channels using ratios w.r.t width
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * hidden_channels)

        # =============== Local (Micro) Branch ===============
        in_channels = in_channels

        if use_positional_grid:
            self.positional_embedding = GridEmbedding2D(
                in_channels=in_channels,
                grid_boundaries=[[0., 1.], [0., 1.]]
            )
            in_channels += self.n_dim     # update number of channels if positional embeddings

        self.lifting = ChannelMLP(
            in_channels = in_channels,
            out_channels = hidden_channels,
            hidden_channels = self.lifting_channels,
            n_layers=2,
            n_dim = self.n_dim,
            non_linearity = non_linearity,
            )

        # =============== Global (Macro) Branch ===============
        # Sinusoidal Embedding 
        self.global_embed = SinusoidalEmbedding(
                                in_channels = n_macro,
                                num_frequencies = macro_embed_nfreq,
                                embedding_type = 'nerf'
                            )

        self.global_embed_dim = n_macro * 2 * macro_embed_nfreq

        # =============== Mix Local-Global ===============
        # FiLM Conditioning (Scale + Shift)
        self.film_gamma = nn.Linear(self.global_embed_dim, hidden_channels)
        self.film_beta  = nn.Linear(self.global_embed_dim, hidden_channels)

        # W&B initialization
        with torch.no_grad():
            self.film_gamma.weight.zero_()
            self.film_gamma.bias.fill_(0.0)
            self.film_beta.weight.zero_()
            self.film_beta.bias.zero_()

        # =============== FNO blocks Core ===============
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=(self.n_modes,) * self.n_dim,
            non_linearity=non_linearity,
            n_layers=n_layers,
            **fno_blocks_kwargs
        )

        # =============== Output Projection ===============
        self.projection = ChannelMLP(
            in_channels = self.hidden_channels,
            out_channels = out_channels,
            hidden_channels = self.projection_channels,
            n_layers = 2,
            n_dim = self.n_dim,
            non_linearity = non_linearity,
            )
        
    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        """
        x_local:  (B, in_channels, H, W)  → microstructural field
        x_global: (B, nmacro)             → macroscopic loading / parameters
        """
        # Local Branch: positional embedding + initial lifting
        if self.use_positional_grid:
            x_local = self.positional_embedding(x_local)
        x = self.lifting(x_local)                         # (B, width, H, W)

        # Global Branch: Sinusoidal Embedding
        global_vec = self.global_embed(x_global)                # (B, global_embed_dim)

        # Mix Local & Global: FiLM Conditioning
        gamma = self.film_gamma(global_vec).unsqueeze(-1).unsqueeze(-1) + 1
        beta  = self.film_beta(global_vec).unsqueeze(-1).unsqueeze(-1)  
        x = gamma * x + beta                              # (B, width, H, W)

        # FNO Core
        output_shape = [None] * self.n_layers
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
            
        # Output
        x = self.projection(x)

        return x    # (B, out_channels, H, W, )
    
    def count_parameters(self) -> int:
            return sum(p.numel() * 2 if p.is_complex() else p.numel() for p in self.parameters() if p.requires_grad)

class Trainer:
    """
    Trainer for DualEncoderFNO.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fun: object = None,
        wandb_log: bool = False,
        device: Optional[torch.device] = None,
        save_dir: str | Path = "checkpoints",
        min_delta: float = 1e-6,        # for saving best model
        max_grad_norm: Optional[float] = 1.0,
        verbose: bool = True,
        ):
         
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.verbose = verbose
        # only log to wandb if a run is active
        self.wandb_log = wandb_log and wandb_available and wandb.run is not None

        self.loss_fun = loss_fun if loss_fun is not None else LpLoss(d=2, p=2)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.min_delta = min_delta
        self.max_grad_norm = max_grad_norm

        self.optimizer = None
        self.scheduler = None

        self.history = {"train_loss": [], "val_loss": []}
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0

    def _process_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moves batch data to the target device asynchronously."""
        x_local, x_global, y = batch
        return (
            x_local.to(self.device, non_blocking=True),
            x_global.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True)
        )

    def _train_one_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Run one batch and return loss and size """
        x_local, x_global, y = self._process_batch(batch)
        batch_size = x_local.size(0)

        self.optimizer.zero_grad(set_to_none=True)          # Zero the parameter gradients
        pred = self.model(x_local, x_global)                # Forward pass: Output shape (B, C, H, W)
        loss = self.loss_fun(pred, y)                       # Compute loss
        loss.backward()                                     # Backpropagate

        # Gradient clipping
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=2.0
            )

        self.optimizer.step()                               # Optimizer step

        return loss, batch_size

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        """Executes a single training epoch and returns the average loss."""
        
        # Set the model to training mode
        self.model.train()

        # Initialize loss and number of samples
        epoch_loss = 0.0
        total_samples = 0

        # Loop over dataloader batches
        for batch in tqdm(dataloader, desc="Train", leave=False, disable=not self.verbose):
            
            # Run batch
            loss, batch_size = self._train_one_batch(batch)
            
            # Weight loss by actual batch size
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        # Step standard scheduler
        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        # Mean epoch loss
        weighted_loss = epoch_loss / total_samples

        return weighted_loss
    
    def _validate_one_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Run one batch and return loss and size """
        # Desactive grad
        with torch.no_grad():
            x_local, x_global, y = self._process_batch(batch)
            batch_size = x_local.size(0)
            pred = self.model(x_local, x_global)                # Forward pass
            loss = self.loss_fun(pred, y)                       # Compute loss

            return loss, batch_size
    
    def _validate_one_epoch(self, dataloader: DataLoader) -> float:
        """Executes a single validation epoch and returns the average loss."""
        
        # Set model to evaluation mode
        self.model.eval()

        # Initialize loss and number of samples
        epoch_loss = 0.0
        total_samples = 0

        # Loop over dataloader batches
        for batch in tqdm(dataloader, desc="Valid", leave=False, disable=not self.verbose):
            
            # Run batch
            loss, batch_size = self._validate_one_batch(batch)
            
            # Weight loss by actual batch size
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # Mean epoch loss
        weighted_loss = epoch_loss / total_samples

        # Step plateau scheduler if used
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(weighted_loss)

        return weighted_loss

    def fit(self, train_loader: DataLoader, 
                val_loader: DataLoader, 
                epochs: int,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler = None,
                patience: int = None,
                model_name: str = "best_DualEncoderFNO.pth",
                verbose: Optional[bool] = None,
                ) -> Dict[str, list]:
        """Trains the model for a specified number of epochs with early stopping."""

        print(f"DualEncoderFNO Training: {epochs} epochs\n")

        if verbose is not None:
            self.verbose = verbose
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience or epochs + 1      # desactive if None

        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss   = self._validate_one_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Check for best model and save
            is_best = val_loss < (self.best_loss - self.min_delta)
            if is_best:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                }
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                    
                torch.save(checkpoint, self.save_dir / model_name)

                if verbose:
                    print(f"   Best model saved (Epoch {epoch})")
            else:
                self.counter += 1

            # Logging
            if self.wandb_log:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "best_val_loss": self.best_loss,
                })

            if verbose or epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {self.best_loss:.6f}")

            # Early stopping check
            if self.counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"\n Training finished. Best epoch: {self.best_epoch} (Loss = {self.best_loss:.6f})")
        return self.history

    def plot_history(self, title: str = "DualEncoderFNO Training"):
        """Plots the training and validation loss curves."""

        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(11, 6))
        plt.plot(epochs, self.history["train_loss"], label='Train L2', color='darkgrey', linewidth=2.2)
        plt.plot(epochs, self.history["val_loss"],   label='Val L2',   color='cornflowerblue', linewidth=2.2)
        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Loss')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.show()

    def load_best_model(self, model_path):
        """Loads the weights of the best performing model from disk."""
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded (Epoch {checkpoint['epoch']}, L2 = {checkpoint['best_loss']:.6f})")
        else:
            print("No saved model found")