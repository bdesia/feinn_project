
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
from typing import Optional, Dict, Tuple, Callable

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

import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py

class RVEDataset(Dataset):
    """
    Optimized Dataset for Dual-Encoder FNO Training on RVE Data.
    - Added `in_memory` flag to bypass HDF5 disk I/O during training.
    - Vectorized normalization and permutation in RAM for massive speedups.
    - Maintains Lazy loading fallback for low-RAM environments.
    """

    def __init__(self, h5_path: str | Path, 
                 split: str = 'train',
                 fraction: float = 1.0,         # for sub-sampling
                 normalize: bool = True, 
                 in_memory: bool = False,):
        
        self.h5_path = Path(h5_path)
        self.split = split
        self.fraction = fraction
        self.normalize = normalize
        self.in_memory = in_memory
        self.archive = None  # Initialized in __getitem__ to avoid pickling errors

        self._check_split()

        # Initial read for metadata and precomputed statistics
        with h5py.File(self.h5_path, 'r') as f:
            N_total = f[split]['x_local'].shape[0]
            stats = f['stats']

            if self.fraction < 1.0:
                self.N = int(N_total * self.fraction)
                # h5py requires that boolean or list indices be strictly ordered for fast access
                indexs = np.sort(np.random.choice(N_total, self.N, replace=False))
            else:
                self.N = N_total
                indexs = slice(None)

            # Normalizers 
            self.x_normalizer = UnitGaussianNormalizer(
                mean=torch.from_numpy(stats['mean_x_local'][:]).float(),
                std=torch.from_numpy(stats['std_x_local'][:]).float()
            )
            self.global_normalizer = UnitGaussianNormalizer(
                mean=torch.from_numpy(stats['mean_x_global'][:]).float(),
                std=torch.from_numpy(stats['std_x_global'][:]).float()
            )
            self.y_normalizer = UnitGaussianNormalizer(
                mean=torch.from_numpy(stats['mean_y_local'][:]).float(),
                std=torch.from_numpy(stats['std_y_local'][:]).float()
            )

            # --- Load to RAM ---
            if self.in_memory:
                print(f"Loading {self.fraction*100:.0f}% of '{split}' split into RAM. This may take a moment...")
                # Slicing [:] reads the entire chunk sequentially (extremely fast compared to random access)
                self.x_local_data = torch.from_numpy(f[split]['x_local'][indexs]).float()
                self.x_global_data = torch.from_numpy(f[split]['x_global'][indexs]).float()
                self.y_local_data = torch.from_numpy(f[split]['y_local'][indexs]).float()

                if self.normalize:
                    # Vectorized operations: Transform the entire dataset at once (C++ backend)
                    self.x_local_data = self.x_normalizer.transform(self.x_local_data)
                    self.x_global_data = self.global_normalizer.transform(self.x_global_data)
                    self.y_local_data = self.y_normalizer.transform(self.y_local_data)

                # Vectorized permutation: (N, H, W, C) -> (N, C, H, W) 
                # Done once per epoch instead of 60,000 times.
                self.x_local_data = self.x_local_data.permute(0, 3, 1, 2).contiguous()
                self.y_local_data = self.y_local_data.permute(0, 3, 1, 2).contiguous()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # --- FAST PATH ---
        if self.in_memory:
            # Data is already parsed, normalized, permuted, and sitting in CPU RAM.
            # This takes microseconds.
            return self.x_local_data[idx], self.x_global_data[idx, :3], self.y_local_data[idx]

        # --- LAZY LOADING FALLBACK ---
        if self.archive is None:
            self.archive = h5py.File(self.h5_path, 'r', swmr=True)

        data = self.archive[self.split]

        x_local = torch.from_numpy(data['x_local'][idx]).float()
        x_global = torch.from_numpy(data['x_global'][idx]).float()
        y_local  = torch.from_numpy(data['y_local'][idx]).float()

        if self.normalize:
            x_local = self.x_normalizer.transform(x_local)
            x_global = self.global_normalizer.transform(x_global)
            y_local = self.y_normalizer.transform(y_local)

        x_local = x_local.permute(2, 0, 1).contiguous()
        y_local = y_local.permute(2, 0, 1).contiguous()

        return x_local, x_global[:3], y_local

    def get_normalizers(self):
        return self.x_normalizer, self.global_normalizer, self.y_normalizer

    def __del__(self):
        if self.archive is not None:
            self.archive.close()
            
    def __getstate__(self):
        state = self.__dict__.copy()
        state['archive'] = None  
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.archive = None  

    def _check_split(self):
        with h5py.File(self.h5_path, 'r') as f:
            if self.split not in f:
                raise ValueError(f"Split '{self.split}' not found. Available splits: {list(f.keys())}")

class DualEncoderFNO(nn.Module):
    """
    Dual-Encoder Fourier Neural Operator for RVE Analysis with FiLM Conditioning
    - Spatial Branch: Lifts local microstructural features (with optional positional grid and sinusoidal embedding) to latent space.
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
    use_positional_grid : bool, optional
        If True, add 2D positional grid embeddings to the input of the lifting block. Default: True
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use. Default: nn.GELU()
    film_per_layer : bool, optional
        If True, compute separate FiLM parameters for each FNO layer (more expressive, more parameters). 
        Default: False (one FiLM conditioning previous to the first FNO block).
    film_mlp_layers : int, optional    
        Number of layers in the FiLM MLP. Default: 2
    film_mlp_neurons : int, optional
        Number of neurons in each hidden layer of the FiLM MLP. Only apply if film_mlp_layers > 1.
        Default: 128 
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 3,
                 n_macro: int = 3,
                 n_modes: int = 16,
                 hidden_channels: int = 64,         # Fourier layer width
                 n_layers: int = 4,
                 lifting_channel_ratio: int = 2,
                 projection_channel_ratio: int = 2,
                 use_positional_grid: bool = True,
                 non_linearity: nn.Module = nn.GELU(),
                 film_per_layer: bool = False,
                 film_mlp_layers: int = 2,
                 film_mlp_neurons: int = 128,
                 **fno_blocks_kwargs
                 ):
        super().__init__()

        self.n_dim = 2      # 2D case
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_macro = n_macro
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.use_positional_grid = use_positional_grid
        self.non_linearity = non_linearity
        self.film_per_layer = film_per_layer
        self.film_mlp_layers = film_mlp_layers
        self.film_mlp_neurons = film_mlp_neurons
        
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

        # =============== FiLM Conditioning ===============
        film_out_features = hidden_channels * n_layers if film_per_layer else hidden_channels
        
        film_layers = []
        in_dim = n_macro
        # Hidden layers if film_mlp_layers > 1
        for _ in range(max(1, film_mlp_layers) - 1):
            film_layers.append(nn.Linear(in_dim, film_mlp_neurons))
            film_layers.append(non_linearity)
            in_dim = film_mlp_neurons
        # Output layer
        film_layers.append(nn.Linear(in_dim, film_out_features * 2))
        
        self.film_mlp = nn.Sequential(*film_layers)
        
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
        
        # Check that n_modes does not exceed Nyquist frequency for the input resolution
        _, _, H, W = x_local.shape
        max_modes = min(H, W) // 2
        
        if self.n_modes > max_modes:
            raise ValueError(
                f"Number of modes (n_modes={self.n_modes}) exceeds the Nyquist frequency "
                f"for the input resolution {H}x{W}. "
                f"n_modes must be <= {max_modes}."
            )
        
        # Local Branch: positional embedding + initial lifting
        if self.use_positional_grid:
            x_local = self.positional_embedding(x_local)
        x = self.lifting(x_local)                           # (B, width, H, W)

        # Mix Local & Global: Compute FiLM parameters
        gammas, betas = self._compute_film_params(x_global)
        
        # FNO Core w/FiLM conditioning
        output_shape = [None] * self.n_layers
        for layer_idx in range(self.n_layers):
            if self.film_per_layer:
                x = gammas[:, layer_idx] * x + betas[:, layer_idx]                          # (B, width, H, W)
            elif layer_idx == 0:
                x = gammas * x + betas
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        # Output
        x = self.projection(x)

        return x    # (B, out_channels, H, W, )

    def _compute_film_params(self, x):
        film_params = self.film_mlp(x)                  # (B, 2 * film_out_features)
        gammas, betas = film_params.chunk(2, dim=-1)    # Each one is (B, film_out_features)
        
        if self.film_per_layer:
            # Reshape to (Batch, n_layers, C, 1, 1) 
            gammas = gammas.view(-1, self.n_layers, self.hidden_channels, 1, 1)
            betas = betas.view(-1, self.n_layers, self.hidden_channels, 1, 1)
        else:
            # Reshape to (Batch, C, 1, 1)
            gammas = gammas.view(-1, self.hidden_channels, 1, 1)
            betas = betas.view(-1, self.hidden_channels, 1, 1)
        return gammas, betas

    def save_config(self, path: str | Path = "./checkpoints/rve_fno_config.pth"):
        """Save model configuration"""
        
        config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "n_macro": self.n_macro,
            "n_modes": self.n_modes,
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "lifting_channel_ratio": self.lifting_channel_ratio,
            "projection_channel_ratio": self.projection_channel_ratio,
            "use_positional_grid": self.use_positional_grid,
            "non_linearity": self.non_linearity,
            "film_per_layer": self.film_per_layer,
            "film_mlp_layers": self.film_mlp_layers,
            "film_mlp_neurons": self.film_mlp_neurons,
        }
        
        torch.save(config, path)
        print(f"Saved configuration at {path}")
        
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
        val_metrics: Dict[str, Callable] = None,
        wandb_log: bool = False,
        device: Optional[torch.device] = None,
        save_dir: str | Path = "checkpoints",
        save: bool = True,
        min_delta: float = 1e-6,        # for saving best model
        max_grad_norm: Optional[float] = 1.0,
        verbose: bool = True,
        ):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        self.verbose = verbose
        # only log to wandb if a run is active
        self.wandb_log = wandb_log and wandb_available and wandb.run is not None

        self.loss_fun = loss_fun if loss_fun is not None else LpLoss(d=2, p=2)
        self.val_metrics = val_metrics or {}
        
        self.save = save
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.min_delta = min_delta
        self.max_grad_norm = max_grad_norm

        self.optimizer = None
        self.scheduler = None

        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        for metric_name in self.val_metrics.keys():
                    self.history[f"val_{metric_name}"] = []
                    
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
        avg_loss = epoch_loss / total_samples

        return avg_loss

    @torch.inference_mode()
    def _validate_one_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_local, x_global, y = self._process_batch(batch)
        batch_size = x_local.size(0)
        
        pred = self.model(x_local, x_global)
        loss = self.loss_fun(pred, y)

        batch_metrics = {name: metric_fn(pred, y).item() for name, metric_fn in self.val_metrics.items()}
        return loss, batch_size, batch_metrics
    
    def _validate_one_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Executes a single validation epoch.
        Returns:
            avg_loss: Mean loss for the epoch.
            avg_metrics: Dictionary of mean values for each extra metric.
        """
        self.model.eval()
        epoch_loss = 0.0
        total_samples = 0
        epoch_metrics_acc = {name: 0.0 for name in self.val_metrics.keys()}

        for batch in tqdm(dataloader, desc="Valid", leave=False, disable=not self.verbose):
            loss, batch_size, batch_metrics = self._validate_one_batch(batch)
            
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for name, value in batch_metrics.items():
                epoch_metrics_acc[name] += value * batch_size
        
        avg_loss = epoch_loss / total_samples
        avg_metrics = {name: acc / total_samples for name, acc in epoch_metrics_acc.items()}

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return avg_loss, avg_metrics

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
            val_loss, val_results = self._validate_one_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            for name, value in val_results.items():
                self.history[f"val_{name}"].append(value)
                
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
                    'val_loss': self.best_loss,
                }
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
                if self.save:
                    torch.save(checkpoint, self.save_dir / model_name)

                if verbose:
                    print(f"   Best model saved (Epoch {epoch})")
            else:
                self.counter += 1

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["lr"].append(current_lr)
            
            # Logging
            if self.wandb_log:
                log_data = {"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr, "epoch": epoch}
                log_data.update({f"val_{k}": v for k, v in val_results.items()})
                wandb.log(log_data)

            if self.verbose:
                metrics_str = " | ".join([f"{k.upper()}: {v:.6f}" for k, v in val_results.items()])
                extra = f" | Val {metrics_str}" if metrics_str else ""
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}{extra} | LR: {current_lr:.2e}")
            
            # Early stopping check
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"\n Training finished. Best epoch: {self.best_epoch} (Loss = {self.best_loss:.6f})")
        return self.history

    def plot_history(self, title: str = "DualEncoderFNO Training"):
        """Plots the training and validation loss curves, and the learning rate."""
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        fig, ax1 = plt.subplots(figsize=(11, 6))

        # Loss curves on the primary y-axis (left)
        line1 = ax1.plot(epochs, self.history["train_loss"], label='Train Loss', color='tab:blue', linewidth=2)
        line2 = ax1.plot(epochs, self.history["val_loss"], label='Val Loss', color='tab:orange', linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, alpha=0.35)

        # Create a secondary y-axis for the learning rate
        ax2 = ax1.twinx()
        
        # Plot the learning rate on the secondary y-axis (right)
        line3 = ax2.plot(epochs, self.history["lr"], label='Learning Rate', color='tab:red', linewidth=2.2, linestyle='--')
        
        ax2.set_ylabel('Learning Rate', fontsize=12)
        
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Combine legends from both axes
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=12)

        plt.title(title, fontsize=17, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_metric(self, metric_name: str, title: str = "DualEncoderFNO Training - Val metrics"):
        """
        Plots the evolution of a specific validation metric over epochs.
        Args:
            metric_name: Key used in the val_metrics dictionary (e.g., 'mse').
            title: Custom plot title.
        """
        key = f"val_{metric_name}"
        if key not in self.history:
            print(f"Metric '{metric_name}' not found in history. Available: {list(self.val_metrics.keys())}")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.history[key]) + 1), self.history[key], 
                 label=f'Validation {metric_name}', color='tab:green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(title or f'Validation {metric_name.upper()} over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def load_model(self, model_path):
        """Loads the weights of the model from disk."""
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded (Epoch {checkpoint['epoch']}, Val loss = {checkpoint['best_loss']:.6f})")
        else:
            print("No saved model found")

class RVEInferencer:
    """
    Class to handle FNO model predictions.
    Takes input tensors, executes the forward pass, and returns
    denormalized predictions (in real physical scale).
    """
    def __init__(self, model, y_normalizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.y_normalizer = y_normalizer
        self.device = device
        
        self.y_normalizer.to(self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, x_local, x_global):
        """
        Performs inference for a single batch or a single sample of data.
        """
        # add batch dimension (just in case)
        if x_local.dim() == 3:
            x_local = x_local.unsqueeze(0)
        if x_global.dim() == 1:
            x_global = x_global.unsqueeze(0)

        x_local = x_local.to(self.device)
        x_global = x_global.to(self.device)
        
        # Forward pass
        pred_norm = self.model(x_local, x_global)
        
        # Invert normalization to return to the physical stress scale
        if self.y_normalizer is not None:
            pred_physical = self.y_normalizer.inverse_transform(pred_norm)
        else:
            pred_physical = pred_norm
            
        return pred_physical.cpu().numpy()
    
    @torch.inference_mode()
    def predict_dataset(self, dataloader):
        self.model.eval()
        y_preds = []
        y_trues = []
        
        for batch in dataloader:
            x_local, x_global, y_local = batch
            x_local = x_local.to(self.device)
            x_global = x_global.to(self.device)
            y_local = y_local.to(self.device)

            # Get predictions (B, 3, 96, 96)
            pred = self.model(x_local, x_global)

            # Permute to (B, 96, 96, 3)
            pred = pred.permute(0, 2, 3, 1).contiguous()
            y_local = y_local.permute(0, 2, 3, 1).contiguous()

            # Normalization: inverse transform
            pred = self.y_normalizer.inverse_transform(pred)
            y_true = self.y_normalizer.inverse_transform(y_local)

            y_preds.append(pred.cpu().numpy())
            y_trues.append(y_true.cpu().numpy())

        return np.concatenate(y_trues, axis=0), np.concatenate(y_preds, axis=0)

class RVEVisualizer:
    """
    Class to visualize and compare Ground-Truth (FEM) results 
    against model predictions (FNO).
    """
    
    @staticmethod
    def plot_cross_sections(phase, y_true, y_pred, index, hline=None, vline=None, variable='Stress'):
        """
        Plots values along horizontal and vertical cross-sectional lines in a 2x3 grid.
        Row 1: Horizontal cross-section (varying X).
        Row 2: Vertical cross-section (varying Y).
        Columns: Tensor components (XX, YY, XY).
        Also overlays the inverted input phase as a shaded background.

        Args:
            phase (ndarray): Input phase data (batch_size, H, W, ...).
            y_true (ndarray): Ground truth tensor (batch_size, H, W, 3).
            y_pred (ndarray): Predicted tensor (batch_size, H, W, 3).
            index (int): Index of the sample to visualize.
            hline (int, optional): Y-index for the horizontal cut. Defaults to center.
            vline (int, optional): X-index for the vertical cut. Defaults to center.
            variable (str): Label for the Y-axis. Defaults to 'Stress'.
        """
        tt = y_true[index]  # FEM
        zz = y_pred[index]  # FNO
        
        # INVERTIMOS LA FASE AQUÍ (1 - fase)
        ph = 1.0 - phase[index].squeeze() 
        
        # Get grid size
        L_x, L_y = tt.shape[0], tt.shape[1]
        
        # If no cut lines are specified, take the center of the RVE
        if hline is None: hline = L_y // 2
        if vline is None: vline = L_x // 2

        # --- Extract Lines ---
        # Horizontal lines (varying X, constant Y = hline)
        hor_true = [tt[:, hline, i] for i in range(3)]
        hor_pred = [zz[:, hline, i] for i in range(3)]
        hor_phase = ph[:, hline]

        # Vertical lines (varying Y, constant X = vline)
        ver_true = [tt[vline, :, i] for i in range(3)]
        ver_pred = [zz[vline, :, i] for i in range(3)]
        ver_phase = ph[vline, :]

        # Create 2x3 grid
        fig, ax = plt.subplots(2, 3, figsize=(21, 10))
        x_axis_hor = np.arange(L_x)
        x_axis_ver = np.arange(L_y)
        
        comp_labels = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\sigma_{xy}$']

        for j in range(3): # Iterate over columns (components)
            
            # ==========================================
            # Row 0: Horizontal Plots (Cross-section X)
            # ==========================================
            ax0 = ax[0, j]
            
            # 1. Plot Phase Background
            ax0_phase = ax0.twinx()
            ax0_phase.fill_between(x_axis_hor, 0, hor_phase, color='gray', alpha=0.3, step='mid', label='Phase')
            ax0_phase.set_ylim(0, 1) # Assuming phase is 0 or 1
            ax0_phase.set_yticks([]) # Hide secondary Y axis ticks
            
            # 2. Plot True and Pred
            l1 = ax0.plot(x_axis_hor, hor_true[j], linewidth=3, color='red', label='FEM')      
            l2 = ax0.plot(x_axis_hor, hor_pred[j], linewidth=3, color='black', label='DualEncoderFNO', linestyle='dashed')

            # Formatting
            ax0.set_title(f'{variable} {comp_labels[j]}', fontsize=18, pad=15)
            ax0.set_xlabel(f'Cross-section X ($Y={hline}$)', fontsize=14, labelpad=10) 
            ax0.set_ylabel(f'{comp_labels[j]} [MPa]', fontsize=14, labelpad=10)
            ax0.set_xlim(0, L_x - 1)
            
            # Combine legends from both axes
            if j == 0:
                lines, labels = ax0.get_legend_handles_labels()
                lines2, labels2 = ax0_phase.get_legend_handles_labels()
                ax0.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=12)

            # Tick formatting
            ax0.xaxis.set_tick_params(which='major', size=8, width=2, direction='in', top=True)
            ax0.xaxis.set_tick_params(which='minor', size=4, width=1, direction='in', top=True)
            ax0.yaxis.set_tick_params(which='major', size=8, width=2, direction='in', right=True)
            ax0.yaxis.set_tick_params(which='minor', size=4, width=1, direction='in', right=True)
            ax0.yaxis.set_major_locator(ticker.MaxNLocator(6))
            ax0.xaxis.set_minor_locator(AutoMinorLocator())
            ax0.yaxis.set_minor_locator(AutoMinorLocator())


            # ==========================================
            # Row 1: Vertical Plots (Cross-section Y)
            # ==========================================
            ax1 = ax[1, j]
            
            # 1. Plot Phase Background
            ax1_phase = ax1.twinx()
            ax1_phase.fill_between(x_axis_ver, 0, ver_phase, color='gray', alpha=0.3, step='mid')
            ax1_phase.set_ylim(0, 1)
            ax1_phase.set_yticks([])
            
            # 2. Plot True and Pred
            ax1.plot(x_axis_ver, ver_true[j], linewidth=3, color='red', label='FEM')  
            ax1.plot(x_axis_ver, ver_pred[j], linewidth=3, color='black', label='DualEncoderFNO', linestyle='dashed')

            # Formatting
            ax1.set_xlabel(f'Cross-section Y ($X={vline}$)', fontsize=14, labelpad=10) 
            ax1.set_ylabel(f'{comp_labels[j]} [MPa]', fontsize=14, labelpad=10)
            ax1.set_xlim(0, L_y - 1)
            
            if j == 0:
                ax1.legend(loc='upper right', fontsize=12)

            # Tick formatting
            ax1.xaxis.set_tick_params(which='major', size=8, width=2, direction='in', top=True)
            ax1.xaxis.set_tick_params(which='minor', size=4, width=1, direction='in', top=True)
            ax1.yaxis.set_tick_params(which='major', size=8, width=2, direction='in', right=True)
            ax1.yaxis.set_tick_params(which='minor', size=4, width=1, direction='in', right=True)
            ax1.yaxis.set_major_locator(ticker.MaxNLocator(6))
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_contours(phase, y_true, y_pred, index, channel=0, xcor=None, ycor=None, cmap='viridis', title= None):
        """
        Plots a 1x3 grid comparing the input phase, ground truth, and model prediction.

        Args:
            phase (ndarray): Input phase data (batch_size, H, W, ...).
            y_true (ndarray): Ground truth tensor (batch_size, H, W, num_components).
            y_pred (ndarray): Predicted tensor (batch_size, H, W, num_components).
            index (int): Index of the sample to visualize.
            channel (int): Channel to plot (0-based index). Defaults to 0.
            xcor (ndarray, optional): X-coordinates grid. Defaults to None.
            ycor (ndarray, optional): Y-coordinates grid. Defaults to None.
            cmap (str): Colormap for the output fields. Defaults to 'viridis'.
        """
        # Convert to 0-based index (e.g., Component 1 -> index 0)
        c_idx = channel 
        
        # Generate coordinates if not provided
        if xcor is None or ycor is None:
            L_x, L_y = y_true[index].shape[0], y_true[index].shape[1]
            xcor, ycor = np.meshgrid(np.arange(L_x), np.arange(L_y), indexing='ij')

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        
        # Min/max bounds for the selected component (Ground Truth & Prediction)
        comb = [y_pred[index][:, :, c_idx], y_true[index][:, :, c_idx]]
        _min, _max = np.min(comb), np.max(comb)

        # Left: Input Phase
        ax[0].pcolormesh(xcor, ycor, phase[index].squeeze(), cmap='gray', shading='auto')
        ax[0].set_title('Phase map', fontsize=12)

        # Center: Ground Truth (FEM)
        ax[1].pcolormesh(xcor, ycor, y_true[index][:, :, c_idx], cmap=cmap, vmin=_min, vmax=_max, shading='auto')
        ax[1].set_title(f'FEM', fontsize=12)

        # Right: Prediction (DualEncoderFNO)
        pcm_pred = ax[2].pcolormesh(xcor, ycor, y_pred[index][:, :, c_idx], cmap=cmap, vmin=_min, vmax=_max, shading='auto')
        ax[2].set_title(f'DualEncoderFNO', fontsize=12)

        # Global formatting
        for i in range(3):
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
            ax[i].xaxis.set_tick_params(width=0)
            ax[i].yaxis.set_tick_params(width=0)
            ax[i].set_aspect('equal')
            for spine in ax[i].spines.values():
                spine.set_linewidth(1)

        # Global colorbar for Center and Right plots
        plt.tight_layout()
        fig.subplots_adjust(right=0.9) 
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        colorbar = plt.colorbar(pcm_pred, cax=cax)
        colorbar.ax.tick_params(labelsize=12) 
        colorbar.outline.set_linewidth(1)
        colorbar.set_label('[MPa]', fontsize=12)

        if title is None:
            title = f'FNO predictions - Channel {channel}'
        fig.suptitle(title, y=1.1)
        plt.show()