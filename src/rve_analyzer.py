
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
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

        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.use_positional_grid = use_positional_grid
        self.n_macro = n_macro

        self.film_per_layer = film_per_layer
        self.film_mlp_layers = film_mlp_layers
        
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
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
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
        avg_loss = epoch_loss / total_samples

        return avg_loss

    @torch.no_grad()
    def _validate_one_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Run one batch and return loss and size """
        # Desactive grad
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
        avg_loss = epoch_loss / total_samples

        # Step plateau scheduler if used
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return avg_loss

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
            if self.counter >= self.patience:
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
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x_local, x_global):
        """
        Performs inference for a single batch of data.
        """
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
    
    @torch.no_grad()
    def predict_dataset(self, dataloader):
        """
        Generates predictions for an entire dataloader.
        Returns numpy arrays of ground-truth and predictions ready for plotting.
        """
        all_preds = []
        all_truths = []
        
        for batch in dataloader:
            x_local, x_global, y_local = batch
            
            # Denormalized predictions
            preds = self.predict(x_local, x_global)
            
            # Denormalized ground-truth
            if self.y_normalizer is not None:
                truths = self.y_normalizer.inverse_transform(y_local.to(self.device)).cpu().numpy()
            else:
                truths = y_local.numpy()
                
            all_preds.append(preds)
            all_truths.append(truths)
            
        return np.concatenate(all_truths, axis=0), np.concatenate(all_preds, axis=0)


class RVEVisualizer:
    """
    Class to visualize and compare Ground-Truth (FEM) results 
    against model predictions (ML).
    """
    
    @staticmethod
    def plot_cross_sections(y_true, y_pred, index, hline=None, vline=None, variable='Stress'):
        """
        Plots values along horizontal and vertical cross-sectional lines.
        Adapts dynamically to the RVE grid size.
        """
        tt = y_true[index]  # FEM
        zz = y_pred[index]  # ML
        
        # Get grid size dynamically (e.g., 96)
        L_x, L_y = tt.shape[0], tt.shape[1]
        
        # If no cut lines are specified, take the center of the RVE
        if hline is None: hline = L_y // 2
        if vline is None: vline = L_x // 2

        # Extract horizontal lines
        lt11, lz11 = tt[hline, :, 0], zz[hline, :, 0]
        lt22, lz22 = tt[hline, :, 1], zz[hline, :, 1]
        lt33, lz33 = tt[hline, :, 2], zz[hline, :, 2]
        horizontal = np.array([[lt11, lz11], [lt22, lz22], [lt33, lz33]])

        # Extract vertical lines
        lt11_v, lz11_v = tt[:, vline, 0], zz[:, vline, 0]
        lt22_v, lz22_v = tt[:, vline, 1], zz[:, vline, 1]
        lt33_v, lz33_v = tt[:, vline, 2], zz[:, vline, 2]
        vertical = np.array([[lt11_v, lz11_v], [lt22_v, lz22_v], [lt33_v, lz33_v]])

        fig, ax = plt.subplots(3, 2, figsize=(20, 27))
        x_axis_hor = np.arange(L_x)
        x_axis_ver = np.arange(L_y)

        for i in range(ax.shape[0]):
            # Horizontal Plots
            ax[i, 0].plot(x_axis_hor, horizontal[i][0], linewidth=6, color='red', label='FEM-Hor')      
            ax[i, 0].plot(x_axis_hor, horizontal[i][1], linewidth=4, color='black', label='ML-Hor', linestyle='dashed')

            ax[i, 0].xaxis.set_tick_params(which='major', size=10, width=3, direction='in', top=True)
            ax[i, 0].xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', top=True)
            ax[i, 0].yaxis.set_tick_params(which='major', size=10, width=3, direction='in', right=True)
            ax[i, 0].yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', right=True)
            
            ax[i, 0].yaxis.set_major_locator(ticker.MaxNLocator(6))
            ax[i, 0].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i, 0].yaxis.set_minor_locator(AutoMinorLocator())
            ax[i, 0].yaxis.set_label_coords(-.1, .5)
            
            ax[i, 0].set_xlabel('Grid X', labelpad=20) 
            ax[i, 0].set_ylabel(f'{variable} Component {i+1}', labelpad=20)
            ax[i, 0].set_xlim(0, L_x - 1)
            ax[i, 0].legend()
 
            # Vertical Plots
            ax[i, 1].plot(x_axis_ver, vertical[i][0], linewidth=6, color='red', label='FEM-Vert')  
            ax[i, 1].plot(x_axis_ver, vertical[i][1], linewidth=4, color='black', label='ML-Vert', linestyle='dashed')

            ax[i, 1].xaxis.set_tick_params(which='major', size=10, width=3, direction='in', top=True)
            ax[i, 1].xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', top=True)
            ax[i, 1].yaxis.set_tick_params(which='major', size=10, width=3, direction='in', right=True)
            ax[i, 1].yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='in', right=True)
            
            ax[i, 1].yaxis.set_major_locator(ticker.MaxNLocator(6))
            ax[i, 1].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i, 1].yaxis.set_minor_locator(AutoMinorLocator())
            ax[i, 1].yaxis.set_label_coords(-.12, .5)

            ax[i, 1].set_xlabel('Grid Y', labelpad=15) 
            ax[i, 1].set_ylabel(f'{variable} Component {i+1}', labelpad=15)
            ax[i, 1].set_xlim(0, L_y - 1)
            ax[i, 1].legend()

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_contours(y_true, y_pred, index, xcor=None, ycor=None, cmap='Reds'):
        """
        Component-wise contour map comparing Ground Truth (Left) vs FNO (Right).
        """
        R = y_pred[index].shape[-1]
        
        # Generate coordinates if not provided
        if xcor is None or ycor is None:
            L_x, L_y = y_true[index].shape[0], y_true[index].shape[1]
            xcor, ycor = np.meshgrid(np.arange(L_x), np.arange(L_y), indexing='ij')

        fig, ax = plt.subplots(R, 2, figsize=(16, 7 * (R + 0.5)))
        
        comb = [y_pred[index], y_true[index]]
        _min, _max = np.min(comb), np.max(comb)
        cbarwidth = 0.03

        for j in range(R):
            # Ground Truth (Left)
            ax1 = ax[j, 0]
            pl1 = ax1.pcolormesh(xcor, ycor, y_true[index][:, :, j], cmap=cmap, vmin=_min, vmax=_max, shading='auto')
            ax1.set_title(f'FEM - Component {j+1}', fontsize=16)
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
            ax1.xaxis.set_tick_params(width=0)
            ax1.yaxis.set_tick_params(width=0)
            ax1.set_aspect('equal')
            for spine in ax1.spines.values():
                spine.set_linewidth(3)

            # Prediction (Right)
            axx2 = ax[j, 1]
            pcm2 = axx2.pcolormesh(xcor, ycor, y_pred[index][:, :, j], cmap=cmap, vmin=_min, vmax=_max, shading='auto')
            axx2.set_title(f'FNO ML - Component {j+1}', fontsize=16)
            axx2.set_yticklabels([])
            axx2.set_xticklabels([])
            axx2.xaxis.set_tick_params(width=0)
            axx2.yaxis.set_tick_params(width=0)
            axx2.set_aspect('equal')
            for spine in axx2.spines.values():
                spine.set_linewidth(3)

        # Global colorbar
        pos1 = ax[0, 1].get_position()
        cax = fig.add_axes([0.94, 0.15, cbarwidth, 0.7])
        colorbar = plt.colorbar(pl1, cax=cax)
        colorbar.ax.tick_params(labelsize=20) 
        colorbar.outline.set_linewidth(3)
        
        plt.show()