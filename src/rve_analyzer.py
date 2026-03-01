
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
from neuralop.models import FNO
from neuralop.layers.embeddings import SinusoidalEmbedding, GridEmbedding2D
from neuralop.losses import LpLoss


class RVEDataset(Dataset):
    """
    Optimized Dataset for Dual-Encoder FNO Training on RVE Data.
    - Lazy loading HDF5 with swmr=True for multiprocessing safety.
    - Channel-first (CHW) layout consistent with NeuralOperator library expectations.
    - Uses UnitGaussianNormalizers with precomputed stats
    """

    def __init__(self, h5_path: str | Path, split: str = 'train'):
        self.h5_path = Path(h5_path)
        self.split = split
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
        # Open file once per worker process (crucial for DataLoader num_workers > 0)
        if self.archive is None:
            self.archive = h5py.File(self.h5_path, 'r', swmr=True)

        data = self.archive[self.split]

        # Load from HDF5 (stored as HWC)
        x_local = torch.from_numpy(data['x_local'][idx]).float()   # (H, W, phase + nstatev + 3 stresses + 3 dstrain)
        x_global = torch.from_numpy(data['x_global'][idx]).float() # (3 strain + nprops,)
        y_local  = torch.from_numpy(data['y_local'][idx]).float()  # (H, W, 3 stresses + nstatev)

        # Apply NeuralOperator normalization (broadcasts across H, W)
        x_local = self.x_normalizer.transform(x_local)
        x_global = self.global_normalizer.transform(x_global)
        y_local = self.y_normalizer.transform(y_local)

        # Permute to Channel-First (CHW) for FNO/Fourier operations
        x_local = x_local.permute(2, 0, 1).contiguous()   # (channels, H, W)
        y_local = y_local.permute(2, 0, 1).contiguous()   # (channels, H, W)

        return x_local, x_global, y_local

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
      (state-of-the-art conditioning; replaces old concatenation + mixer).
    - FNO Core: Multi-layer Fourier Neural Operator.
    - Output Projection: Maps back to target fields (stresses, damage, plastic strain, etc.).

    This is the 2025-2026 recommended architecture for nonlinear multiscale solid mechanics.
    SinusoidalEmbedding + FiLM together provide superior accuracy and training stability.
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 3,
                 nmacro: int = 7,
                 modes: int = 16,
                 width: int = 64,
                 n_layers: int = 4,
                 macro_embed_nfreq: int = 6,
                 use_positional_grid: bool = True,
                 device: str = 'cpu'):
        super().__init__()

        self.width = width
        self.use_positional_grid = use_positional_grid
        self.nmacro = nmacro
        self.macro_embed_nfreq = macro_embed_nfreq

        # =============== Spatial (Micro) Branch ===============
        spatial_in = in_channels + 2 if use_positional_grid else in_channels
        self.spatial_lift = nn.Conv2d(spatial_in, width, kernel_size=1, bias=True)

        if use_positional_grid:
            self.grid_embed = GridEmbedding2D(
                in_channels=in_channels,
                grid_boundaries=[[0., 1.], [0., 1.]]
            )

        # =============== Global (Macro) Branch - Sinusoidal Embedding ===============
        self.global_embed = SinusoidalEmbedding(
            in_channels=nmacro,
            num_freqs=macro_embed_nfreq,
            embedding_type='nerf'
        )

        self.global_embed_dim = nmacro * 2 * macro_embed_nfreq

        # =============== FiLM Conditioning (Scale + Shift) ===============
        self.film_gamma = nn.Linear(self.global_embed_dim, width)
        self.film_beta  = nn.Linear(self.global_embed_dim, width)

        # =============== FNO Core ===============
        self.fno = FNO(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=width,
            out_channels=width,
            n_layers=n_layers
        )

        # =============== Output Projection ===============
        self.project = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        """
        x_local:  (B, in_channels, H, W)  → microstructural field
        x_global: (B, nmacro)             → macroscopic loading / parameters
        """
        B, _, H, W = x_local.shape

        # Spatial Branch
        if self.use_positional_grid:
            x_local = self.grid_embed(x_local)
        spatial = self.spatial_lift(x_local)                    # (B, width, H, W)

        # Global Branch - Sinusoidal Embedding
        global_vec = self.global_embed(x_global)                # (B, global_embed_dim)

        # FiLM Conditioning
        gamma = self.film_gamma(global_vec).view(B, self.width, 1, 1)
        beta  = self.film_beta(global_vec).view(B, self.width, 1, 1)
        x = gamma * spatial + beta                              # (B, width, H, W)

        # FNO Core + Output
        x = self.fno(x)
        x = self.project(x)

        return x    # (B, out_channels, H, W, )


class Trainer:
    """
    Trainer for DualEncoderFNO.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: str | Path = "checkpoints",
        patience: int = 25,
        min_delta: float = 1e-6,
        verbose: bool = True
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.loss_fn = LpLoss(
            d=2, 
            p=2, 
            relative=True, 
            reduce_dims=[-1, -2]
        ).to(self.device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_model_path = self.save_dir / "best_DualEncoderFNO.pth"

    def _process_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moves batch data to the target device asynchronously."""
        x_local, x_global, y = batch
        return (
            x_local.to(self.device, non_blocking=True),
            x_global.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True)
        )

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """Executes a single training epoch and returns the average loss."""
        self.model.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch in tqdm(dataloader, desc="Train", leave=False, disable=not self.verbose):
            x_local, x_global, y = self._process_batch(batch)
            batch_size = x_local.size(0)

            self.optimizer.zero_grad()

            # Forward pass: Output shape (B, C, H, W)
            pred = self.model(x_local, x_global)                    

            loss = self.loss_fn(pred, y)
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # Weight loss by actual batch size
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        # Step standard schedulers (ignoring ReduceLROnPlateau)
        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        return epoch_loss / total_samples

    def validate_one_epoch(self, dataloader: DataLoader) -> float:
        """Executes a single validation epoch and returns the average loss."""
        self.model.eval()
        epoch_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Valid", leave=False, disable=not self.verbose):
                x_local, x_global, y = self._process_batch(batch)
                batch_size = x_local.size(0)
                
                pred = self.model(x_local, x_global)
                loss = self.loss_fn(pred, y)
                
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

        return epoch_loss / total_samples

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int = 150) -> Dict[str, list]:
        """Trains the model for a specified number of epochs with early stopping."""
        print(f"DualEncoderFNO Training – {n_epochs} epochs\n")

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss   = self.validate_one_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Step plateau scheduler if used
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            # Check for best model and save
            is_best = val_loss < (self.best_loss - self.min_delta)
            if is_best:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                }, self.best_model_path)
                if self.verbose:
                    print(f"   Best model saved (Epoch {epoch})")
            else:
                self.counter += 1

            # Logging
            if self.verbose or epoch % 5 == 0 or epoch == n_epochs:
                print(f"Epoch {epoch:3d}/{n_epochs} | "
                      f"Train L2: {train_loss:.6f} | "
                      f"Val L2: {val_loss:.6f} | "
                      f"Best: {self.best_loss:.6f}")

            # Early stopping check
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"\n Training finished. Best epoch: {self.best_epoch} (L2 = {self.best_loss:.6f})")
        return self.history

    def plot_history(self, title: str = "DualEncoderFNO Training - Relative L2 Loss"):
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

    def load_best_model(self):
        """Loads the weights of the best performing model from disk."""
        if self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model loaded (Epoch {checkpoint['epoch']}, L2 = {checkpoint['best_loss']:.6f})")
        else:
            print("No saved model found")