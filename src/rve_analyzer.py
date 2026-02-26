
import h5py
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.models import FNO
from neuralop.layers.embeddings import SinusoidalEmbedding, GridEmbedding2D

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

            # Official Normalizers - mean/std are per-channel (shape (C,) or broadcastable)
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

        # Permute to Channel-First (CHW) for FNO/Fourier operations
        x_local = x_local.permute(2, 0, 1)   # (channels, H, W)
        y_local = y_local.permute(2, 0, 1)   # (channels, H, W)

        # Apply NeuralOperator normalization (broadcasts across H, W)
        x_local = self.x_normalizer.transform(x_local)
        x_global = self.global_normalizer.transform(x_global)
        y_local = self.y_normalizer.transform(y_local)

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
                 macro_embed_nfreq: int = 6,      # 4-8 is optimal for mechanics
                 embed_type: str = 'nerf',
                 use_positional_grid: bool = True,
                 device: str = 'cpu'):
        super().__init__()

        self.width = width
        self.use_positional_grid = use_positional_grid
        self.nmacro = nmacro
        self.macro_embed_nfreq = macro_embed_nfreq
        self.embed_type = embed_type

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
            embedding_type=embed_type
        ).to(device)

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

        return x.permute(0, 2, 3, 1)   # (B, H, W, out_channels)

