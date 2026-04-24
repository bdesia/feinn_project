import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple

class MicrostructureGenerator(ABC):
    """
    Abstract base class for RVE microstructure generators.
    Designed for multiscale FEM and Neural Operator integration.
    """
    def __init__(self, resolution: int = 96):
        """
        Args:
            resolution (int): Grid size for the Neural Operator (must match DataLoader).
        """
        self.res = resolution

    @abstractmethod
    def generate(self, fhard: float, device: str = "cpu", dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate the RVE microstructure for a given volume fraction.
        
        Returns:
            phase_tensor: (1, res, res) Image with 0.0 (soft) and 1.0 (hard).
            mask_soft: (res, res) Float mask for matrix phase.
            mask_hard: (res, res) Float mask for inclusion phase.
        """
        pass

    def plot(self, fhard: float, title: str = "RVE Microstructure") -> None:
        """
        Visualizes the generated RVE and its geometric phase masks.
        """
        phase, m_soft, m_hard = self.generate(fhard)
        
        # Convert to numpy for matplotlib
        phase_np = phase.squeeze(0).cpu().numpy()
        ms_np = m_soft.cpu().numpy()
        mh_np = m_hard.cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Phase Tensor (As seen by the FNO)
        im0 = ax[0].imshow(phase_np, cmap='viridis', origin='lower')
        ax[0].set_title(f"{title}\n(fhard: {phase_np.mean():.3f})")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        # 2. Soft Mask
        im1 = ax[1].imshow(ms_np, cmap='gray', origin='lower')
        ax[1].set_title("Soft Mask (Matrix)")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        # 3. Hard Mask
        im2 = ax[2].imshow(mh_np, cmap='gray', origin='lower')
        ax[2].set_title("Hard Mask (Inclusion)")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        for a in ax:
            a.set_axis_off()
        
        plt.tight_layout()
        plt.show()


# ==========================================
# IMPLEMENTATIONS
# ==========================================

class CentralFiber(MicrostructureGenerator):
    """
    Microstructure with a single central circular inclusion.
    Normalized geometry ([0,1]x[0,1]) independent of resolution.
    """
    def generate(self, fhard: float, device: str = "cpu", dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Radius for a single circle: Area = PI * r^2 = fhard
        radius = np.sqrt(fhard / np.pi)
        
        # Native PyTorch grid generation
        coords = torch.linspace(0, 1.0, self.res, device=device, dtype=dtype)
        Y, X = torch.meshgrid(coords, coords, indexing='ij')
        
        # Distance to center (0.5, 0.5)
        dist = torch.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
        
        # Masks generation without boolean casting issues
        mask_hard = (dist <= radius).to(dtype)
        mask_soft = 1.0 - mask_hard
        phase_tensor = mask_hard.unsqueeze(0)
        
        return phase_tensor, mask_soft, mask_hard


class CentralCornerFiber(MicrostructureGenerator):
    """
    Periodic RVE with central inclusion and corner quarters.
    Equivalent to 2 full circles for volume fraction calculation.
    """
    def generate(self, fhard: float, device: str = "cpu", dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Radius for 2 equivalent circles: Area = 2 * PI * r^2 = fhard
        radius = np.sqrt(fhard / (2 * np.pi))
        
        coords = torch.linspace(0, 1.0, self.res, device=device, dtype=dtype)
        Y, X = torch.meshgrid(coords, coords, indexing='ij')
        
        # Distances to center and 4 corners
        dist_c = torch.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
        dist_bl = torch.sqrt(X**2 + Y**2)
        dist_br = torch.sqrt((X - 1.0)**2 + Y**2)
        dist_tl = torch.sqrt(X**2 + (Y - 1.0)**2)
        dist_tr = torch.sqrt((X - 1.0)**2 + (Y - 1.0)**2)
        
        # Union of geometries (Logical OR)
        mask_hard = ( (dist_c <= radius) | (dist_bl <= radius) | 
                      (dist_br <= radius) | (dist_tl <= radius) | 
                      (dist_tr <= radius) ).to(dtype)
        
        mask_soft = 1.0 - mask_hard
        phase_tensor = mask_hard.unsqueeze(0)
        
        return phase_tensor, mask_soft, mask_hard


class RandomBlocks(MicrostructureGenerator):
    """
    Block-based microstructure with exact volume fraction assignment.
    """
    def __init__(self, resolution: int = 96, n_blocks: int = 8):
        super().__init__(resolution)
        self.n_blocks = n_blocks

    def generate(self, fhard: float, device: str = "cpu", dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_total = self.n_blocks**2
        n_hard = round(fhard * n_total)
        
        # Generate random indices using numpy
        coarse = torch.zeros((self.n_blocks, self.n_blocks), device=device, dtype=dtype)
        idx = torch.randperm(self.n_blocks**2)[:n_hard]
        coarse.view(-1)[idx] = 1.0
        
        # Convert to tensor and upsample to full resolution natively in PyTorch
        coarse_t = coarse.unsqueeze(0).unsqueeze(0)
        phase_tensor = F.interpolate(coarse_t, size=(self.res, self.res), mode='nearest').squeeze(0)
        
        mask_hard = (phase_tensor.squeeze(0) > 0.5).to(dtype) 
        mask_soft = 1.0 - mask_hard

        return phase_tensor, mask_soft, mask_hard