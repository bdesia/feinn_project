from mesher import Mesh2D
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class MicrostructurePool:
    """
    Precomputes a pool of raw microstructures (phase + masks) indexed by fhard.
    """
    def __init__(self,
                 generator: object,
                 fhard_bins: torch.Tensor,
                 meshing: bool = False,
                 delta_x: Optional[float] = 0.10,
                 delta_y: Optional[float] = 0.10,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float64):
        
        self.device = device
        self.dtype = dtype
        self.fhard_bins = fhard_bins.to(device=device, dtype=dtype)
        self.num_bins = len(fhard_bins)

        print(f"Building MicrostructurePool with {self.num_bins} bins...")

        pool_phase = []
        pool_mask_soft = []
        pool_mask_hard = []
        if meshing:
            pool_mesh = []
            
        for i, f in enumerate(self.fhard_bins):
            phase, mask_soft, mask_hard = generator.generate(
                f.item(), device=device, dtype=dtype
            )
            
            pool_phase.append(phase)
            pool_mask_soft.append(mask_soft)
            pool_mask_hard.append(mask_hard)
            if meshing:
                pool_mesh.append(generator.to_mesh(lx=delta_x, ly=delta_y))
                
            if (i + 1) % 5 == 0 or i == self.num_bins - 1:
                print(f"   → {i+1:2d}/{self.num_bins} microstructures generated")

        self.pool_phase = torch.stack(pool_phase)      # (num_bins, C, H, W) raw
        self.pool_mask_soft = torch.stack(pool_mask_soft)
        self.pool_mask_hard = torch.stack(pool_mask_hard)
        if meshing:
            self.pool_mesh = torch.stack(pool_mesh)
        
        print(f"MicrostructurePool ready | Shape: {self.pool_phase.shape}")

    def get_rves(self, tags: torch.Tensor) -> torch.Tensor:
        """tags (nelem, ngp) → raw RVEs (nelem*ngp, C, H, W)"""
        return self.pool_phase[tags.view(-1).long()]

    def get_masks(self, tags: torch.Tensor):
        """Return soft and hard masks for given tags"""
        idx = tags.view(-1).long()
        return self.pool_mask_soft[idx], self.pool_mask_hard[idx]

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
        
        self.phase_tensor = None
        self.mask_soft = None
        self.mask_hard = None

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

    def resize(self, new_resolution: int) -> None:
        """
        Resizes the generated microstructure to a new resolution using nearest-neighbor interpolation.
        Interpolates only the phase tensor and re-derives masks to ensure exact boolean partitions.

        Args:
            new_resolution (int): Target resolution (new_resolution x new_resolution).
        """
        if self.phase_tensor is None or self.mask_soft is None or self.mask_hard is None:
            raise RuntimeError("No microstructure generated yet. Call generate() first.")
        
        if new_resolution <= 0:
            raise ValueError("new_resolution must be a positive integer.")
            
        if self.res == new_resolution:
            print(f"Already at resolution {new_resolution}x{new_resolution}. No changes made.")
            return

        # Prepare tensor for F.interpolate: (1, H, W) -> (1, 1, H, W)
        inp = self.phase_tensor.unsqueeze(0)
        
        # Apply nearest interpolation to preserve exact 0.0/1.0 values
        out = F.interpolate(inp, size=(new_resolution, new_resolution), mode='nearest')
        
        # Update phase_tensor: (1, 1, new_res, new_res) -> (1, new_res, new_res)
        self.phase_tensor = out.squeeze(0)
        
        # Re-derive masks to guarantee mask_hard + mask_soft == 1.0
        self.mask_hard = self.phase_tensor.squeeze(0)
        self.mask_soft = 1.0 - self.mask_hard
        
        # Update internal resolution
        old_res = self.res
        self.res = new_resolution
        
        print(f"Microstructure resized: {old_res}x{old_res} → {new_resolution}x{new_resolution}")

    def plot(self, title: str = "RVE Microstructure") -> None:
        """
        Visualizes the generated RVE and its geometric phase masks.
        """
        
        # Convert to numpy for matplotlib
        phase_np = self.phase_tensor.squeeze(0).cpu().numpy()
        ms_np = self.mask_soft.cpu().numpy()
        mh_np = self.mask_hard.cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Phase Tensor (As seen by the FNO)
        im0 = ax[0].imshow(phase_np, cmap='viridis', origin='lower')
        ax[0].set_title(f"{title}\n(fhard: {phase_np.mean():.3f})")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        # 2. Soft Mask
        im1 = ax[1].imshow(ms_np, cmap='gray_r', origin='lower')
        ax[1].set_title("Soft Mask (Matrix)")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        # 3. Hard Mask
        im2 = ax[2].imshow(mh_np, cmap='gray_r', origin='lower')
        ax[2].set_title("Hard Mask (Inclusion)")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        for a in ax:
            a.set_axis_off()
        
        plt.tight_layout()
        plt.show()

    def to_mesh(self,
                lx: float,
                ly: float,
                device: str = "cpu",
                dtype: torch.dtype = torch.float64,
                verbose: bool = False) -> Tuple['Mesh2D', list, list]:
        """
        Convert voxel-based microstructure into a coarser Q4 finite element mesh.
        
        - Mesh resolution = voxel_resolution / 2  (2×2 voxels per element)
        - Element material assigned by majority vote in each 2x2 voxel block.
        
        Returns:
            mesh, pairs_left_right, pairs_bottom_top
        """
        from mesher import UniformQuadMesh2D, get_periodic_pairs_kdtree

        # Ensure microstructure exists
        if self.mask_hard is None:
            self.generate(0.3, device=device, dtype=dtype)

        # Coarser mesh: half resolution (2 voxels per element)
        nelem_by_side = int(0.5 * self.res)
        mesh = UniformQuadMesh2D(lx=lx, ly=ly, nx=nelem_by_side, ny=nelem_by_side, elem_type='Q4')
        mesh.compute()

        # Assign material groups - 2x2 voxel to 1 element mapping
        mask_hard_np = self.mask_hard.cpu().numpy()  # shape: (res, res)
        
        hard_elems = []
        soft_elems = []
        
        step = 2  # voxels per element
        for j in range(nelem_by_side):      # mesh row
            for i in range(nelem_by_side):  # mesh column
                # Extract 2x2 block from voxel grid
                block = mask_hard_np[j*step : (j+1)*step, i*step : (i+1)*step]
                
                # Majority vote
                hard_fraction = block.mean()
                elem_id = j * nelem_by_side + i + 1   # 1-based, row-major
                
                if hard_fraction > 0.5:
                    hard_elems.append(elem_id)
                else:
                    soft_elems.append(elem_id)

        mesh.add_element_group('hard_s', hard_elems)
        mesh.add_element_group('soft_s', soft_elems)

        mesh.add_node_group('fixed_n', [1])
        
        # Periodic boundary pairs
        left_right_pairs, bottom_top_pairs = get_periodic_pairs_kdtree(
            mesh, delta_x=lx, delta_y=ly
        )

        if verbose:
            print(f"to_mesh() → {nelem_by_side}×{nelem_by_side} Q4 elements "
                f"({self.res}×{self.res} voxels) | "
                f"hard_s: {len(hard_elems)} | soft_s: {len(soft_elems)}")

        return mesh, left_right_pairs, bottom_top_pairs
    
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
        self.mask_hard = (dist <= radius).to(dtype)
        self.mask_soft = 1.0 - self.mask_hard
        self.phase_tensor = self.mask_hard.unsqueeze(0)
        
        return self.phase_tensor, self.mask_soft, self.mask_hard

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
        self.mask_hard = ( (dist_c <= radius) | (dist_bl <= radius) | 
                      (dist_br <= radius) | (dist_tl <= radius) | 
                      (dist_tr <= radius) ).to(dtype)
        
        self.mask_soft = 1.0 - self.mask_hard
        self.phase_tensor = self.mask_hard.unsqueeze(0)
        
        return self.phase_tensor, self.mask_soft, self.mask_hard

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
        self.phase_tensor = F.interpolate(coarse_t, size=(self.res, self.res), mode='nearest').squeeze(0)
        
        self.mask_hard = (self.phase_tensor.squeeze(0) > 0.5).to(dtype) 
        self.mask_soft = 1.0 - self.mask_hard
        
        return self.phase_tensor, self.mask_soft, self.mask_hard
    
class Layered(MicrostructureGenerator):
    """
    Generates a layered (striped) microstructure with controllable orientation and frequency.
    """
    
    def __init__(self, resolution: int = 96, angle: float = 0.0, n_layers: int = 8):
        """
        Args:
            resolution (int): Grid size (res x res).
            angle (float): Lamination angle in degrees (0° = horizontal, 90° = vertical).
            n_layers (int): Number of repeating hard/soft layer pairs.
        """
        super().__init__(resolution)
        self.angle = angle
        self.n_layers = max(1, n_layers)

    def generate(self, fhard: float, device: str = "cpu", dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates the Representative Volume Element (RVE).
        
        Args:
            fhard (float): Volume fraction of the hard phase [0.0, 1.0].
            device (str): Computation device ('cpu' or 'cuda').
            dtype (torch.dtype): Data type for the tensors.
            
        Returns:
            Tuple containing phase_tensor, mask_soft, and mask_hard.
        """
        # Clamp volume fraction to valid range
        fhard = max(0.0, min(1.0, fhard))

        # Calculate normal vector for layer orientation
        # 0° -> nx=0, ny=1 (variation along Y -> horizontal layers)
        # 90° -> nx=1, ny=0 (variation along X -> vertical layers)
        theta = torch.tensor(self.angle * torch.pi / 180.0, device=device, dtype=dtype)
        nx = torch.sin(theta)
        ny = torch.cos(theta)

        # Create an evenly spaced periodic grid in [0, 1)
        coords = torch.arange(self.res, device=device, dtype=dtype) / self.res
        Y, X = torch.meshgrid(coords, coords, indexing='ij')

        # Project coordinates onto the normal vector and scale by frequency
        proj = X * nx + Y * ny
        scaled = proj * self.n_layers
        
        # CRITICAL FIX: Use modulo 1.0 instead of torch.frac()
        # This properly handles negative projections by wrapping them to [0, 1)
        frac = scaled % 1.0

        # Generate binary masks based on the volume fraction threshold
        self.mask_hard = (frac < fhard).to(dtype)
        self.mask_soft = 1.0 - self.mask_hard
        self.phase_tensor = self.mask_hard.unsqueeze(0)

        return self.phase_tensor, self.mask_soft, self.mask_hard