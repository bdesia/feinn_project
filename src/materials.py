
from abc import ABC, abstractmethod
import json
from typing import Tuple, Optional, Dict, Callable

import torch
# from torch.func import vmap, jacrev
# from torchgen import model

class MaterialBase(ABC):
    """
    Abstract base class for all material models.

    Attributes
    ----------
    n_state : int
        Number of internal state variables per integration point.
    is_vectorized : bool
        Indicates whether the material is operating in vectorized mode
        (multiple elements and Gauss points simultaneously).
    """
    def __init__(self, n_state: int = 0, 
                 dtype: torch.dtype = torch.float64, 
                 device: str = 'cpu',
                 tag: Optional[int] = None):

        self.n_state = n_state
        self.dtype = dtype
        self.device = device
        self.tag = tag

    def set_tag(self, tag):
        self.tag = tag
        return self
    
    def to(self, dtype=None, device=None):
        """Update device and dtype of the material."""
        if dtype: self.dtype = dtype
        if device: self.device = device
        
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(dtype=self.dtype, device=self.device))
        return self

    def init_state(self, nelem: int, ngp2: int, **kwargs) -> torch.Tensor:
        """
        Initialize material history variables.

        Args:
            nelem: Number of elements in the batch.
            ngp2: Number of Gauss points per element.
            **kwargs: Optional context dict (e.g., coords, temperature).
            
        Returns:
            State tensor of shape (nelem, ngp2, n_state).
        """
        # Default behavior: return zero tensor for standard materials
        return torch.zeros((nelem, ngp2, self.n_state), device=self.device, dtype=self.dtype)

    @abstractmethod
    def update_state(
        self,
        strain: torch.Tensor,
        state_old: torch.Tensor,
        isTangent: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return (stress, state_new, tangent). Tangent = None if isTangent=False."""
        pass

class LinearElastic(MaterialBase):
    """
    Linear isotropic elastic material under plane strain assumption (2D).

    Uses Voigt notation: strain/stress vector = [ε_xx, ε_yy, 2ε_xy].
    """
    def __init__(self, emod: float, nu: float, tag: Optional[int] = None, **kwargs):
        super().__init__(n_state=0, tag=tag, **kwargs)
        """
        Parameters
        ----------
        emod  : float
            Young's modulus (Pa).
        nu : float
            Poisson's ratio (dimensionless).
        """
        if emod <= 0:
            raise ValueError("Young's modulus (E) must be positive.")
        if not (-1 <= nu < 0.5):
            raise ValueError("Poisson's ratio (nu) must be in [-1, 0.5).")

        self.emod = emod
        self.nu = nu

        # Pre-computed factors for plane strain constitutive matrix
        self._factor1 = emod * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        self._factor2 = emod * nu / ((1 + nu) * (1 - 2 * nu))
        self.G = emod / (2 * (1 + nu))

    def _get_constitutive_matrix(self) -> torch.Tensor:
        """
        Returns the constant elastic constitutive matrix (3x3).

        Returns
        -------
        torch.Tensor
            Elastic stiffness matrix in Voigt notation.
        """
        C11 = self._factor1
        C12 = self._factor2
        C33 = self.G
        return torch.tensor([[C11, C12, 0.0],
                             [C12, C11, 0.0],
                             [0.0,  0.0, C33]], dtype=self.dtype, device=self.device)

    def update_state(self, strain, state_old, isTangent=True):
        C = self._get_constitutive_matrix()
        stress = torch.einsum('ij,...j->...i', C, strain)
        
        ddsdde = None
        if isTangent:
            ddsdde = C.view(1, 1, 3, 3).expand(strain.shape[0], strain.shape[1], 3, 3)
        
        return stress, state_old, ddsdde

class LinearElasticPlaneStress(MaterialBase):
    """
    Linear isotropic elastic material under plane stress assumption (2D).

    Uses Voigt notation: strain/stress vector = [ε_xx, ε_yy, 2ε_xy].
    """
    def __init__(self, emod: float, nu: float, tag: Optional[int] = None, **kwargs):
        super().__init__(n_state=0, tag=tag, **kwargs)
        """
        Parameters
        ----------
        emod  : float
            Young's modulus (Pa).
        nu : float
            Poisson's ratio (dimensionless).
        """
        if emod <= 0:
            raise ValueError("Young's modulus (E) must be positive.")
        if not (-1 <= nu < 0.5):
            raise ValueError("Poisson's ratio (nu) must be in [-1, 0.5).")

        self.emod = emod
        self.nu = nu

        self._factor1 = emod / (1 - nu**2)

    def _get_constitutive_matrix(self) -> torch.Tensor:
        C11 = self._factor1
        C12 = self._factor1 * self.nu
        C33 = self._factor1 * (1 - self.nu) / 2
        return torch.tensor([[C11, C12, 0.0],
                             [C12, C11, 0.0],
                             [0.0,  0.0, C33]], device=self.device, dtype=self.dtype)

    def update_state(self, strain, state_old, isTangent=True):
        C = self._get_constitutive_matrix()
        stress = torch.einsum('ij,...j->...i', C, strain)
        
        ddsdde = None
        if isTangent:
            ddsdde = C.view(1, 1, 3, 3).expand(strain.shape[0], strain.shape[1], 3, 3)
        
        return stress, state_old, ddsdde

class NexpElastic(MaterialBase):
    """
    Nonlinear isotropic compressible elastic material under plane strain (2D).
    Potential: w(ε) = (9/2) k (tr(ε)/3)^2 + e0 * (s0 / (1+m)) * (e_eq / e0)^{1+m} 
    from Yvonnet et al. (2009), Eqs. (39)-(40).
    Voigt notation: [ε_xx, ε_yy, 2ε_xy]
    """

    def __init__(self, k: float, e0: float, s0: float, m: float, tag: Optional[int] = None, **kwargs):
        super().__init__(n_state=0, tag=tag, **kwargs)

        if k <= 0 or e0 <= 0 or s0 <= 0:
            raise ValueError("k, e0, s0 must be positive.")
        if not (0 <= m <= 1):
            raise ValueError("m must be between 0 and 1.")

        self.k = k
        self.e0 = e0
        self.s0 = s0
        self.m = m

        self.P = torch.tensor([[2/3., -1/3., 0.0],
                               [-1/3., 2/3., 0.0],
                               [0.0,   0.0,  0.5]], dtype=self.dtype, device=self.device)

        self.vol_matrix = torch.tensor([[1.0, 1.0, 0.0],
                                        [1.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0]], dtype=self.dtype, device=self.device)

    def _get_kinematics(self, strain: torch.Tensor):
        """Calculates deviatoric components, invariants, and beta for reuse."""
        exx, eyy, gxy = strain.unbind(-1)
        tre = exx + eyy
        em = tre / 3.0

        edxx = exx - em
        edyy = eyy - em
        edxy = gxy / 2.0

        # Corrección: Se incluye em**2 equivalente a edzz**2
        eeq_sq = (2.0/3.0) * (edxx**2 + edyy**2 + em**2 + 2*edxy**2) 
        eeq = torch.sqrt(eeq_sq + 1e-24)

        ratio = eeq / self.e0
        powered = ratio ** (self.m - 1.0)
        beta = (2.0 / 3.0) * (self.s0 / self.e0) * powered

        return tre, edxx, edyy, edxy, eeq, beta

    def get_constitutive_matrix(self, strain: torch.Tensor) -> torch.Tensor:
        """Tangent stiffness matrix (consistent with stress)."""
        _, edxx, edyy, edxy, eeq, beta = self._get_kinematics(strain)

        gamma = beta * (self.m - 1.0) * (2.0 / 3.0) / (eeq**2 + 1e-24)

        ed_v = torch.stack([edxx, edyy, edxy], dim=-1)
        outer = ed_v.unsqueeze(-1) * ed_v.unsqueeze(-2)

        C_dev = beta.unsqueeze(-1).unsqueeze(-1) * self.P + gamma.unsqueeze(-1).unsqueeze(-1) * outer
        C_vol = self.k * self.vol_matrix

        return C_vol + C_dev

    def update_state(self, strain, state_old, isTangent=True):
        tre, edxx, edyy, edxy, _, beta = self._get_kinematics(strain)

        vol = self.k * tre
        sxx = beta * edxx
        syy = beta * edyy
        sxy = beta * edxy

        stress = torch.stack([vol + sxx, vol + syy, sxy], dim=-1)

        ddsdde = None
        if isTangent:
            # Corrección: C ya tiene dimensiones (..., 3, 3), no requiere .view().expand()
            ddsdde = self.get_constitutive_matrix(strain) 

        return stress, state_old, ddsdde

class NLElasticMatrix(MaterialBase):
    """
    Non-linear isotropic elastic material for the matrix
    Stress tensor: T = Km (tr E) I + Gm(E^D) E^D
    with Gm(E^D) = α₁ / (α₂ + ||E^D||₂)
    
    Plane strain assumption (2D), Voigt notation: [ε_xx, ε_yy, 2ε_xy].
    """

    def __init__(self, Km: float, alpha1: float, alpha2: float, tag: Optional[int] = None, **kwargs):
        super().__init__(n_state=0, tag=tag, **kwargs)

        if Km <= 0 or alpha1 <= 0:
            raise ValueError("Km and alpha1 must be positive.")
        if alpha2 < 0:
            raise ValueError("alpha2 must be non-negative.")

        self.Km = Km
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.P = torch.tensor([[2/3., -1/3., 0.0],
                               [-1/3., 2/3., 0.0],
                               [0.0,   0.0,  0.5]],
                              dtype=self.dtype, device=self.device)
        
        self.vol_matrix = torch.tensor([[1.0, 1.0, 0.0],
                                        [1.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0]],
                                       dtype=self.dtype, device=self.device)

    def _get_kinematics(self, strain: torch.Tensor):
        """Calculates volumetric strain, deviatoric components, ||E^D||₂ and Gm."""
        exx, eyy, gxy = strain.unbind(-1)
        tre = exx + eyy
        em = tre / 3.0

        edxx = exx - em
        edyy = eyy - em
        edxy = gxy / 2.0

        # ||E^D||₂ (Frobenius norm including ε_zz = 0)
        normED_sq = edxx**2 + edyy**2 + em**2 + 2 * edxy**2
        normED = torch.sqrt(normED_sq + 1e-24)

        Gm = self.alpha1 / (self.alpha2 + normED)

        return tre, edxx, edyy, edxy, normED, Gm

    def get_constitutive_matrix(self, strain: torch.Tensor) -> torch.Tensor:
        """Consistent tangent stiffness matrix (analytical)."""
        tre, edxx, edyy, edxy, normED, Gm = self._get_kinematics(strain)

        # Derivative terms
        dGm_dn = -Gm / (self.alpha2 + normED + 1e-12)
        gamma = dGm_dn / (normED + 1e-12)          # = (dGm/dn) / n

        ed_v = torch.stack([edxx, edyy, edxy], dim=-1)
        outer = ed_v.unsqueeze(-1) * ed_v.unsqueeze(-2)

        C_dev = (Gm.unsqueeze(-1).unsqueeze(-1) * self.P +
                 gamma.unsqueeze(-1).unsqueeze(-1) * outer)
        
        C_vol = self.Km * self.vol_matrix

        return C_vol + C_dev

    def update_state(self, strain, state_old, isTangent=True):
        tre, edxx, edyy, edxy, _, Gm = self._get_kinematics(strain)

        hydro = self.Km * tre
        sxx = hydro + Gm * edxx
        syy = hydro + Gm * edyy
        sxy = Gm * edxy

        stress = torch.stack([sxx, syy, sxy], dim=-1)

        ddsdde = None
        if isTangent:
            ddsdde = self.get_constitutive_matrix(strain)

        return stress, state_old, ddsdde

from rve_analyzer import DualEncoderFNO
from microstructures import MicrostructurePool
from microstructures import MicrostructureGenerator

class FNOmat(MaterialBase):
    """
    Surrogate material model based on DualEncoderFNO for FEINN.
    Microstructure pool is built in [0, 1]. 
    The external fhard function must return values in [0, 1].
    """
    def __init__(self,
                 model: torch.nn.Module,
                 normalizers: Dict[str, object],
                 microstructure_pool: MicrostructurePool,
                 fhard: Callable[[torch.Tensor], torch.Tensor],     # coords → fhard ∈ [0, 1]
                 chunk_size: int = 512,
                 n_state: int = 3,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        
        super().__init__(n_state=n_state, dtype=dtype, device=device)

        self.model = model.eval().to(device, dtype)

        self.x_normalizer = normalizers['x_normalizer']
        self.global_normalizer = normalizers['global_normalizer']
        self.y_normalizer = normalizers['y_normalizer']

        self.pool = microstructure_pool
        self.fhard = fhard
        self.chunk_size = chunk_size

        # Normalize pool once
        self.pool_phase_norm = self.x_normalizer.transform(self.pool.pool_phase)

    def init_state(self, nelem: int, ngp: int, coords: torch.Tensor) -> torch.Tensor:
        
        fhard_eval = self.fhard(coords)                                   # debe estar en [0, 1]

        diffs = torch.abs(fhard_eval.unsqueeze(-1) - self.pool.fhard_bins)
        tags = torch.argmin(diffs, dim=-1)

        state = torch.zeros((nelem, ngp, self.n_state), 
                            dtype=self.dtype, device=self.device)
        state[..., 0] = tags.to(self.dtype)                               # RVE tag
        return state

    def update_state(self, strain: torch.Tensor, state_old: torch.Tensor, isTangent: bool = False):
        """
        Updates the material state using the FNO surrogate model.
        Returns homogenized stress and updated internal variables (max Von Mises per phase).
        """
        ddsdde = None
        if isTangent:
            raise NotImplementedError("Tangent stiffness not implemented yet for FNOmat.")

        nelem, ngp, _ = strain.shape
        N = nelem * ngp

        # Extract local microstructures: (N, C, H, W)
        tags = state_old[..., 0].long()
        x_local = self.pool_phase_norm[tags.view(-1)]
        _, _, H, W = x_local.shape
        
        # Normalize global macro-strain
        x_global = strain.reshape(N, 3)
        x_global = self.global_normalizer.transform(x_global)

        # FNO prediction: returns full micromechanical stress field (N, 3, H, W)
        pred_norm = self._inference_in_chunks(x_local, x_global)

        # Inverse normalization and spatial reshape: (nelem, ngp, 3, H, W)
        pred_physical = self.y_normalizer.inverse_transform(pred_norm)
        sigma = pred_physical.view(nelem, ngp, 3, H, W)

        # Homogenization: Spatial average over H (dim=3) and W (dim=4)
        # Returns macroscopic stress per Gauss point: (nelem, ngp, 3)
        stress_hom = sigma.mean(dim=(3, 4))

        # Extract maximum micro-stress per phase
        vm_soft, vm_hard = self._compute_vm_per_phase(sigma, tags, H, W)

        # Update historical state variables with the new maximums
        state_new = state_old.clone()
        state_new[..., 1] = torch.maximum(state_old[..., 1], vm_soft)
        state_new[..., 2] = torch.maximum(state_old[..., 2], vm_hard)

        return stress_hom, state_new, ddsdde

    def _inference_in_chunks(self, x_local: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        """
        Performs batched inference to prevent GPU Out-Of-Memory (OOM) errors.
        Returns the full spatial prediction tensor.
        """
        N, _, H, W = x_local.shape
        pred = torch.zeros((N, 3, H, W), dtype=self.dtype, device=self.device)

        for i in range(0, N, self.chunk_size):
            j = min(i + self.chunk_size, N)
            with torch.no_grad():
                pred[i:j] = self.model(x_local[i:j], x_global[i:j])
                
        return pred

    def _compute_vm_per_phase(self, sigma: torch.Tensor, tags: torch.Tensor, H: int, W: int):
        """
        Computes the maximum Von Mises stress for the soft and hard phases
        by applying spatial masks to the full micro-stress field.
        """
        nelem, ngp, _, _, _ = sigma.shape
        
        # Retrieve and reshape masks to match integration points geometry
        tags_flat = tags.view(-1)
        mask_soft, mask_hard = self.pool.get_masks(tags_flat)
        
        mask_soft = mask_soft.view(nelem, ngp, H, W)
        mask_hard = mask_hard.view(nelem, ngp, H, W)

        # Extract individual stress components
        sxx = sigma[:, :, 0, :, :]
        syy = sigma[:, :, 1, :, :]
        sxy = sigma[:, :, 2, :, :]

        # Compute the full Von Mises micro-field
        vm_sq = sxx**2 + syy**2 - sxx*syy + 3 * sxy**2

        # Apply phase masks (zeros out the other phase) and find the spatial maximum
        vm_sq_soft_max = (vm_sq * mask_soft).amax(dim=(2, 3))
        vm_sq_hard_max = (vm_sq * mask_hard).amax(dim=(2, 3))

        vm_soft_max = torch.sqrt(vm_sq_soft_max)
        vm_hard_max = torch.sqrt(vm_sq_hard_max)

        return vm_soft_max, vm_hard_max