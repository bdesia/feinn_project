
from abc import ABC, abstractmethod
import json
from typing import Tuple, Optional, Dict, Callable

import torch
from torch.utils.checkpoint import checkpoint
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

from microstructures import MicrostructurePool

class PartialGradientFNO(torch.autograd.Function):
    """
    Custom autograd function for memory-efficient training.
    Performs a graph-free forward pass and computes backward gradients 
    only with respect to the global strain (x_global).
    """
    @staticmethod
    def forward(ctx, model, x_local, x_global):
        """ 
        No gradient tracking in forward pass to save memory. 
        """
        ctx.save_for_backward(x_global)     # Save only global strain for backward pass
        ctx.model = model                   # Save model for backward pass
        ctx.x_local = x_local.detach()      # Detach local features to prevent gradient tracking (no gradients needed for x_local)

        # Enforce float32 to prevent cuFFT errors on non-power-of-2 grids
        x_local_f32 = x_local.contiguous().to(torch.float32)
        x_global_f32 = x_global.contiguous().to(torch.float32)

        with torch.no_grad():
            output = model(x_local_f32, x_global_f32)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        # Retrieve saved tensors and model from saved context
        x_global = ctx.saved_tensors[0]
        model = ctx.model
        x_local = ctx.x_local

        with torch.enable_grad():
            # Locally rebuild the graph for the backward pass
            x_g_tmp = x_global.detach().to(torch.float32).requires_grad_(True)
            out_tmp = model(x_local.to(torch.float32), x_g_tmp)

            grad_x_global = torch.autograd.grad(
                out_tmp, x_g_tmp,
                grad_outputs=grad_output.to(torch.float32),
                retain_graph=False
            )[0]

        return None, None, grad_x_global    # No gradients for model and x_local, only for x_global


class FNOmat(MaterialBase):
    """
    FNO-based surrogate material model for multiscale FE simulations.
    Evaluates homogenized stress and tangent stiffness efficiently.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 normalizers: Dict[str, object],
                 microstructure_pool: object,
                 fhard: Callable[[torch.Tensor], torch.Tensor],
                 chunk_size: int = 48,
                 n_state: int = 3,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cuda'):
        
        super().__init__(n_state=n_state, dtype=dtype, device=device)

        self.model = model.to(device).eval()        # evaluation mode, no dropout/batchnorm updates
        self.model.requires_grad_(False)            # ensure no gradients for model parameters during solver backward pass
        
        self.pool = microstructure_pool             # Pre-computed microstructure features for all Gauss points
        self.fhard = fhard                          # Function to compute fraction of hard phase from microstructure features
        self.chunk_size = chunk_size                # Number of samples to process per chunk during inference and autograd

        # Intance normalizers
        self.x_norm = normalizers['x_normalizer'].to(device)
        self.g_norm = normalizers['global_normalizer'].to(device)
        self.y_norm = normalizers['y_normalizer'].to(device)

        # Reshape image normalizers for spatial broadcasting (C, H, W)
        for norm in [self.x_norm, self.y_norm]:
            if norm.mean.ndim == 1:
                norm.mean = norm.mean.view(-1, 1, 1)
                norm.std = norm.std.view(-1, 1, 1)

        # Pre-normalize microstructure features for all Gauss points and store on device
        self.pool_phase_norm = self.x_norm.transform(self.pool.pool_phase.to(device))
        
        # Output dimensions for reshaping predictions
        self.out_C = self.y_norm.mean.shape[0]
        self.out_H = self.pool.pool_phase.shape[2]
        self.out_W = self.pool.pool_phase.shape[3]
        
    def update_state(self, strain: torch.Tensor, state_old: torch.Tensor, isTangent: bool = False):
        """
        Updates the material state for the given strain increment.
        Returns homogenized stress and optionally the tangent stiffness matrix.
        """
        nelem, ngp, _ = strain.shape        # (nelem, ngp, 3)
        N = nelem * ngp                     # Total number of Gauss points across the batch
        
        tags = state_old[..., 0].long()
        x_local_norm = self.pool_phase_norm[tags.view(-1)]
        
        x_global = strain.reshape(N, 3).to(self.dtype)
        x_global_norm = self.g_norm.transform(x_global)

        # Forward pass and homogenization
        pred_norm = self._inference_in_chunks(x_local_norm, x_global_norm)       # normalized stress predictions (N, 3, H, W)
        pred_physical = self.y_norm.inverse_transform(pred_norm)            # physical stress predictions (N, 3, H, W)
        
        _, _, H, W = x_local_norm.shape
        sigma = pred_physical.view(nelem, ngp, 3, H, W)     # reshape to (nelem, ngp, 3, H, W)
        stress_hom = sigma.mean(dim=(3, 4))                 # homogeneization. shape: (nelem, ngp, 3)

        # Compute tangent stiffness if requested by the solver
        ddsdde = self._compute_tangent_autograd(x_local_norm, x_global_norm, nelem, ngp) if isTangent else None

        return stress_hom, state_old, ddsdde

    def _inference_in_chunks(self, x_local_norm: torch.Tensor, x_global_norm: torch.Tensor) -> torch.Tensor:
        """
        Performs inference in batches (chunks) to limit peak VRAM consumption.
        """
        N = x_local_norm.shape[0]
        output = torch.empty((N, self.out_C, self.out_H, self.out_W), dtype=self.dtype, device=self.device)
        
        for i in range(0, N, self.chunk_size):
            j = min(i + self.chunk_size, N)
            output[i:j] = PartialGradientFNO.apply(self.model, x_local_norm[i:j], x_global_norm[i:j])
            
        return output

    def _compute_tangent_autograd(self, x_local_norm: torch.Tensor, x_global_norm: torch.Tensor, nelem: int, ngp: int):
        """
        Computes the consistent tangent stiffness matrix (Jacobian) using chunked autograd.
        Applies chain rule scaling to return values in physical units.
        """
        N = x_local_norm.shape[0]
        ddsdde_norm = torch.zeros((N, 3, 3), dtype=self.dtype, device=self.device)

        # Chain rule factor: d(sigma_phys) / d(epsilon_phys) = d(sigma_phys) / d(epsilon_norm) * (1 / std_epsilon)
        scale_factor = 1.0 / self.g_norm.std.view(1, 3)

        # Force gradient tracking, overriding any global torch.no_grad() from the solver
        with torch.enable_grad():
            
            # Process in chunks to prevent OOM
            for i in range(0, N, self.chunk_size):
                j = min(i + self.chunk_size, N)

                chunk_x_l = x_local_norm[i:j]
                chunk_x_g = x_global_norm[i:j].detach().requires_grad_(True)

                # Single forward pass per chunk; float32 bypasses cuFFT constraints
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    pred_norm = self.model(chunk_x_l.to(torch.float32), chunk_x_g.to(torch.float32))
                    
                    # Un-normalize directly within the computational graph
                    pred_physical = self.y_norm.inverse_transform(pred_norm)
                    stress_hom = pred_physical.mean(dim=(2, 3)) # Shape: (chunk_size, 3)
                    
                    base_grad_out = torch.zeros_like(stress_hom)
                    
                # Compute Jacobian columns (Voigt notation components)
                for k in range(3):
                    base_grad_out[:, k] = 1.0
                    
                    grad_k = torch.autograd.grad(
                        outputs=stress_hom,
                        inputs=chunk_x_g,
                        grad_outputs=base_grad_out,
                        retain_graph=True if k < 2 else False, 
                        only_inputs=True
                    )[0]
                    
                    base_grad_out[:, k] = 0.0
                    
                    ddsdde_norm[i:j, k, :] = grad_k

        # Scale to physical units, reshape, and enforce mechanical symmetry
        ddsdde = (ddsdde_norm * scale_factor).view(nelem, ngp, 3, 3)
        ddsdde = (ddsdde + ddsdde.transpose(-1, -2)) / 2.0

        return ddsdde