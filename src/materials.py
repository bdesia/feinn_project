
from abc import ABC, abstractmethod
import json
from typing import Tuple, Optional, Dict, Callable
import warnings

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

class ROMmat(MaterialBase):
    """
    Rule of Mixtures (ROM) homogenized material model for two phases.
    Computes effective properties using Voigt (isostrain) or Reuss (isostress) bounds.

    Parameters
    ----------
    mat_1, mat_2 : MaterialBase
        Phase materials. Must share dtype and device.
    fraction_1, fraction_2 : float, optional
        Volume fractions. One can be inferred from the other.
    method : str
        'voigt' or 'reuss'.
    reuss_max_iter : int
        Maximum Newton-Raphson iterations for the Reuss local solve (default: 50).
    reuss_tol : float
        Absolute residual tolerance for the Reuss NR convergence (default: 1e-10).
    tag : int, optional

    Notes
    -----
    Voigt (isostrain):
        Both phases share ε_macro. Exact for any material model (linear or nonlinear).
            σ = f1·σ1(ε) + f2·σ2(ε)
            C = f1·C1 + f2·C2

    Reuss (isostress) — Newton-Raphson local solve:
        Finds ε1 such that σ1(ε1) = σ2(ε2), with ε2 = (ε_macro - f1·ε1) / f2.
        Exact for path-independent (elastic) materials, both linear and nonlinear.
        For history-dependent materials (n_state > 0), uses state_old during NR
        iterations, which is consistent only for elastic constitutive models.
        Consistent tangent: C_eff = (f1·C1⁻¹ + f2·C2⁻¹)⁻¹ evaluated at local strains.
    """

    def __init__(
        self,
        mat_1: MaterialBase,
        mat_2: MaterialBase,
        fraction_1: Optional[float] = None,
        fraction_2: Optional[float] = None,
        method: str = 'voigt',
        reuss_max_iter: int = 50,
        reuss_tol: float = 1e-10,
        tag: Optional[int] = None,
        **kwargs
    ):
        if mat_1.dtype != mat_2.dtype or mat_1.device != mat_2.device:
            raise ValueError(
                f"mat_1 and mat_2 must share the same dtype and device. "
                f"Got mat_1({mat_1.dtype}, {mat_1.device}) "
                f"vs mat_2({mat_2.dtype}, {mat_2.device})."
            )

        if fraction_1 is None and fraction_2 is None:
            raise ValueError(
                "At least one volume fraction (fraction_1 or fraction_2) must be provided."
            )
        elif fraction_1 is None:
            fraction_1 = 1.0 - fraction_2
        elif fraction_2 is None:
            fraction_2 = 1.0 - fraction_1

        if abs(fraction_1 + fraction_2 - 1.0) > 1e-5:
            raise ValueError(
                f"Volume fractions must sum to 1.0. "
                f"Got {fraction_1} + {fraction_2} = {fraction_1 + fraction_2:.6f}."
            )

        method = method.lower()
        if method not in ('voigt', 'reuss'):
            raise ValueError(f"Method must be 'voigt' or 'reuss'. Got: '{method}'.")

        if method == 'reuss' and (mat_1.n_state > 0 or mat_2.n_state > 0):
            warnings.warn(
                "Reuss with history-dependent material(s): NR iterations use state_old "
                "at every step. This is consistent for elastic models but may drift for "
                "path-dependent (e.g. plastic) materials.",
                stacklevel=2
            )

        self.mat_1 = mat_1
        self.mat_2 = mat_2
        self.f1 = fraction_1
        self.f2 = fraction_2
        self.method = method
        self.reuss_max_iter = reuss_max_iter
        self.reuss_tol = reuss_tol

        super().__init__(
            n_state=mat_1.n_state + mat_2.n_state,
            dtype=mat_1.dtype,
            device=mat_1.device,
            tag=tag,
            **kwargs
        )

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def to(self, dtype=None, device=None):
        """Propagates dtype/device changes to both sub-materials."""
        super().to(dtype, device)
        self.mat_1.to(dtype, device)
        self.mat_2.to(dtype, device)
        return self

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def init_state(self, nelem: int, ngp2: int, **kwargs) -> torch.Tensor:
        """
        Initializes and concatenates history tensors for both phases.
        Returns shape (nelem, ngp2, n_state_1 + n_state_2). Safe when n_state == 0.
        """
        state_1 = self.mat_1.init_state(nelem, ngp2, **kwargs)
        state_2 = self.mat_2.init_state(nelem, ngp2, **kwargs)
        return torch.cat([state_1, state_2], dim=-1)

    # ------------------------------------------------------------------
    # Reuss local Newton-Raphson
    # ------------------------------------------------------------------

    def _reuss_local_nr(
        self,
        strain_macro: torch.Tensor,
        state_old_1: torch.Tensor,
        state_old_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finds local phase strains satisfying the Reuss isostress condition via NR.

        Residual   : R(ε1) = σ1(ε1) - σ2(ε2),   ε2 = (ε_macro - f1·ε1) / f2
        Jacobian   : K = C1 + (f1/f2)·C2
        NR update  : ε1 ← ε1 - K⁻¹·R

        Parameters
        ----------
        strain_macro : (nelem, ngp, 3)

        Returns
        -------
        eps1, eps2 : converged local strains, both (nelem, ngp, 3)
        """
        eps1 = strain_macro.clone()

        for ite in range(self.reuss_max_iter):
            eps2 = (strain_macro - self.f1 * eps1) / self.f2

            sigma1, _, C1 = self.mat_1.update_state(eps1, state_old_1, isTangent=True)
            sigma2, _, C2 = self.mat_2.update_state(eps2, state_old_2, isTangent=True)

            R = sigma1 - sigma2                             # (nelem, ngp, 3)

            if R.abs().max().item() < self.reuss_tol:
                break

            # K = dR/dε1 = C1 + (f1/f2)·C2
            K = C1 + (self.f1 / self.f2) * C2              # (nelem, ngp, 3, 3)
            eps1 = eps1 - torch.linalg.solve(K, R)         # (nelem, ngp, 3)

        else:
            warnings.warn(
                f"Reuss local NR did not converge after {self.reuss_max_iter} iterations. "
                f"Final residual: {R.abs().max().item():.3e}. "
                "Consider increasing reuss_max_iter or reuss_tol.",
                stacklevel=3
            )

        eps2 = (strain_macro - self.f1 * eps1) / self.f2
        return eps1, eps2

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update_state(
        self,
        strain: torch.Tensor,
        state_old: torch.Tensor,
        isTangent: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes homogenized stress and (optionally) consistent tangent stiffness.
        """
        s1 = self.mat_1.n_state
        state_old_1 = state_old[..., :s1]
        state_old_2 = state_old[..., s1:]

        if self.method == 'voigt':
            # Isostrain: both phases see ε_macro — exact for any nonlinear model
            sigma1, state_new_1, C1 = self.mat_1.update_state(strain, state_old_1, isTangent)
            sigma2, state_new_2, C2 = self.mat_2.update_state(strain, state_old_2, isTangent)

            stress  = self.f1 * sigma1 + self.f2 * sigma2
            ddsdde  = (self.f1 * C1 + self.f2 * C2) if isTangent else None

        else:  # reuss
            # Step 1: NR local solve to find ε1, ε2 satisfying isostress condition
            eps1, eps2 = self._reuss_local_nr(strain, state_old_1, state_old_2)

            # Step 2: Final update at converged local strains → correct states and tangents
            sigma1, state_new_1, C1 = self.mat_1.update_state(eps1, state_old_1, isTangent)
            sigma2, state_new_2, C2 = self.mat_2.update_state(eps2, state_old_2, isTangent)

            # At convergence σ1 ≈ σ2; average cancels any residual numerical drift
            stress = self.f1 * sigma1 + self.f2 * sigma2

            # Consistent tangent: C_eff = (f1·C1⁻¹ + f2·C2⁻¹)⁻¹ at local strains
            ddsdde = torch.linalg.inv(
                self.f1 * torch.linalg.inv(C1) + self.f2 * torch.linalg.inv(C2)
            ) if isTangent else None

        state_new = torch.cat([state_new_1, state_new_2], dim=-1)
        return stress, state_new, ddsdde

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
                 device: str = 'cuda',
                 seed: Optional[int] = None):

        super().__init__(n_state=n_state, dtype=dtype, device=device)

        self.model = model.to(device).eval()        # evaluation mode, no dropout/batchnorm updates
        self.model.requires_grad_(False)            # ensure no gradients for model parameters during solver backward pass

        self.pool = microstructure_pool             # Pre-computed microstructure features for all Gauss points
        self.fhard = fhard                          # Function to compute fraction of hard phase from microstructure features
        self.chunk_size = chunk_size                # Number of samples to process per chunk during inference and autograd
        self.seed = seed

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

    def to(self, dtype=None, device=None):
        super().to(dtype, device)
        if device:
            self.model        = self.model.to(device)
            self.x_norm       = self.x_norm.to(device)
            self.g_norm       = self.g_norm.to(device)
            self.y_norm       = self.y_norm.to(device)
            self.pool_phase_norm = self.pool_phase_norm.to(device)
        return self

    def init_state(self, nelem: int, ngp: int, **kwargs) -> torch.Tensor:
        """Assign RVE tag based on spatial fhard distribution."""
        coords = kwargs['coords']
        fhard_eval = self.fhard(coords)
        fhard_eval = torch.clamp(fhard_eval, 0.0, 1.0)

        fhard_bins = self.pool.fhard_bins.to(device=fhard_eval.device)
        diffs = torch.abs(fhard_eval.unsqueeze(-1) - fhard_bins)
        bin_idx = torch.argmin(diffs, dim=-1)                               # (nelem, ngp)

        gen = torch.Generator(device='cpu')   # always CPU so same seed → same sequence on any device
        if self.seed is not None:
            gen.manual_seed(self.seed)
        micro_idx = torch.randint(0, self.pool.nmicro, bin_idx.shape,
                                  generator=gen).to(self.device)

        tags = bin_idx.to(self.device) * self.pool.nmicro + micro_idx        # flat pool index

        state = torch.zeros((nelem, ngp, self.n_state),
                            dtype=self.dtype, device=self.device)
        state[..., 0] = tags.to(self.dtype)
        return state

    def update_state(self, strain: torch.Tensor, state_old: torch.Tensor, isTangent: bool = True):
        """
        Updates the material state for the given strain increment.
        Returns homogenized stress and optionally the tangent stiffness matrix.
        """
        nelem, ngp, _ = strain.shape        # (nelem, ngp, 3)
        N = nelem * ngp                     # Total number of Gauss points across the batch
        
        tags = torch.round(state_old[..., 0]).long()
        x_local_norm = self.pool_phase_norm[tags.view(-1)]      # assign microstructure based on RVE tag (stored in pos 0 of state).
        
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
        output = torch.zeros((N, self.out_C, self.out_H, self.out_W), dtype=self.dtype, device=self.device)
        
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
                        inputs=[chunk_x_g],
                        grad_outputs=base_grad_out,
                        retain_graph=True if k < 2 else False, 
                        # only_inputs=True
                    )[0]
                    
                    base_grad_out[:, k] = 0.0
                    
                    ddsdde_norm[i:j, k, :] = grad_k

        # Scale to physical units, reshape, and enforce mechanical symmetry
        ddsdde = (ddsdde_norm * scale_factor).view(nelem, ngp, 3, 3)
        ddsdde = (ddsdde + ddsdde.transpose(-1, -2)) / 2.0

        return ddsdde

from solver import NFEA
from conditions import PeriodicBoundaryCondition, BoundaryCondition

class FEMmat(MaterialBase):
    """
    FEM-based material model for multiscale FE simulations (FE^2).
    Evaluates homogenized stress and tangent stiffness dynamically running
    a local FEA problem on the RVE.

    Parameters
    ----------
    exact_integration : bool
        If True, homogenizes stress using Gauss quadrature weights and Jacobian
        determinants (w_g * detJ), which is exact for any mesh topology.
        If False, uses a simple arithmetic mean over all Gauss points, which is
        exact only for uniform (voxelized) meshes where all elements have equal
        volume.
    fd_scheme : {'forward', 'central'}
        Finite difference scheme for the numerical tangent stiffness.
        'forward'  uses O(δ) accuracy and requires 3 extra RVE solves per GP.
        'central'  uses O(δ²) accuracy and requires 6 extra RVE solves per GP,
        which is more accurate for nonlinear materials at a higher cost.
    """
    def __init__(self,
                 soft: MaterialBase,
                 hard: MaterialBase,
                 microstructure_pool: object,
                 fhard: Callable[[torch.Tensor], torch.Tensor],
                 n_state: int = 3,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cuda',
                 exact_integration: bool = False,
                 fd_scheme: str = 'forward',
                 seed: Optional[int] = None,
                 **kwargs):

        super().__init__(n_state=n_state, dtype=dtype, device=device, **kwargs)

        if fd_scheme not in ('forward', 'central'):
            raise ValueError(f"fd_scheme must be 'forward' or 'central', got '{fd_scheme}'.")

        self.pool = microstructure_pool
        self.fhard = fhard
        self.exact_integration = exact_integration
        self.fd_scheme = fd_scheme
        self.seed = seed

        self.matfield = {
            'soft_s': soft,
            'hard_s': hard
        }

    def to(self, dtype=None, device=None):
        super().to(dtype, device)
        for mat in self.matfield.values():
            mat.to(dtype, device)
        return self

    def init_state(self, nelem: int, ngp: int, **kwargs) -> torch.Tensor:
        """Assign RVE tag based on spatial fhard distribution."""
        coords = kwargs['coords']
        fhard_eval = self.fhard(coords)
        fhard_eval = torch.clamp(fhard_eval, 0.0, 1.0)

        fhard_bins = self.pool.fhard_bins.to(device=fhard_eval.device)
        diffs = torch.abs(fhard_eval.unsqueeze(-1) - fhard_bins)
        bin_idx = torch.argmin(diffs, dim=-1)                               # (nelem, ngp)

        gen = torch.Generator(device='cpu')   # always CPU so same seed → same sequence on any device
        if self.seed is not None:
            gen.manual_seed(self.seed)
        micro_idx = torch.randint(0, self.pool.nmicro, bin_idx.shape,
                                  generator=gen).to(self.device)

        tags = bin_idx.to(self.device) * self.pool.nmicro + micro_idx      # flat pool index

        state = torch.zeros((nelem, ngp, self.n_state),
                            dtype=self.dtype, device=self.device)
        state[..., 0] = tags.to(self.dtype)
        return state

    def update_state(self, strain: torch.Tensor, state_old: torch.Tensor, isTangent: bool = True):
        """
        Updates the material state by running local FEM homogenizations.
        Iterates sequentially over all macro Gauss points.
        """
        nelem, ngp, _ = strain.shape
        N = nelem * ngp

        strain_flat = strain.view(-1, 3)
        tags_flat = torch.round(state_old[..., 0]).long().view(-1)

        stress_flat = torch.zeros(N, 3, dtype=self.dtype, device=self.device)
        ddsdde_flat = torch.zeros(N, 3, 3, dtype=self.dtype, device=self.device) if isTangent else None

        for i in range(N):
            macro_e = strain_flat[i]
            tag = tags_flat[i].item()

            sigma_hom = self._solve_rve(macro_e, tag)
            stress_flat[i] = sigma_hom

            if isTangent:
                ddsdde_flat[i] = self._compute_numerical_tangent(macro_e, tag, sigma_hom)

        stress_out = stress_flat.view(nelem, ngp, 3)
        self._stress_cache = stress_out.clone()   # reused by get_stress_field()
        ddsdde = ddsdde_flat.view(nelem, ngp, 3, 3) if isTangent else None
        return stress_out, state_old, ddsdde

    def get_stress_field(self) -> torch.Tensor:
        """Return the stress field from the last converged Newton iteration (no RVE re-runs)."""
        if not hasattr(self, '_stress_cache'):
            raise RuntimeError("No stress cached yet — run the FEM² solve first.")
        return self._stress_cache

    def _solve_rve(self, macro_strain: torch.Tensor, tag: int) -> torch.Tensor:
        """
        Sets up and solves the NFEA problem for a single RVE given a macroscopic strain.
        """
        mesh, left_right_pairs, bottom_top_pairs = self.pool.meshes[tag]

        delta_x = float(mesh.coordinates[:, 0].max() - mesh.coordinates[:, 0].min())
        delta_y = float(mesh.coordinates[:, 1].max() - mesh.coordinates[:, 1].min())

        pbc = PeriodicBoundaryCondition(
            exx=macro_strain[0].item(),
            eyy=macro_strain[1].item(),
            gxy=macro_strain[2].item(),
            left_right_pairs=left_right_pairs,
            bottom_top_pairs=bottom_top_pairs,
            delta_x=delta_x,
            delta_y=delta_y
        )

        mpcs = pbc.get_constraints()
        bcs = {
            'fixed_n': [BoundaryCondition(dof=1, value=0.0),
                        BoundaryCondition(dof=2, value=0.0)],
        }

        fem_solver = NFEA(
            mesh=mesh,
            bcs=bcs,
            matfld=self.matfield,
            mpcs=mpcs,
            verbose=False,
            device=self.device
        )

        fem_solver.run_complete(nsteps=1)

        sigma_list = []
        wdV_list = [] if self.exact_integration else None

        for batch in fem_solver.quad_batches.values():
            u_local = batch.get_local_disp(fem_solver.udisp)
            eps_gp = batch.compute_infinitesimal_strain(u_local)
            sigma_gp, _, _ = batch.material.update_state(eps_gp, batch.state, isTangent=False)
            sigma_list.append(sigma_gp.view(-1, 3))
            if self.exact_integration:
                wdV_list.append(batch.weighted_dV.view(-1))

        sigma_flat = torch.cat(sigma_list, dim=0)

        if self.exact_integration:
            wdV_flat = torch.cat(wdV_list, dim=0)
            V_total = wdV_flat.sum()
            stress_hom = (sigma_flat * wdV_flat.unsqueeze(-1)).sum(dim=0) / V_total
        else:
            stress_hom = sigma_flat.mean(dim=0)

        return stress_hom

    def _compute_numerical_tangent(self, macro_strain: torch.Tensor, tag: int, base_stress: torch.Tensor, pert: float = 1e-5) -> torch.Tensor:
        """
        Computes C = d(sigma_hom)/d(epsilon_hom) via finite differences.
        'forward' : O(δ)  — 
        'central' : O(δ²) — more accurate for nonlinear materials.
        """
        C = torch.zeros((3, 3), dtype=self.dtype, device=self.device)

        for i in range(3):
            e_p = macro_strain.clone()
            e_p[i] += pert

            if self.fd_scheme == 'central':
                e_m = macro_strain.clone()
                e_m[i] -= pert
                C[:, i] = (self._solve_rve(e_p, tag) - self._solve_rve(e_m, tag)) / (2.0 * pert)
            else:
                C[:, i] = (self._solve_rve(e_p, tag) - base_stress) / pert

        return 0.5 * (C + C.transpose(-1, -2))