
from abc import ABC, abstractmethod
import json
from typing import Tuple, Optional, Dict, Callable

import torch
from torch.func import vmap, jacrev

from rve_analyzer import DualEncoderFNO
from microstrutures_gen import MicrostructureGenerator

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

class FNOmat(MaterialBase):
    """
    Surrogate material model based on DualEncoderFNO for FEINN.
    Evaluates homogenized stress, consistent tangent stiffness, 
    and tracks maximum localized Von Mises stress per phase.
    """

    def __init__(self,
                 config_path: str,
                 normalizer_path: str,
                 checkpoint_path: str,
                 micro_generator: MicrostructureGenerator,
                 chunk_size: int = 4096,
                 tag: Optional[int] = None,
                 dtype: torch.dtype = torch.float64,
                 device: str = 'cpu',
                 **kwargs):
        
        # Initialize base class with n_state=3:
        # [0: Microstructure Tag, 1: Max VM Soft, 2: Max VM Hard]
        super().__init__(n_state=3, dtype=dtype, device=device, tag=tag)
        
        self.chunk_size = chunk_size
        self.micro_generator = micro_generator
        
        # 1. Load Architecture Configuration
        if str(config_path).endswith('.pth'):
            cfg = torch.load(config_path, map_location='cpu')
        else:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                
        # Handle nested config dictionaries (e.g., best_params)
        model_params = cfg if isinstance(cfg, dict) and 'n_modes' in cfg else cfg.get('best_params', cfg)

        # 2. Instantiate DualEncoderFNO
        self.model = DualEncoderFNO(
            in_channels=model_params.get('in_channels', 1),
            out_channels=model_params.get('out_channels', 3),
            n_macro=model_params.get('n_macro', 3),
            n_modes=model_params.get('n_modes', 32),
            hidden_channels=model_params.get('hidden_channels', 64),
            n_layers=model_params.get('n_layers', 4),
            channel_mlp_dropout=model_params.get('channel_mlp_dropout', 0.05),
            film_mlp_layers=model_params.get('film_mlp_layers', 2),
            film_mlp_neurons=model_params.get('film_mlp_neurons', 128),
            film_mlp_dropout=model_params.get('film_mlp_dropout', 0.0),
            use_sinusoidal_emb=model_params.get('use_sinusoidal_emb', True),
            sin_emb_nfreq=model_params.get('sin_emb_nfreq', 6),
            use_positional_grid=True,
            film_per_layer=True
        ).to(self.device).to(self.dtype)

        # 3. Load Model Checkpoint (Weights)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 4. Load Normalizers
        normalizers_dict = torch.load(normalizer_path, map_location=self.device)
        self.x_normalizer = normalizers_dict['x_normalizer']           # For microstructure image
        self.global_normalizer = normalizers_dict['global_normalizer'] # For macro strain
        self.y_normalizer = normalizers_dict['y_normalizer']           # For stress field

        # 5. Internal attributes for spatial caching
        self.microstructures: Dict[int, Dict[str, torch.Tensor]] = {}
        self.node_coords: Optional[torch.Tensor] = None
        self.f_formula = None 

    def register_microstructure(self, tag: int, phase: torch.Tensor, m_soft: torch.Tensor, m_hard: torch.Tensor):
        """Caches the generated microstructure image and its spatial masks."""
        self.microstructures[tag] = {
            'image': phase,
            'mask_soft': m_soft,
            'mask_hard': m_hard
        }

    def init_state(self, nelem: int, ngp2: int, **kwargs) -> torch.Tensor:
        """Initializes history variables and maps spatial RVEs dynamically."""
        state = torch.zeros((nelem, ngp2, self.n_state), device=self.device, dtype=self.dtype)

        coords = kwargs.get('coords')
        if coords is None:
            raise RuntimeError("FNOmat requires 'coords' in kwargs for init_state.")

        self.node_coords = coords.to(self.device, self.dtype)

        if self.f_formula is None:
            print("FNOmat: warning - f_formula not set. Returning zero state.")
            return state

        # Mock Gauss points mapping (replace with element shape functions in production)
        gp_coords = torch.mean(self.node_coords, dim=1, keepdim=True)
        coords_flat = gp_coords.view(-1, 2)
        B = coords_flat.shape[0]

        tags = torch.zeros(B, dtype=torch.long, device=self.device)

        for i in range(B):
            x, y = coords_flat[i]
            fhard = float(self.f_formula(x.item(), y.item()))
            tag = int(round(fhard * 10000))

            if tag not in self.microstructures:
                phase, m_soft, m_hard = self.micro_generator.generate(fhard, self.device, self.dtype)
                self.register_microstructure(tag, phase, m_soft, m_hard)

            tags[i] = tag

        state[..., 0] = tags.view(nelem, ngp2)
        return state

    def update_state(self, strain: torch.Tensor, state_old: torch.Tensor, isTangent: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Constitutive evaluation: scales inputs, computes homogenized stress, 
        evaluates tangent stiffness via autograd, and updates per-phase states.
        """
        nelem, ngp2, _ = strain.shape
        strain_flat = strain.view(-1, 3)
        tags_flat = state_old[..., 0].view(-1)
        B = strain_flat.shape[0]

        stress_out = torch.empty_like(strain_flat)
        vm_soft_out = torch.empty(B, device=self.device, dtype=self.dtype)
        vm_hard_out = torch.empty(B, device=self.device, dtype=self.dtype)
        D_out = torch.empty((B, 3, 3), device=self.device, dtype=self.dtype) if isTangent else None
        
        def single_element_forward(s_phys: torch.Tensor, m_phys: torch.Tensor) -> torch.Tensor:
            """
            Internal forward mapping for jacrev. 
            Mapea deformación e imagen física al espacio latente, y devuelve tensión física.
            """
            s_norm = self.global_normalizer.transform(s_phys.unsqueeze(0)).squeeze(0)
            m_norm = self.x_normalizer.transform(m_phys.unsqueeze(0)).squeeze(0)
            
            field_norm = self.model(s_norm.unsqueeze(0), m_norm.unsqueeze(0)).squeeze(0) 
            macro_norm = field_norm.mean(dim=(-2, -1))
            
            return self.y_normalizer.inverse_transform(macro_norm.unsqueeze(0)).squeeze(0)

        if isTangent:
            batched_jacobian_fn = vmap(jacrev(single_element_forward, argnums=0), in_dims=(0, 0))

        for i in range(0, B, self.chunk_size):
            end = min(i + self.chunk_size, B)
            s_chunk = strain_flat[i:end]
            t_chunk = tags_flat[i:end]
            
            # Fetch cached physical data
            micro_chunk = torch.stack([self.microstructures[int(t.item())]['image'] for t in t_chunk])
            mask_soft_chunk = torch.stack([self.microstructures[int(t.item())]['mask_soft'] for t in t_chunk])
            mask_hard_chunk = torch.stack([self.microstructures[int(t.item())]['mask_hard'] for t in t_chunk])
            
            with torch.no_grad(): 
                # Normalizar entradas físicas para inferencia de campo completo
                s_chunk_norm = self.global_normalizer.transform(s_chunk)
                micro_chunk_norm = self.x_normalizer.transform(micro_chunk)
                
                # Inferencia pura en espacio latente
                stress_field_norm = self.model(s_chunk_norm, micro_chunk_norm)
                
                # Desnormalizar el campo 2D a espacio físico (MPa)
                stress_field_phys = self.y_normalizer.inverse_transform(stress_field_norm)
                stress_out[i:end] = stress_field_phys.mean(dim=(-2, -1))
                
                if isTangent:
                    # El Jacobiano se calcula pasando tensores FÍSICOS a single_element_forward
                    D_out[i:end] = batched_jacobian_fn(s_chunk, micro_chunk)

                # Phase-separated Maximum Von Mises Extraction
                s_xx = stress_field_phys[:, 0, :, :]
                s_yy = stress_field_phys[:, 1, :, :]
                s_xy = stress_field_phys[:, 2, :, :]
                vm_field = torch.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2) # (B_chunk, H, W)

                # Apply exact geometrical masks provided by the generator
                vm_hard_out[i:end] = (vm_field * mask_hard_chunk).amax(dim=(-2, -1))
                vm_soft_out[i:end] = (vm_field * mask_soft_chunk).amax(dim=(-2, -1))
            
        # Update History Variables
        state_new = state_old.clone()
        
        vm_soft_old = state_old[..., 1].view(-1)
        state_new[..., 1] = torch.maximum(vm_soft_old, vm_soft_out).view(nelem, ngp2)
        
        vm_hard_old = state_old[..., 2].view(-1)
        state_new[..., 2] = torch.maximum(vm_hard_old, vm_hard_out).view(nelem, ngp2)

        # Reshape to element batch dimensions
        stress_res = stress_out.view(nelem, ngp2, 3)
        ddsdde_res = D_out.view(nelem, ngp2, 3, 3) if isTangent else None

        return stress_res, state_new, ddsdde_res