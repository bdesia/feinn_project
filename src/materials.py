import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional

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
                 device: str = 'cpu'):
        
        self.n_state = n_state
        self.dtype = dtype
        self.device = device

    def to(self, dtype=None, device=None):
        """Update device and dtype of the material."""
        if dtype: self.dtype = dtype
        if device: self.device = device
        
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(dtype=self.dtype, device=self.device))
        return self

    def init_state(self, nelem: int, ngp2: int) -> torch.Tensor:
        """Initialize the state tensor."""
        return torch.zeros((nelem, ngp2, self.n_state),
                           device=self.device, dtype=self.dtype)

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
    def __init__(self, emod: float, nu: float):
        super().__init__(n_state=0)
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
    def __init__(self, emod: float, nu: float):
        super().__init__(n_state=0)
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

    def __init__(self, k: float, e0: float, s0: float, m: float):
        super().__init__(n_state=0)

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