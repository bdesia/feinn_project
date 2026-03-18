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