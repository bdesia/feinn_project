from typing import Optional
import torch
from abc import ABC, abstractmethod


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

    def __init__(self, n_state: int = 0):
        self.n_state: int = n_state
        self.is_vectorized: bool = False
        self.nelem: Optional[int] = None
        self.ngp2: Optional[int] = None

    @abstractmethod
    def get_constitutive_matrix(self, *args, **kwargs) -> torch.Tensor:
        """Return the constitutive (or algorithmic tangent) matrix."""
        pass

    @abstractmethod
    def compute_stress(self, *args, **kwargs) -> torch.Tensor:
        """Compute stress and update internal variables."""
        pass

    def vectorize(self, nelem: int, ngp2: int):
        """Prepare the material for vectorized evaluation over nelem elements and npoints Gauss points per element."""
        if nelem <= 0 or ngp2 <= 0:
            raise ValueError("nelem and ngp2 must be positive integers.")
        self.nelem = nelem
        self.ngp2 = ngp2
        self.is_vectorized = True

class LinearElastic(MaterialBase):
    """
    Linear isotropic elastic material under plane strain assumption (2D).

    Uses Voigt notation: strain/stress vector = [ε_xx, ε_yy, 2ε_xy].
    """
    def __init__(self, emod: float, nu: float):
        super().__init__()
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
        self.G = emod / (2 * (1 + nu))                     # Shear modulus

    def get_constitutive_matrix(self) -> torch.Tensor:
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

        C = torch.tensor([[C11, C12, 0.0],
                          [C12, C11, 0.0],
                          [0.0,  0.0, C33]], dtype=torch.float64)
        
        if self.is_vectorized:
            if self.nelem is None or self.ngp2 is None:
                raise ValueError("Vectorized mode enabled but nelem or ngp not set. Call vectorize(nelem, ngp) first.")
            C = C.unsqueeze(0).unsqueeze(0).expand(self.nelem, self.ngp2, -1, -1)
        return C

    def compute_stress(self, strain: torch.Tensor) -> torch.Tensor:
        """
        Compute stress from total strain: σ = C ε.

        Parameters
        ----------
        strain : torch.Tensor
            Total strain in Voigt notation. Shape (3,) or (ngp, 3).

        Returns
        -------
        torch.Tensor
            Cauchy stress in Voigt notation. Same shape as input.
        """
        C = self.get_constitutive_matrix().to(strain.device)
        C = C.to(strain.dtype)
        return torch.einsum('...ij,...j->...i', C, strain)

    def compute_pk2_stress(self, gl_strain: torch.Tensor) -> torch.Tensor:
        """Compute 2nd Piola-Kirchhoff stress from Green-Lagrange strain."""
        return self.compute_stress(gl_strain)


class LinearElasticPlaneStress(MaterialBase):
    """
    Linear isotropic elastic material under plane stress assumption (2D).

    Uses Voigt notation: strain/stress vector = [ε_xx, ε_yy, 2ε_xy].
    """
    def __init__(self, emod: float, nu: float):
        super().__init__()
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

    def get_constitutive_matrix(self) -> torch.Tensor:
        """
        Returns the constant elastic constitutive matrix (3x3) for plane stress.

        Returns
        -------
        torch.Tensor
            Elastic stiffness matrix in Voigt notation.
        """
        C11 = self._factor1
        C12 = self._factor1 * self.nu
        C33 = self._factor1 * (1 - self.nu) / 2

        C = torch.tensor([[C11, C12, 0.0],
                          [C12, C11, 0.0],
                          [0.0,  0.0, C33]], dtype=torch.float64)
        
        if self.is_vectorized:
            if self.nelem is None or self.ngp2 is None:
                raise ValueError("Vectorized mode enabled but nelem or ngp not set. Call vectorize(nelem, ngp) first.")
            C = C.unsqueeze(0).unsqueeze(0).expand(self.nelem, self.ngp2, -1, -1)
        
        return C

    def compute_stress(self, strain: torch.Tensor) -> torch.Tensor:
        """
        Compute stress from total strain: σ = C ε.

        Parameters
        ----------
        strain : torch.Tensor
            Total strain in Voigt notation. Shape (3,) or (ngp, 3).

        Returns
        -------
        torch.Tensor
            Cauchy stress in Voigt notation. Same shape as input.
        """
        
        C = self.get_constitutive_matrix().to(strain.device)
        C = C.to(strain.dtype)
        return torch.einsum('...ij,...j->...i', C, strain)
    
    def compute_pk2_stress(self, gl_strain: torch.Tensor) -> torch.Tensor:
        """Second Piola-Kirchhoff stress"""
        return self.compute_stress(gl_strain)