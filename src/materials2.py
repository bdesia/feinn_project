import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class MaterialBase(ABC):
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

        pass



class LinearElastic(MaterialBase):
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
        self.G = emod / (2 * (1 + nu))                     # Shear modulus

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

        C = torch.tensor([[C11, C12, 0.0],
                          [C12, C11, 0.0],
                          [0.0,  0.0, C33]], dtype=self.dtype, device=self.device)
        
        return C

    def update_state(self, strain, state_old, isTangent=True):
        """
        Update stress, state variables and, if required, tangent stiffness

        Parameters
        ----------
        strain : torch.Tensor
            Total strain in Voigt notation. Shape (3,) or (ngp, 3).

        Returns
        -------
        torch.Tensor
            Cauchy stress in Voigt notation. Same shape as input.
        """
        C = self._get_constitutive_matrix()
        # stress = C : strain (vectorizado)
        stress = torch.einsum('ij,...j->...i', C, strain)
       
        ddsdde = None
        if isTangent:
            # Expandimos C para el batch (nelem, ngp2, 3, 3) sin duplicar memoria
            ddsdde = C.view(1, 1, 3, 3).expand(strain.shape[0], strain.shape[1], 3, 3)
           
        return stress, state_old, ddsdde

class LinearElasticPlaneStress(MaterialBase):
    def __init__(self, emod: float, nu: float):
        super().__init__(n_state=0)
        self.emod = emod
        self.nu = nu

    def _get_C(self) -> torch.Tensor:
        """Matriz de rigidez para Tensión Plana."""
        f = self.emod / (1 - self.nu**2)
        c_mat = [[f, f * self.nu, 0.0],
                 [f * self.nu, f, 0.0],
                 [0.0, 0.0, f * (1 - self.nu) / 2]]
        return torch.tensor(c_mat, device=self.device, dtype=self.dtype)

    def update_state(self, strain, state_old, is_tangent=True):
        C = self._get_C()
        stress = torch.einsum('ij,...j->...i', C, strain)
       
        tangent = None
        if is_tangent:
            tangent = C.view(1, 1, 3, 3).expand(strain.shape[0], strain.shape[1], 3, 3)
           
        return stress, state_old, tangent