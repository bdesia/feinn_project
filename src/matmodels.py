
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
        (multiple Gauss points simultaneously).
    """

    @abstractmethod
    def __init__(self):
        self.n_state: int
        self.is_vectorized: bool = False

    @abstractmethod
    def get_constitutive_matrix(self, *args, **kwargs) -> torch.Tensor:
        """Return the constitutive (or algorithmic tangent) matrix."""
        pass

    @abstractmethod
    def compute_stress(self, *args, **kwargs) -> torch.Tensor:
        """Compute stress and update internal variables."""
        pass

    @abstractmethod
    def vectorize(self, ngp: int):
        """Prepare the material for vectorized evaluation over ngp Gauss points."""
        pass


class LinearElastic(MaterialBase):
    """
    Linear isotropic elastic material under plane strain assumption (2D).

    Uses Voigt notation: strain/stress vector = [ε_xx, ε_yy, 2ε_xy].
    """

    def __init__(self, emod: float, nu: float):
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

        self.n_state = 0
        self.is_vectorized = False

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
        return torch.einsum('ij,...j->...i', C, strain)

    def compute_pk2_stress(self, gl_strain: torch.Tensor) -> torch.Tensor:
        """Compute 2nd Piola-Kirchhoff stress from Green-Lagrange strain."""
        return self.compute_stress(gl_strain)

    def vectorize(self, ngp: int):
        """
        Dummy method for interface compatibility.
        """
        self.is_vectorized = True


class LinearElasticPlaneStress(MaterialBase):
    """
    Linear isotropic elastic material under plane stress assumption (2D).

    Uses Voigt notation: strain/stress vector = [ε_xx, ε_yy, 2ε_xy].
    """

    def __init__(self, emod: float, nu: float):
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

        self.n_state = 0
        self.is_vectorized = False

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
        return torch.einsum('ij,...j->...i', C, strain)

    def compute_pk2_stress(self, gl_strain: torch.Tensor) -> torch.Tensor:
        """Second Piola-Kirchhoff stress"""
        return self.compute_stress(gl_strain)

    def vectorize(self, ngp: int):
        """Dummy method for interface compatibility."""
        self.is_vectorized = True

class J2PlasticityPlaneStrain(MaterialBase):
    """
    J2 (von Mises) plasticity with mixed linear isotropic/kinematic hardening
    under plane strain. Small strains.

    The hardening is controlled by a single parameter alpha ∈ [0, 1]:
        alpha = 0.0 → fully isotropic hardening
        alpha = 1.0 → fully kinematic hardening
        0 < alpha < 1 → linear mixture

    Internal state variables (7 per integration point):
        - ep          : plastic strain (3)
        - alpha_var   : accumulated plastic strain (isotropic variable)
        - backstress  : kinematic backstress tensor (3)
    """

    def __init__(self, E: float, nu: float, sigma_y0: float,
                 H: float = 0.0, alpha: float = 0.0):
        """
        Parameters
        ----------
        E        : float
            Young's modulus (Pa).
        nu       : float
            Poisson's ratio.
        sigma_y0 : float
            Initial yield stress (Pa).
        H        : float, optional
            Total hardening modulus (Pa). Default = 0.0 (perfect plasticity).
        alpha    : float, optional
            Kinematic hardening ratio (0 ≤ alpha ≤ 1).
            H_iso = (1 - alpha) * H
            H_kin = alpha * H
            Default = 0.0 (pure isotropic).
        """
        if E <= 0:
            raise ValueError("Young's modulus (E) must be positive.")
        if not (-1 <= nu < 0.5):
            raise ValueError("Poisson's ratio (nu) must be in [-1, 0.5).")
        if sigma_y0 < 0:
            raise ValueError("Initial yield stress must be non-negative.")
        if H < 0:
            raise ValueError("Hardening modulus H must be non-negative.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1 inclusive.")

        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H = H
        self.alpha = alpha

        # Split hardening moduli
        self.H_iso = (1.0 - alpha) * H
        self.H_kin = alpha * H

        self.G = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self._factor1 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        self._factor2 = self.lam

        # Internal state (initialized in vectorize)
        self.ep = None
        self.alpha_var = None        # accumulated plastic strain (for isotropic)
        self.backstress = None
        self.plastic_idx = None

        self.n_state = 7
        self.is_vectorized = False

    def vectorize(self, ngp: int):
        """Initialize internal state variables for ngp Gauss points."""
        device = torch.device("cpu")
        dtype = torch.float64

        self.ep = torch.zeros(ngp, 3, device=device, dtype=dtype)
        self.alpha_var = torch.zeros(ngp, device=device, dtype=dtype)
        self.backstress = torch.zeros(ngp, 3, device=device, dtype=dtype)
        self.plastic_idx = torch.zeros(ngp, dtype=torch.bool, device=device)

        self.is_vectorized = True
        self.ngp = ngp

    def _elastic_constitutive_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([[self._factor1, self._factor2, 0.0],
                             [self._factor2, self._factor1, 0.0],
                             [0.0,        0.0,       self.G]], device=device, dtype=dtype)

    def compute_stress(self, total_strain: torch.Tensor) -> tuple:
        """
        Radial return mapping integration.
        Updates internal variables in-place.
        """
        device = total_strain.device
        dtype = total_strain.dtype

        C_el = self._elastic_constitutive_matrix(device, dtype)

        # Trial elastic strain and stress
        e_el_trial = total_strain - self.ep
        sigma_trial = torch.einsum('ij,gj->gi', C_el, e_el_trial)
        xi_trial = sigma_trial - self.backstress

        # Deviatoric part (only xx and yy contribute to hydrostatic pressure in plane strain)
        hydro = (xi_trial[:, 0] + xi_trial[:, 1]) / 3.0
        dev_trial = xi_trial.clone()
        dev_trial[:, 0] -= hydro
        dev_trial[:, 1] -= hydro

        # von Mises equivalent stress
        J2 = dev_trial[:, 0]**2 + dev_trial[:, 1]**2 + 2 * dev_trial[:, 2]**2
        vm_seq = torch.sqrt(torch.clamp(J2, min=1e-24) * 1.5)

        # Current yield stress (isotropic part)
        sigma_y = self.sigma_y0 + self.H_iso * self.alpha_var
        phi = vm_seq - torch.sqrt(2.0 / 3.0) * sigma_y

        # Initialize output with trial stress (elastic by default)
        sigma = sigma_trial.clone()

        # Plastic correction
        plastic = phi > 0.0
        if torch.any(plastic):
            idx = plastic

            # Flow direction (n already includes √2 factor for shear)
            n = dev_trial[idx] / vm_seq[idx].unsqueeze(1)
            n[:, 2] *= torch.sqrt(2.0)

            # Effective hardening modulus
            H_eff = self.H_kin + (2.0 / 3.0) * self.H_iso
            denom = 2.0 * self.G + H_eff

            delta_gamma = phi[idx] / denom

            # Update stress
            sigma[idx] = sigma_trial[idx] - 2.0 * self.G * delta_gamma.unsqueeze(1) * n

            # Update internal variables
            self.ep[idx] += delta_gamma.unsqueeze(1) * n
            self.alpha_var[idx] += torch.sqrt(2.0 / 3.0) * delta_gamma
            self.backstress[idx] += self.H_kin * delta_gamma.unsqueeze(1) * n
            self.plastic_idx[idx] = True

        return sigma, self.ep, self.alpha_var, self.backstress, self.plastic_idx

    def get_constitutive_matrix(self, total_strain: torch.Tensor) -> torch.Tensor:
        """
        Consistent algorithmic tangent modulus.
        """
        device = total_strain.device
        dtype = total_strain.dtype
        C_el = self._elastic_constitutive_matrix(device, dtype)

        ngp = total_strain.shape[0] if self.is_vectorized else 1
        C_tan = C_el.unsqueeze(0).repeat(ngp, 1, 1)

        # Recompute trial state (same as in compute_stress)
        e_el_trial = total_strain - self.ep
        sigma_trial = torch.einsum('ij,gj->gi', C_el, e_el_trial)
        xi_trial = sigma_trial - self.backstress
        hydro = (xi_trial[:, 0] + xi_trial[:, 1]) / 3.0
        dev_trial = xi_trial.clone()
        dev_trial[:, 0] -= hydro
        dev_trial[:, 1] -= hydro

        J2 = dev_trial[:, 0]**2 + dev_trial[:, 1]**2 + 2 * dev_trial[:, 2]**2
        vm_seq = torch.sqrt(torch.clamp(J2, min=1e-24) * 1.5)

        sigma_y = self.sigma_y0 + self.H_iso * self.alpha_var
        phi = vm_seq - torch.sqrt(2.0 / 3.0) * sigma_y

        plastic = phi > 0.0
        if torch.any(plastic):
            idx = plastic

            n = dev_trial[idx] / vm_seq[idx].unsqueeze(1)
            n[:, 2] *= torch.sqrt(2.0)

            H_eff = self.H_kin + (2.0 / 3.0) * self.H_iso
            denom = 2.0 * self.G + H_eff
            delta_gamma = phi[idx] / denom

            theta = 1.0 - 2.0 * self.G * delta_gamma / vm_seq[idx]

            # Outer product n ⊗ n
            nn = torch.einsum('ij,ik->ijk', n, n)
            beta = (2.0 * self.G * delta_gamma / vm_seq[idx]).unsqueeze(-1).unsqueeze(-1)
            outer_term = beta * nn

            # Base elasto-plastic tangent
            C_ep = theta.unsqueeze(-1).unsqueeze(-1) * C_el + 2.0 * self.G * outer_term

            # Correction for mixed hardening
            if H_eff > 0:
                gamma = 1.0 - (2.0 * self.G * delta_gamma * H_eff) / (vm_seq[idx] * denom)
                C_ep = gamma * C_ep + (1.0 - gamma) * 2.0 * self.G * nn

            C_tan[idx] = C_ep

        return C_tan

    def compute_pk2_stress(self, gl_strain: torch.Tensor) -> tuple:
        """Second Piola-Kirchhoff stress."""
        return self.compute_stress(gl_strain)