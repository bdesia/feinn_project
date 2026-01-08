
import numpy as np
from abc import ABC, abstractmethod

class MaterialBase:
    @abstractmethod
    def get_constitutive_matrix(self):
        pass

    @abstractmethod
    def compute_stress(self):
        pass


class LinearElastic(MaterialBase):
    """
    Material elástico lineal isotrópo bajo hipótesis de plane strain (2-D).
    """

    def __init__(self, EMOD: float, nu: float):
        """
        Parámetros
        ----------
        EMOD  : módulo de Young (Pa)
        nu : coeficiente de Poisson (adimensional)
        """
        if EMOD <= 0:
            raise ValueError("El módulo de Young (E) debe ser positivo.")
        if not (-1 <= nu < 0.5):
            raise ValueError("El coeficiente de Poisson (nu) debe estar en [-1, 0.5).")

        self.EMOD = EMOD
        self.nu = nu

        # Pre-computation of parameters for constitutive matrix
        self._factor1 = EMOD * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        self._factor2 = EMOD * nu / ((1 + nu) * (1 - 2 * nu))

    def get_constitutive_matrix(self) -> np.ndarray:
        """
        Matriz constitutiva C para plane strain (3x3).
        """
        C11 = self._factor1
        C12 = self._factor2
        C33 = self.EMOD / (2 * (1 + self.nu))   # G = E/(2(1+ν))

        C = np.array([[C11, C12, 0.0],
                      [C12, C11, 0.0],
                      [0.0, 0.0, C33]])
        return C

    def compute_stress(self, strain: np.ndarray) -> np.ndarray:
        """
        σ = C ε
        """
        C = self.get_constitutive_matrix()
        return C @ strain

    def compute_pk2_stress(self, gl_strain: np.ndarray) -> np.ndarray:

        return self.compute_stress(gl_strain)

class LinearElasticPlaneStress(MaterialBase):
    """
    Material elástico lineal isotrópo bajo hipótesis de plane stress (2-D).
    """

    def __init__(self, EMOD: float, nu: float):
        """
        Parámetros
        ----------
        EMOD  : módulo de Young (Pa)
        nu : coeficiente de Poisson (adimensional)
        """
        if EMOD <= 0:
            raise ValueError("El módulo de Young (E) debe ser positivo.")
        if not (-1 <= nu < 0.5):
            raise ValueError("El coeficiente de Poisson (nu) debe estar en [-1, 0.5).")

        self.EMOD = EMOD
        self.nu = nu

        # Pre-computation of parameters for constitutive matrix
        self._factor1 = EMOD * (1 - nu**2)
    
    def get_constitutive_matrix(self) -> np.ndarray:
        """
        Matriz constitutiva C para plane stress (3x3).
        """
        C11 = self._factor1
        C12 = self._factor1 * self.nu
        C33 = self._factor1 * (1 - self.nu)/2

        C = np.array([[C11, C12, 0.0],
                      [C12, C11, 0.0],
                      [0.0, 0.0, C33]])
        return C

    def compute_stress(self, strain: np.ndarray) -> np.ndarray:
        """
        σ = C ε
        """
        C = self.get_constitutive_matrix()
        return C @ strain

    def compute_pk2_stress(self, gl_strain: np.ndarray) -> np.ndarray:

        return self.compute_stress(gl_strain)