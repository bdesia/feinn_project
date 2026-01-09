

from abc import ABC, abstractmethod
from collections import deque
import torch
import math
from typing import Dict, Any

class LossBalancer(ABC):
    """
    Clase base abstracta para estrategias de balanceo adaptativo de pérdidas.
    Define la interfaz común que todas las estrategias deben implementar.
    """
    
    def __init__(
        self,
        initial_bc_weight: float = 1e4,
        initial_force_scaler: float = 1.0,
        update_freq: int = 10,
        min_bc_weight: float = 1e2,
        max_bc_weight: float = 1e8,
        min_force_scaler: float = 0.1,
        max_force_scaler: float = 10.0,
    ):
        self.initial_bc_weight = initial_bc_weight
        self.initial_force_scaler = initial_force_scaler
        
        self.current_bc_weight = initial_bc_weight
        self.current_force_scaler = initial_force_scaler
        
        self.update_freq = update_freq
        self.min_bc_weight = min_bc_weight
        self.max_bc_weight = max_bc_weight
        self.min_force_scaler = min_force_scaler
        self.max_force_scaler = max_force_scaler
        
        self.epoch_counter = 0

    @abstractmethod
    def update(
        self,
        grad_flow: Dict[str, Dict[str, torch.Tensor]],
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Actualiza los pesos de balanceo basándose en gradientes y pérdidas del epoch actual.
        
        Args:
            grad_flow: Diccionario con claves 'Domain', 'BoundaryConditions', etc.
                       Cada valor es un dict {param_name: gradient_tensor}
            losses: Diccionario con pérdidas escalares (e.g. {'Domain': loss_domain, 'BoundaryConditions': loss_bc})
        
        Returns:
            Dict con información del update (pesos nuevos, si se actualizó, métricas, etc.)
        """
        pass

    def get_current_weights(self) -> Dict[str, float]:
        """Devuelve los pesos actuales para usar en el cálculo de pérdida."""
        return {
            'bc_weight': self.current_bc_weight,
            'force_scaler': self.current_force_scaler
        }

    def reset(self):
        """Resetea el estado (útil si reentrenas el modelo)."""
        self.epoch_counter = 0
        self.current_bc_weight = self.initial_bc_weight
        self.current_force_scaler = self.initial_force_scaler
        

class GradNormBalancer(LossBalancer):
    """
    Implementación de GradNorm para balanceo adaptativo.
    
    Basado en:
    - Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing..." (2018)
    - Adaptaciones comunes en PINNs (Bischof & Kraus, 2021; etc.)
    
    Idea principal:
    Ajustar el peso del término BC para que la norma del gradiente proveniente
    del loss_domain sea similar a la del loss_bc.
    """
    
    def __init__(
        self,
        initial_bc_weight: float = 1e4,
        initial_force_scaler: float = 1.0,
        update_freq: int = 20,
        alpha: float = 0.5,              # Exponente de suavizado (típicamente 0.5 o 1.0)
        patience: int = 50,              # Ventana para promedio móvil de normas
        **kwargs
    ):
        super().__init__(
            initial_bc_weight=initial_bc_weight,
            initial_force_scaler=initial_force_scaler,
            update_freq=update_freq,
            **kwargs
        )
        self.alpha = alpha
        self.patience = patience
        
        # Buffers para promedio móvil
        self.domain_grad_norms = deque(maxlen=patience)
        self.bc_grad_norms = deque(maxlen=patience)

    def _compute_grad_norm(self, grad_dict: Dict[str, torch.Tensor]) -> float:
        """Calcula la norma L2 promedio de los gradientes (solo pesos)."""
        if not grad_dict:
            return 0.0
        total_norm_sq = 0.0
        count = 0
        for name, grad in grad_dict.items():
            if grad is not None and 'weight' in name:  # Solo pesos, ignorar biases
                total_norm_sq += (grad.norm(p=2) ** 2).item()
                count += 1
        return math.sqrt(total_norm_sq / (count + 1e-12)) if count > 0 else 0.0

    def update(
        self,
        grad_flow: Dict[str, Dict[str, torch.Tensor]],
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self.epoch_counter += 1
        
        # Solo actualizar cada update_freq épocas
        if self.epoch_counter % self.update_freq != 0:
            return {
                'updated': False,
                'bc_weight': self.current_bc_weight,
                'force_scaler': self.current_force_scaler
            }

        domain_grad = self._compute_grad_norm(grad_flow.get('Domain', {}))
        bc_grad = self._compute_grad_norm(grad_flow.get('BoundaryConditions', {}))

        # Evitar división por cero
        domain_grad = max(domain_grad, 1e-8)
        bc_grad = max(bc_grad, 1e-8)

        self.domain_grad_norms.append(domain_grad)
        self.bc_grad_norms.append(bc_grad)

        updated = False
        info = {
            'grad_norm_domain': domain_grad,
            'grad_norm_bc': bc_grad,
            'updated': False,
            'bc_weight': self.current_bc_weight,
            'force_scaler': self.current_force_scaler
        }

        # Necesitamos al menos algunas muestras para promediar
        if len(self.domain_grad_norms) < 5:
            return info

        avg_domain_grad = sum(self.domain_grad_norms) / len(self.domain_grad_norms)
        avg_bc_grad = sum(self.bc_grad_norms) / len(self.bc_grad_norms)

        # Ratio de normas de gradiente
        ratio = avg_domain_grad / avg_bc_grad
        factor = ratio ** self.alpha  # Suavizado con alpha

        new_bc_weight = self.current_bc_weight * factor
        new_bc_weight = torch.clamp(
            torch.tensor(new_bc_weight),
            self.min_bc_weight,
            self.max_bc_weight
        ).item()

        if abs(new_bc_weight - self.current_bc_weight) > 1e-2 * self.current_bc_weight:
            self.current_bc_weight = new_bc_weight
            updated = True

        info.update({
            'updated': updated,
            'bc_weight': self.current_bc_weight,
            'factor': factor,
            'ratio': ratio,
            'avg_grad_domain': avg_domain_grad,
            'avg_grad_bc': avg_bc_grad
        })

        return info

class ForceScalerGradNormBalancer(LossBalancer):
    """
    """
    
    def __init__(
        self,
        initial_bc_weight: float = 1e4,
        initial_force_scaler: float = 1.0,
        update_freq: int = 20,
        alpha: float = 0.5,
        patience: int = 50,
        min_force_scaler: float = 0.1,
        max_force_scaler: float = 100.0,   # Puedes permitir valores más altos si querés
        **kwargs
    ):
        super().__init__(
            initial_bc_weight=initial_bc_weight,
            initial_force_scaler=initial_force_scaler,
            update_freq=update_freq,
            min_bc_weight=initial_bc_weight,      # No lo vamos a cambiar
            max_bc_weight=initial_bc_weight,
            min_force_scaler=min_force_scaler,
            max_force_scaler=max_force_scaler,
            **kwargs
        )
        self.alpha = alpha
        self.patience = patience
        
        self.domain_grad_norms = deque(maxlen=patience)
        self.bc_grad_norms = deque(maxlen=patience)

    def _compute_grad_norm(self, grad_dict: Dict[str, torch.Tensor]) -> float:
        # (mismo método que antes)
        if not grad_dict:
            return 0.0
        total_norm_sq = 0.0
        count = 0
        for name, grad in grad_dict.items():
            if grad is not None and 'weight' in name:
                total_norm_sq += (grad.norm(p=2) ** 2).item()
                count += 1
        return math.sqrt(total_norm_sq / (count + 1e-12)) if count > 0 else 0.0

    def update(
        self,
        grad_flow: Dict[str, Dict[str, torch.Tensor]],
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self.epoch_counter += 1
        
        if self.epoch_counter % self.update_freq != 0:
            return {
                'updated': False,
                'bc_weight': self.current_bc_weight,
                'force_scaler': self.current_force_scaler
            }

        domain_grad = self._compute_grad_norm(grad_flow.get('Domain', {}))
        bc_grad = self._compute_grad_norm(grad_flow.get('BoundaryConditions', {}))

        domain_grad = max(domain_grad, 1e-8)
        bc_grad = max(bc_grad, 1e-8)

        self.domain_grad_norms.append(domain_grad)
        self.bc_grad_norms.append(bc_grad)

        if len(self.domain_grad_norms) < 5:
            return {
                'updated': False,
                'bc_weight': self.current_bc_weight,
                'force_scaler': self.current_force_scaler,
                'grad_norm_domain': domain_grad,
                'grad_norm_bc': bc_grad
            }

        avg_domain_grad = sum(self.domain_grad_norms) / len(self.domain_grad_norms)
        avg_bc_grad = sum(self.bc_grad_norms) / len(self.bc_grad_norms)

        # Queremos que grad_domain ≈ grad_bc
        # Si grad_domain < grad_bc → el domain está "demasiado fácil" → aumentar force_scaler
        ratio = avg_domain_grad / avg_bc_grad
        factor = ratio ** self.alpha

        new_force_scaler = self.current_force_scaler * factor
        new_force_scaler = torch.clamp(
            torch.tensor(new_force_scaler),
            self.min_force_scaler,
            self.max_force_scaler
        ).item()

        updated = abs(new_force_scaler - self.current_force_scaler) > 1e-2 * self.current_force_scaler
        if updated:
            self.current_force_scaler = new_force_scaler

        return {
            'updated': updated,
            'bc_weight': self.current_bc_weight,        # fijo
            'force_scaler': self.current_force_scaler,  # actualizado
            'factor': factor,
            'ratio': ratio,
            'avg_grad_domain': avg_domain_grad,
            'avg_grad_bc': avg_bc_grad,
            'grad_norm_domain': domain_grad,
            'grad_norm_bc': bc_grad
        }