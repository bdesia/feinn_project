import abc
import math
import random
import torch
from typing import Dict, Any


class LossBalancer(abc.ABC):
    """
    Abstract base class for dynamic loss balancing strategies.
    Defines the standard interface for updating loss weights during training.
    """
    def __init__(self, **kwargs):
        self.epoch_counter = 0

    @abc.abstractmethod
    def update(
        self,
        grad_flow: Dict[str, Dict[str, torch.Tensor]],
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Calculates and updates the adaptive weights for the loss components.

        Args:
            grad_flow: Dictionary of gradients (can be empty if not required by the strategy).
            losses: Dictionary containing the current unweighted loss scalars.

        Returns:
            Dict containing at least:
                - 'updated' (bool): Whether weights were successfully updated.
                - 'weights' (Dict[str, float]): The new adaptive weights for each loss term.
        """
        pass


class ReLoBraLoBalancer(LossBalancer):
    """
    Relative Loss Balancing with Random Lookback (ReLoBraLo).
    Adapts loss weights based on the historical evolution of loss components 
    without requiring additional gradient computations.
    """
    def __init__(
        self,
        alpha: float = 0.5,             # Balances short-term vs long-term history
        rho: float = 0.99,              # EMA memory factor for weights
        temperature: float = 0.1,       # Softmax temperature for weight distribution
        lookback_prob: float = 0.05,    # Probability to update the historical reference (tR)
        **kwargs
    ):
        # Initialize the abstract base class (sets self.epoch_counter = 0)
        super().__init__(**kwargs) 
        
        self.alpha = alpha
        self.rho = rho
        self.temperature = temperature
        self.lookback_prob = lookback_prob
        
        # Historical loss records
        self.loss_0: Dict[str, float] = {}   # Losses at initial step L(0)
        self.loss_tR: Dict[str, float] = {}  # Losses at last lookback step L(tR)
        
        # Current adaptive weights (λ_i)
        self.lambdas: Dict[str, float] = {}
        
    def update(
        self,
        grad_flow: Dict[str, Dict[str, torch.Tensor]], # Unused in ReLoBraLo, required by interface
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Calculates and updates adaptive weights based on current loss magnitudes.
        """
        self.epoch_counter += 1
        
        # Extract scalar values and ignore empty loss components (e.g., values effectively zero)
        current_losses = {k: v.item() for k, v in losses.items() if v.item() > 1e-12}
        
        if not current_losses:
            return {'updated': False, 'weights': self.lambdas, 'lookback_triggered': False}

        # Initialize historical states on the first iteration
        if not self.loss_0:
            self.loss_0 = current_losses.copy()
            self.loss_tR = current_losses.copy()
            self.lambdas = {k: 1.0 for k in current_losses.keys()}
            return {'updated': True, 'weights': self.lambdas.copy(), 'lookback_triggered': False}

        # Compute unnormalized suggested weights (lambda_hat)
        lambda_hat = {}
        unnormalized_weights = {}
        total_exp = 0.0
        
        for k, loss_t in current_losses.items():
            # Handle late-appearing loss terms safely
            if k not in self.loss_0:
                self.loss_0[k] = loss_t
                self.loss_tR[k] = loss_t
                self.lambdas[k] = 1.0
                
            # Compute relative losses (short and long term)
            ratio_short = loss_t / (self.loss_tR[k] + 1e-8)
            ratio_long = loss_t / (self.loss_0[k] + 1e-8)
            
            # Expected value before softmax
            val = (self.alpha * ratio_short + (1 - self.alpha) * ratio_long) / self.temperature
            
            # Clip to prevent exponent overflow
            val = min(val, 50.0)
            exp_val = math.exp(val)
            
            unnormalized_weights[k] = exp_val
            total_exp += exp_val
            
        # Apply Softmax scaling and Exponential Moving Average (EMA)
        num_losses = len(current_losses)
        for k in current_losses.keys():
            if total_exp > 0:
                # Scale by num_losses to keep overall learning rate consistent
                lambda_hat[k] = (unnormalized_weights[k] / total_exp) * num_losses
            else:
                lambda_hat[k] = 1.0
                
            # EMA blending
            self.lambdas[k] = self.rho * self.lambdas.get(k, 1.0) + (1 - self.rho) * lambda_hat[k]

        # Trigger Random Lookback stochastically
        lookback_triggered = random.random() < self.lookback_prob
        if lookback_triggered:
            self.loss_tR = current_losses.copy()
            
        return {
            'updated': True, 
            'weights': self.lambdas.copy(),
            'lookback_triggered': lookback_triggered
        }