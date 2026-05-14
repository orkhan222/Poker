"""
Value Network for state value estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValueNetworkConfig:
    """Configuration for value network"""
    input_dim: int = 115
    hidden_dims: List[int] = None
    output_dim: int = 1  # Single value output
    dropout_rate: float = 0.2
    activation: str = 'relu'
    use_batch_norm: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class ValueNetwork(nn.Module):
    """
    Value network that estimates state value (expected return).
    Used for advantage calculation in PPO/A2C.
    """
    
    def __init__(self, config: ValueNetworkConfig = None):
        super().__init__()
        
        if config is None:
            config = ValueNetworkConfig()
        self.config = config
        
        # Build layers
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(config.activation))
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer (no activation - raw value)
        self.value_head = nn.Linear(prev_dim, config.output_dim)
        
        self.apply(self._init_weights)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Value tensor of shape (batch_size, 1)
        """
        features = self.hidden_layers(x)
        value = self.value_head(features)
        return value.squeeze(-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward"""
        return self.forward(x)


class DualHeadNetwork(nn.Module):
    """
    Dual-head network that outputs both policy and value.
    Shared backbone with separate heads.
    """
    
    def __init__(
        self,
        input_dim: int = 115,
        hidden_dims: List[int] = [256, 256],
        num_actions: int = 6,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # Shared backbone
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            (policy_logits, value) tuple
        """
        shared_features = self.shared_layers(x)
        
        policy_logits = self.policy_head(shared_features)
        value = self.value_head(shared_features).squeeze(-1)
        
        return policy_logits, value
    
    def get_action_and_value(
        self, 
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Get action and value
        
        Args:
            x: Input tensor
            deterministic: If True, use argmax; else sample
            
        Returns:
            (action_idx, value, log_prob) tuple
        """
        policy_logits, value = self.forward(x)
        probs = F.softmax(policy_logits, dim=-1)
        
        if deterministic:
            action_idx = torch.argmax(probs, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx))
        
        return action_idx, value.item(), probs[0, action_idx].item()


class QuantileValueNetwork(nn.Module):
    """
    Quantile regression value network for distributional RL.
    Outputs quantiles of value distribution.
    """
    
    def __init__(
        self,
        input_dim: int = 115,
        hidden_dims: List[int] = [256, 256],
        num_quantiles: int = 32,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_quantiles = num_quantiles
        
        # Tau values (quantile fractions)
        self.tau = torch.linspace(0.01, 0.99, num_quantiles)
        
        # Network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.quantile_head = nn.Linear(prev_dim, num_quantiles)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Quantile values of shape (batch_size, num_quantiles)
        """
        features = self.feature_layers(x)
        quantiles = self.quantile_head(features)
        return quantiles
    
    def get_mean_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get mean of quantile distribution"""
        quantiles = self.forward(x)
        return quantiles.mean(dim=-1)
    
    def get_risk_adjusted_value(self, x: torch.Tensor, risk_aversion: float = 0.5) -> torch.Tensor:
        """
        Get risk-adjusted value (CVaR-like)
        
        Args:
            x: Input tensor
            risk_aversion: Higher = more risk-averse (focus on lower quantiles)
        """
        quantiles = self.forward(x)
        num_quantiles = quantiles.size(-1)
        
        # Weight lower quantiles more for risk-averse
        weights = torch.linspace(1 - risk_aversion, 1 + risk_aversion, num_quantiles)
        weights = weights / weights.sum()
        
        weighted_value = (quantiles * weights.to(quantiles.device)).sum(dim=-1)
        return weighted_value