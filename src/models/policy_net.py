"""
Policy Network for action probability prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PolicyNetworkConfig:
    """Configuration for policy network"""
    input_dim: int = 115  # Feature dimension
    hidden_dims: List[int] = None  # Hidden layer dimensions
    output_dim: int = 6  # Number of actions (fold, check, call, bet, raise, all_in)
    dropout_rate: float = 0.3
    activation: str = 'relu'
    use_batch_norm: bool = True
    use_residual: bool = False
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    
    Input: Feature vector from game state
    Output: Logits for 6 poker actions
    """
    
    def __init__(self, config: PolicyNetworkConfig = None):
        super().__init__()
        
        if config is None:
            config = PolicyNetworkConfig()
        self.config = config
        
        # Build layers
        layers = []
        prev_dim = config.input_dim
        
        for i, hidden_dim in enumerate(config.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation(config.activation))
            
            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, config.output_dim)
        
        # Residual connection
        self.use_residual = config.use_residual
        if config.use_residual:
            self.residual_proj = nn.Linear(config.input_dim, config.hidden_dims[-1])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits tensor of shape (batch_size, output_dim)
        """
        # Residual connection
        if self.use_residual:
            residual = self.residual_proj(x)
            x = self.hidden_layers(x)
            x = x + residual
        else:
            x = self.hidden_layers(x)
        
        # Output layer
        logits = self.output_layer(x)
        
        return logits
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities from logits
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor of shape (batch_size, output_dim)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and its probability
        
        Args:
            x: Input tensor
            deterministic: If True, return argmax action; else sample
            
        Returns:
            (action_indices, action_probs) tuple
        """
        probs = self.get_action_probs(x)
        
        if deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            actions = torch.multinomial(probs, 1).squeeze(-1)
        
        action_probs = probs[range(len(actions)), actions]
        
        return actions, action_probs


class MultiHeadPolicyNetwork(nn.Module):
    """
    Multi-head policy network with separate heads for:
    - Action type prediction
    - Bet size prediction (regression)
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
        
        # Action head
        self.action_head = nn.Linear(hidden_dims[1], num_actions)
        
        # Bet size head (for bet/raise actions)
        self.bet_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1] - multiply by max_bet
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
            (action_logits, bet_size_predictions) tuple
        """
        shared_features = self.shared_layers(x)
        
        action_logits = self.action_head(shared_features)
        bet_predictions = self.bet_head(shared_features)
        
        return action_logits, bet_predictions
    
    def get_action_and_bet(
        self, 
        x: torch.Tensor, 
        max_bet: float = 1000.0,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Get action and bet size
        
        Args:
            x: Input tensor
            max_bet: Maximum allowed bet (player's stack)
            deterministic: If True, use argmax; else sample
            
        Returns:
            (action_idx, bet_size, confidence) tuple
        """
        action_logits, bet_ratio = self.forward(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1).item()
        else:
            action_idx = torch.multinomial(action_probs, 1).item()
        
        bet_size = bet_ratio.item() * max_bet
        confidence = action_probs[0, action_idx].item()
        
        return action_idx, bet_size, confidence


class LSTMPolicyNetwork(nn.Module):
    """
    LSTM-based policy network for sequential action prediction.
    Uses action history to inform decisions.
    """
    
    def __init__(
        self,
        input_dim: int = 115,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 6,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden_state: Optional initial hidden state
            
        Returns:
            (logits, new_hidden_state) tuple
        """
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        
        # Attention over sequence
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        logits = self.output_layer(context)
        
        return logits, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)