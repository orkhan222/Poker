"""
CRF-style and Minimax losses for robust poker training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class CRFLoss(nn.Module):
    """
    Conditional Random Field style loss for sequential action prediction.
    Models dependencies between consecutive actions.
    """
    
    def __init__(
        self,
        num_actions: int = 6,
        transition_weight: float = 0.1,
        use_transition_matrix: bool = True
    ):
        """
        Args:
            num_actions: Number of possible actions
            transition_weight: Weight for transition loss
            use_transition_matrix: Learn transition matrix between actions
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.transition_weight = transition_weight
        
        if use_transition_matrix:
            # Learnable transition matrix (log space)
            self.transition_matrix = nn.Parameter(
                torch.zeros(num_actions, num_actions)
            )
        else:
            self.register_buffer('transition_matrix', torch.zeros(num_actions, num_actions))
    
    def forward(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute CRF loss
        
        Args:
            logits: Action logits of shape (batch_size, num_actions)
            actions: True action indices of shape (batch_size,)
            prev_actions: Previous action indices (for sequence)
            
        Returns:
            Loss value
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, actions)
        
        if prev_actions is not None and self.transition_weight > 0:
            # Transition loss
            batch_size = len(actions)
            transition_scores = self.transition_matrix[prev_actions, actions]
            transition_loss = -transition_scores.mean()
            
            # Combine losses
            total_loss = ce_loss + self.transition_weight * transition_loss
        else:
            total_loss = ce_loss
        
        return total_loss


class MinimaxLoss(nn.Module):
    """
    Minimax loss for robust policy optimization.
    Optimizes for worst-case opponent strategy.
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        use_entropy_bonus: bool = True,
        entropy_coef: float = 0.01
    ):
        """
        Args:
            epsilon: Robustness parameter (perturbation size)
            temperature: Temperature for softmax
            use_entropy_bonus: Add entropy bonus for exploration
            entropy_coef: Entropy coefficient
        """
        super().__init__()
        
        self.epsilon = epsilon
        self.temperature = temperature
        self.use_entropy_bonus = use_entropy_bonus
        self.entropy_coef = entropy_coef
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        advantages: torch.Tensor,
        old_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute minimax loss
        
        Args:
            policy_logits: Current policy logits
            advantages: Advantage values
            old_logits: Previous policy logits (for KL constraint)
            
        Returns:
            Loss value
        """
        # Get action probabilities
        probs = F.softmax(policy_logits / self.temperature, dim=-1)
        
        # Worst-case opponent strategy (adversarial)
        # Perturb probabilities in the direction of lowest advantage
        if advantages is not None:
            # Create adversarial perturbation
            adv_expanded = advantages.unsqueeze(-1).expand_as(probs)
            perturbation = self.epsilon * adv_expanded * (1 - probs)
            adversarial_probs = probs - perturbation
            adversarial_probs = F.relu(adversarial_probs)
            adversarial_probs = adversarial_probs / adversarial_probs.sum(dim=-1, keepdim=True)
            
            # Loss against adversarial opponent
            policy_loss = -(adversarial_probs.detach() * torch.log(probs + 1e-8)).sum(dim=-1)
            policy_loss = (policy_loss * advantages).mean()
        else:
            # Standard policy gradient
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(log_probs * advantages.unsqueeze(-1)).sum(dim=-1).mean()
        
        # KL constraint with old policy
        if old_logits is not None:
            old_probs = F.softmax(old_logits, dim=-1)
            kl_div = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(probs + 1e-8))).sum(dim=-1)
            kl_loss = 0.5 * (kl_div ** 2).mean()
            policy_loss = policy_loss + kl_loss
        
        # Entropy bonus
        if self.use_entropy_bonus:
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            policy_loss = policy_loss - self.entropy_coef * entropy
        
        return policy_loss


class NashEquilibriumLoss(nn.Module):
    """
    Nash equilibrium loss for two-player zero-sum games.
    Minimizes exploitability.
    """
    
    def __init__(
        self,
        num_actions: int = 6,
        regularization: float = 0.001
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.regularization = regularization
    
    def forward(
        self,
        our_policy: torch.Tensor,
        opp_policy: torch.Tensor,
        payoff_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Nash equilibrium loss
        
        Args:
            our_policy: Our policy probabilities (batch_size, num_actions)
            opp_policy: Opponent policy probabilities (batch_size, num_actions)
            payoff_matrix: Payoff matrix (num_actions, num_actions)
            
        Returns:
            Exploitability loss
        """
        batch_size = our_policy.size(0)
        
        if payoff_matrix is None:
            # Default payoff: higher value for beating opponent
            # Simplified: rock-paper-scissors style
            payoff_matrix = self._create_default_payoff().to(our_policy.device)
        
        # Compute expected payoff
        # E[payoff] = our_policy^T * payoff_matrix * opp_policy
        expected_payoff = torch.einsum('bi,ij,bj->b', our_policy, payoff_matrix, opp_policy)
        
        # Best response to opponent
        best_response_value = torch.max(
            torch.einsum('ij,bj->bi', payoff_matrix, opp_policy), 
            dim=-1
        )[0]
        
        # Exploitability = how much opponent can gain by deviating
        exploitability = (best_response_value - expected_payoff).mean()
        
        # Add regularization to keep policies smooth
        reg_loss = self.regularization * (
            -(our_policy * torch.log(our_policy + 1e-8)).sum(dim=-1).mean() +
            -(opp_policy * torch.log(opp_policy + 1e-8)).sum(dim=-1).mean()
        )
        
        total_loss = exploitability + reg_loss
        
        return total_loss
    
    def _create_default_payoff(self) -> torch.Tensor:
        """Create default rock-paper-scissors style payoff matrix"""
        # Actions: fold, check, call, bet, raise, all_in
        # Higher values mean better for row player (us)
        payoff = torch.zeros(self.num_actions, self.num_actions)
        
        # Bet beats check
        payoff[3, 1] = 1.0  # bet vs check
        # Raise beats call and bet
        payoff[4, 2] = 0.8
        payoff[4, 3] = 0.5
        # All-in beats raise
        payoff[5, 4] = 1.0
        # Fold loses to everything
        payoff[:, 0] = -1.0
        
        return payoff


class AdversarialLoss(nn.Module):
    """
    Adversarial training loss for robust agent.
    Alternates between policy and adversary optimization.
    """
    
    def __init__(
        self,
        lambda_adv: float = 0.1,
        eps: float = 0.05
    ):
        super().__init__()
        
        self.lambda_adv = lambda_adv
        self.eps = eps
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        rewards: torch.Tensor,
        done_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial loss
        
        Returns:
            (policy_loss, value_loss) tuple
        """
        # Value loss (standard MSE)
        value_loss = F.mse_loss(value_pred, rewards)
        
        # Policy loss with adversarial perturbation
        probs = F.softmax(policy_logits, dim=-1)
        
        # Add adversarial noise to state (simulated)
        # In practice, you'd perturb the input features
        adv_noise = torch.randn_like(policy_logits) * self.eps
        adv_logits = policy_logits + adv_noise
        
        adv_probs = F.softmax(adv_logits, dim=-1)
        
        # KL divergence between clean and adversarial policy
        kl_div = (probs * (torch.log(probs + 1e-8) - torch.log(adv_probs + 1e-8))).sum(dim=-1)
        
        policy_loss = -(probs * rewards.unsqueeze(-1)).sum(dim=-1).mean()
        policy_loss = policy_loss + self.lambda_adv * kl_div.mean()
        
        return policy_loss, value_loss


class RobustPPOLoss(nn.Module):
    """
    Robust PPO loss with confidence bounds and clipping.
    """
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_robust_clipping: bool = True
    ):
        super().__init__()
        
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_robust_clipping = use_robust_clipping
    
    def forward(
        self,
        new_logits: torch.Tensor,
        old_logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """
        Compute robust PPO loss
        
        Returns:
            (total_loss, info_dict) tuple
        """
        # Get probabilities
        new_probs = F.softmax(new_logits, dim=-1)
        old_probs = F.softmax(old_logits, dim=-1)
        
        # Probability ratios
        new_action_probs = new_probs[range(len(actions)), actions]
        old_action_probs = old_probs[range(len(actions)), actions]
        
        ratio = new_action_probs / (old_action_probs + 1e-8)
        
        # Clipped surrogate objective
        if self.use_robust_clipping:
            # Adaptive clipping based on advantage magnitude
            adv_std = advantages.std().item()
            dynamic_eps = self.clip_epsilon * (1 + adv_std)
            clipped_ratio = torch.clamp(ratio, 1 - dynamic_eps, 1 + dynamic_eps)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss (clipped)
        value_pred_clipped = values + torch.clamp(
            values - values.detach(),
            -self.clip_epsilon,
            self.clip_epsilon
        )
        
        value_loss = F.mse_loss(values, returns)
        value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
        value_loss = torch.max(value_loss, value_loss_clipped) * self.value_coef
        
        # Entropy bonus
        entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'advantage_mean': advantages.mean().item()
        }
        
        return total_loss, info