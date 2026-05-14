"""
Neural network policy agent for poker
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .base_agent import BaseAgent, AgentAction
from src.models.policy_net import PolicyNetwork
from src.data.features import PokerFeatureExtractor, PokerState


class PolicyAgent(BaseAgent):
    """
    Neural network policy agent.
    Uses a trained policy network to make decisions.
    """
    
    def __init__(
        self,
        name: str = "PolicyAgent",
        model_path: Optional[str] = None,
        device: str = 'cpu',
        deterministic: bool = False,
        feature_extractor: Optional[PokerFeatureExtractor] = None
    ):
        super().__init__(name)
        
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.feature_extractor = feature_extractor or PokerFeatureExtractor()
        
        # Initialize policy network
        self.model = PolicyNetwork().to(self.device)
        
        # Load pretrained weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.model.eval()
        
        # Action mapping
        self.action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
    
    def load_model(self, model_path: str):
        """Load pretrained model weights"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), model_path)
        print(f"✅ Saved model to {model_path}")
    
    def _create_poker_state(self, state: Dict) -> PokerState:
        """Convert raw state dict to PokerState object"""
        parsed = self._parse_state(state)
        
        return PokerState(
            hole_cards=parsed['hole_cards'],
            board_cards=parsed['board_cards'],
            agent_stack=parsed['agent_stack'],
            pot=parsed['pot'],
            current_bet=parsed['current_bet'],
            street=parsed['street'],
            legal_actions=parsed['legal_actions'],
            opponent_stacks=parsed.get('opponent_stacks', {}),
            opponent_waiting_time=parsed.get('opponent_waiting_time'),
            time_since_agent_bet=parsed.get('time_since_agent_bet')
        )
    
    def _get_action_probs(self, state: PokerState) -> np.ndarray:
        """Get action probabilities from policy network"""
        with torch.no_grad():
            tensor = self.feature_extractor.state_to_tensor(state)
            tensor = tensor.unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()[0]
    
    def _sample_action(self, probs: np.ndarray, legal_actions: List[str]) -> Tuple[int, str]:
        """Sample action from probability distribution"""
        # Create mask for legal actions
        mask = np.zeros(len(self.action_names))
        for action in legal_actions:
            for i, name in enumerate(self.action_names):
                if action.lower() == name:
                    mask[i] = 1.0
                    break
        
        if mask.sum() == 0:
            return 2, 'call'  # Default to call if no legal actions found
        
        # Apply mask and renormalize
        masked_probs = probs * mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = mask / mask.sum()
        
        if self.deterministic:
            action_idx = np.argmax(masked_probs)
        else:
            action_idx = np.random.choice(len(self.action_names), p=masked_probs)
        
        return action_idx, self.action_names[action_idx]
    
    def _calculate_bet_size(self, action: str, state: Dict) -> float:
        """Calculate bet/raise size"""
        parsed = self._parse_state(state)
        
        if action == 'bet':
            # Standard bet sizes based on pot
            pot = parsed['pot']
            if parsed['street'] == 'preflop':
                return min(pot * 0.75, parsed['agent_stack'])
            elif parsed['street'] == 'flop':
                return min(pot * 0.66, parsed['agent_stack'])
            elif parsed['street'] == 'turn':
                return min(pot * 0.66, parsed['agent_stack'])
            else:  # river
                return min(pot * 0.5, parsed['agent_stack'])
        
        elif action == 'raise':
            # Raise based on current bet
            current_bet = parsed['current_bet']
            min_raise = parsed['min_raise']
            pot = parsed['pot']
            
            # Standard raise sizes
            raise_size = current_bet + max(min_raise, pot * 0.5)
            return min(raise_size, parsed['agent_stack'])
        
        elif action == 'call':
            return parsed['current_bet']
        
        elif action == 'all_in':
            return parsed['agent_stack']
        
        return 0.0
    
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """Make decision using policy network"""
        parsed = self._parse_state(state)
        
        # Create PokerState and get action probabilities
        poker_state = self._create_poker_state(state)
        probs = self._get_action_probs(poker_state)
        
        # Sample action
        action_idx, action_name = self._sample_action(probs, parsed['legal_actions'])
        
        # Calculate bet size if needed
        bet_size = 0.0
        if action_name in ['bet', 'raise', 'call', 'all_in']:
            bet_size = self._calculate_bet_size(action_name, state)
        
        # Get confidence
        confidence = float(probs[action_idx])
        
        # Create action
        agent_action = AgentAction(
            action=action_name,
            bet_size=bet_size,
            confidence=confidence
        )
        
        # Record
        self.action_history.append({
            'state': parsed,
            'action': action_name,
            'bet_size': bet_size,
            'confidence': confidence,
            'probs': probs.tolist()
        })
        
        return agent_action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.action_history = []
    
    def set_training_mode(self, training: bool):
        """Set model to training or evaluation mode"""
        if training:
            self.model.train()
        else:
            self.model.eval()


class PretrainedPolicyAgent(PolicyAgent):
    """Policy agent with specific pretrained model"""
    
    def __init__(
        self,
        model_name: str = "default",
        **kwargs
    ):
        model_paths = {
            'default': 'experiments/checkpoints/best_model.pt',
            'rl': 'experiments/checkpoints/rl_model_final.pt',
            'self_play': 'experiments/checkpoints/self_play_model.pt'
        }
        
        model_path = model_paths.get(model_name, model_paths['default'])
        super().__init__(model_path=model_path, **kwargs)
        self.model_name = model_name