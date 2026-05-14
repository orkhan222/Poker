"""
Feature extraction module - Convert poker state to tensors
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PokerState:
    """Poker state representation for feature extraction"""
    hole_cards: List[str]
    board_cards: List[str]
    agent_stack: float
    pot: float
    current_bet: float
    street: str
    legal_actions: List[str]
    opponent_stacks: Optional[Dict[str, float]] = None
    opponent_waiting_time: Optional[float] = None
    time_since_agent_bet: Optional[float] = None


class PokerFeatureExtractor:
    """
    Convert poker game state to tensor for neural network input
    """
    
    def __init__(self, max_opponents: int = 5):
        self.max_opponents = max_opponents
        
        # Card mapping (52 cards)
        self.rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                         '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        self.suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
        
        # Street mapping
        self.street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        
        # Action mapping for legal actions
        self.action_map = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}
        
        # Fixed feature dimension: 115
        self.feature_dim = 115
    
    def _cards_to_one_hot(self, cards: List[str]) -> List[float]:
        """Convert list of cards to 52-dim one-hot vector"""
        one_hot = [0.0] * 52
        
        for card in cards:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1] if len(card) == 2 else card[2]
                
                if rank in self.rank_map and suit in self.suit_map:
                    idx = self.rank_map[rank] * 4 + self.suit_map[suit]
                    if idx < 52:
                        one_hot[idx] = 1.0
        
        return one_hot
    
    def _street_to_one_hot(self, street: str) -> List[float]:
        """Convert street to 4-dim one-hot vector"""
        one_hot = [0.0, 0.0, 0.0, 0.0]
        if street.lower() in self.street_map:
            idx = self.street_map[street.lower()]
            if idx < 4:
                one_hot[idx] = 1.0
        else:
            one_hot[0] = 1.0
        return one_hot
    
    def _legal_actions_to_vector(self, legal_actions: List[str]) -> List[float]:
        """Convert legal actions to 6-dim binary vector"""
        vector = [0.0] * 6
        for action in legal_actions:
            if action.lower() in self.action_map:
                idx = self.action_map[action.lower()]
                if idx < 6:
                    vector[idx] = 1.0
        return vector
    
    def state_to_tensor(self, state: PokerState) -> torch.Tensor:
        """
        Convert poker state to tensor - FIXED 115 DIMENSIONS
        
        Returns:
            Tensor of shape (115,)
        """
        features = []
        
        # 1. Hole cards: 52 dimensions
        features.extend(self._cards_to_one_hot(state.hole_cards))
        
        # 2. Board cards: 52 dimensions
        features.extend(self._cards_to_one_hot(state.board_cards))
        
        # 3. Agent stack (normalized to [0,1]): 1 dimension
        features.append(min(state.agent_stack / 10000, 1.0))
        
        # 4. Pot size (normalized): 1 dimension
        features.append(min(state.pot / 20000, 1.0))
        
        # 5. Current bet (normalized): 1 dimension
        features.append(min(state.current_bet / 5000, 1.0))
        
        # 6. Street (one-hot): 4 dimensions
        features.extend(self._street_to_one_hot(state.street))
        
        # 7. Legal actions: 6 dimensions
        features.extend(self._legal_actions_to_vector(state.legal_actions))
        
        # Current total: 52 + 52 + 1 + 1 + 1 + 4 + 6 = 117
        # Need to reduce to 115, so we'll combine or remove 2 dimensions
        
        # Instead, let's use exact 115 dimensions:
        # 52 (hole) + 52 (board) + 1 (stack) + 1 (pot) + 1 (bet) + 4 (street) + 4 (simplified legal) = 115
        
        # Rebuild with exact 115 dimensions
        features = []
        
        # 1. Hole cards: 52
        features.extend(self._cards_to_one_hot(state.hole_cards))
        
        # 2. Board cards: 52
        features.extend(self._cards_to_one_hot(state.board_cards))
        
        # 3. Agent stack: 1
        features.append(min(state.agent_stack / 10000, 1.0))
        
        # 4. Pot size: 1
        features.append(min(state.pot / 20000, 1.0))
        
        # 5. Current bet: 1
        features.append(min(state.current_bet / 5000, 1.0))
        
        # 6. Street: 4
        features.extend(self._street_to_one_hot(state.street))
        
        # 7. Simplified legal actions (only 4 most important): fold, call, raise, check
        legal_simplified = [0.0, 0.0, 0.0, 0.0]
        for action in state.legal_actions:
            if action.lower() == 'fold':
                legal_simplified[0] = 1.0
            elif action.lower() == 'call':
                legal_simplified[1] = 1.0
            elif action.lower() == 'raise':
                legal_simplified[2] = 1.0
            elif action.lower() == 'check':
                legal_simplified[3] = 1.0
        features.extend(legal_simplified)
        
        # Total: 52 + 52 + 1 + 1 + 1 + 4 + 4 = 115
        # Exactly 115 dimensions!
        
        # Verify dimension
        if len(features) != 115:
            print(f"Warning: Feature dimension is {len(features)}, expected 115")
            # Pad or truncate to 115
            if len(features) < 115:
                features.extend([0.0] * (115 - len(features)))
            else:
                features = features[:115]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def batch_states_to_tensor(self, states: List[PokerState]) -> torch.Tensor:
        """Convert batch of states to tensor"""
        batch_features = [self.state_to_tensor(state).numpy() for state in states]
        return torch.tensor(np.array(batch_features), dtype=torch.float32)


def create_state_tensor(
    hole_cards: List[str],
    board_cards: List[str],
    agent_stack: float,
    pot: float,
    current_bet: float,
    street: str,
    legal_actions: List[str],
    opponent_stacks: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """Convenience function to create state tensor"""
    state = PokerState(
        hole_cards=hole_cards,
        board_cards=board_cards,
        agent_stack=agent_stack,
        pot=pot,
        current_bet=current_bet,
        street=street,
        legal_actions=legal_actions,
        opponent_stacks=opponent_stacks
    )
    
    extractor = PokerFeatureExtractor()
    return extractor.state_to_tensor(state)