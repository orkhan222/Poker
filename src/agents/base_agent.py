"""
Abstract base class for all poker agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Standard action types"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    action: str  # fold, check, call, bet, raise, all_in
    bet_size: float = 0.0
    confidence: float = 0.0
    reasoning: Optional[str] = None
    
    def to_tuple(self) -> Tuple[str, float]:
        return (self.action, self.bet_size)
    
    def to_dict(self) -> Dict:
        return {
            'action': self.action,
            'bet_size': self.bet_size,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }


class BaseAgent(ABC):
    """
    Abstract base class for all poker agents.
    All agents must implement the act method.
    """
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.hand_count = 0
        self.action_history = []
        self.total_profit = 0.0
        
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """
        Decide on an action based on the current game state.
        
        Args:
            state: Dictionary containing game state information:
                - hole_cards: List of agent's cards
                - board_cards: List of community cards
                - agent_stack: Agent's remaining chips
                - pot: Current pot size
                - current_bet: Current bet amount to call
                - street: Current street (preflop, flop, turn, river)
                - legal_actions: List of legal actions
                - opponent_stacks: Dict of opponent stacks
                - opponent_waiting_time: Optional waiting time info
                - action_history: Previous actions in this hand
                
        Returns:
            AgentAction object containing action and bet_size
        """
        pass
    
    def reset(self):
        """Reset agent state for a new hand or session"""
        self.action_history = []
    
    def update_after_hand(self, profit: float, hand_result: Dict):
        """
        Update agent after a hand is complete.
        Can be overridden for learning agents.
        
        Args:
            profit: Profit/loss from the hand
            hand_result: Full hand result information
        """
        self.hand_count += 1
        self.total_profit += profit
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            'name': self.name,
            'hand_count': self.hand_count,
            'total_profit': self.total_profit,
            'avg_profit_per_hand': self.total_profit / max(1, self.hand_count)
        }
    
    def _parse_state(self, state: Dict) -> Dict:
        """Helper to parse and validate state dictionary"""
        parsed = {
            'hole_cards': state.get('hole_cards', []),
            'board_cards': state.get('board_cards', []),
            'agent_stack': float(state.get('agent_stack', 1000)),
            'pot': float(state.get('pot', 0)),
            'current_bet': float(state.get('current_bet', 0)),
            'street': state.get('street', 'preflop'),
            'legal_actions': state.get('legal_actions', ['fold', 'call', 'raise']),
            'opponent_stacks': state.get('opponent_stacks', {}),
            'min_raise': float(state.get('min_raise', 10)),
            'action_history': state.get('action_history', [])
        }
        return parsed
    
    def _is_action_legal(self, action: str, legal_actions: list) -> bool:
        """Check if action is in legal actions"""
        return action.lower() in [a.lower() for a in legal_actions]
    
    def _calculate_bet_size(self, pot: float, stack: float, street: str) -> float:
        """
        Default bet size calculation.
        Can be overridden by subclasses.
        """
        # Standard bet sizes based on pot
        if street == 'preflop':
            return pot * 0.75  # 3/4 pot preflop
        elif street == 'flop':
            return pot * 0.66  # 2/3 pot on flop
        elif street == 'turn':
            return pot * 0.66  # 2/3 pot on turn
        else:  # river
            return pot * 0.5   # 1/2 pot on river