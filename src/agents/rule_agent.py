"""
Rule-based and random baseline agents
"""

import random
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentAction


class RuleBasedAgent(BaseAgent):
    """
    Rule-based poker agent with configurable strategies.
    Implements basic poker heuristics.
    """
    
    def __init__(
        self,
        name: str = "RuleAgent",
        strategy: str = "conservative",
        aggression: float = 0.5
    ):
        """
        Initialize rule-based agent.
        
        Args:
            name: Agent name
            strategy: 'conservative', 'aggressive', 'tight', 'loose', 'always_fold', 'always_call'
            aggression: Value between 0 and 1 (0=passive, 1=very aggressive)
        """
        super().__init__(name)
        self.strategy = strategy
        self.aggression = aggression
    
    def _evaluate_hand_strength(self, hole_cards: List[str], board_cards: List[str]) -> float:
        """
        Evaluate approximate hand strength (0-1)
        Simplified evaluation for rule-based agent
        """
        if len(hole_cards) < 2:
            return 0.0
        
        ranks = [c[0] for c in hole_cards[:2]]
        suits = [c[1] if len(c) == 2 else c[2] for c in hole_cards[:2]]
        
        # Rank values
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                      '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        
        rank1 = rank_values.get(ranks[0], 0)
        rank2 = rank_values.get(ranks[1], 0)
        
        is_pair = ranks[0] == ranks[1]
        is_suited = suits[0] == suits[1] if len(suits) >= 2 else False
        
        # Base strength
        if is_pair:
            strength = 0.7 + (rank1 - 2) / 12 * 0.3
        elif is_suited and max(rank1, rank2) >= 12:
            strength = 0.6
        elif max(rank1, rank2) >= 13:
            strength = 0.5
        elif is_suited:
            strength = 0.4
        else:
            strength = 0.2 + (max(rank1, rank2) - 2) / 12 * 0.3
        
        # Adjust for number of board cards (made hands are stronger)
        if len(board_cards) >= 5:
            # River - made hand
            strength = min(1.0, strength + 0.2)
        elif len(board_cards) >= 4:
            # Turn - drawing
            strength = min(1.0, strength + 0.1)
        elif len(board_cards) >= 3:
            # Flop
            strength = strength
        
        return min(1.0, strength)
    
    def _conservative_strategy(self, strength: float, legal_actions: List[str], 
                                pot: float, current_bet: float, stack: float) -> AgentAction:
        """Conservative/tight strategy"""
        if strength > 0.8:
            # Strong hand - raise
            if 'raise' in legal_actions:
                bet_size = min(pot * 0.5, stack)
                return AgentAction('raise', bet_size, confidence=strength)
            elif 'bet' in legal_actions:
                bet_size = min(pot * 0.5, stack)
                return AgentAction('bet', bet_size, confidence=strength)
        
        elif strength > 0.6:
            # Decent hand - call if not too expensive
            if 'call' in legal_actions and current_bet < pot * 0.3:
                return AgentAction('call', current_bet, confidence=strength)
            elif 'check' in legal_actions:
                return AgentAction('check', 0, confidence=strength)
        
        elif strength > 0.4:
            # Marginal hand - call only if cheap
            if 'call' in legal_actions and current_bet < pot * 0.2:
                return AgentAction('call', current_bet, confidence=strength)
        
        # Default: fold or check
        if 'check' in legal_actions:
            return AgentAction('check', 0, confidence=0.3)
        return AgentAction('fold', 0, confidence=0.3)
    
    def _aggressive_strategy(self, strength: float, legal_actions: List[str],
                              pot: float, current_bet: float, stack: float) -> AgentAction:
        """Aggressive/loose strategy"""
        if strength > 0.5:
            # Raise with moderate+ hands
            if 'raise' in legal_actions:
                bet_size = min(pot * 0.8, stack)
                return AgentAction('raise', bet_size, confidence=strength)
            elif 'bet' in legal_actions:
                bet_size = min(pot * 0.7, stack)
                return AgentAction('bet', bet_size, confidence=strength)
        
        elif strength > 0.3:
            # Call or bet
            if 'bet' in legal_actions:
                bet_size = min(pot * 0.4, stack)
                return AgentAction('bet', bet_size, confidence=strength)
            elif 'call' in legal_actions:
                return AgentAction('call', current_bet, confidence=strength)
        
        # Bluff occasionally
        if random.random() < self.aggression * 0.2:
            if 'raise' in legal_actions:
                return AgentAction('raise', min(pot * 0.5, stack), confidence=0.2)
        
        if 'check' in legal_actions:
            return AgentAction('check', 0, confidence=0.3)
        return AgentAction('fold', 0, confidence=0.3)
    
    def _tight_strategy(self, strength: float, legal_actions: List[str],
                         pot: float, current_bet: float, stack: float) -> AgentAction:
        """Tight strategy - only play premium hands"""
        if strength > 0.75:
            if 'raise' in legal_actions:
                return AgentAction('raise', min(pot * 0.6, stack), confidence=strength)
            elif 'bet' in legal_actions:
                return AgentAction('bet', min(pot * 0.5, stack), confidence=strength)
            elif 'call' in legal_actions:
                return AgentAction('call', current_bet, confidence=strength)
        
        elif strength > 0.6 and current_bet < pot * 0.25:
            if 'call' in legal_actions:
                return AgentAction('call', current_bet, confidence=strength)
        
        # Fold unless check is free
        if 'check' in legal_actions:
            return AgentAction('check', 0, confidence=0.3)
        return AgentAction('fold', 0, confidence=0.3)
    
    def _loose_strategy(self, strength: float, legal_actions: List[str],
                         pot: float, current_bet: float, stack: float) -> AgentAction:
        """Loose strategy - play many hands"""
        if strength > 0.4:
            if 'raise' in legal_actions:
                return AgentAction('raise', min(pot * 0.7, stack), confidence=strength)
            elif 'bet' in legal_actions:
                return AgentAction('bet', min(pot * 0.6, stack), confidence=strength)
        
        if 'call' in legal_actions:
            return AgentAction('call', current_bet, confidence=strength)
        
        if 'check' in legal_actions:
            return AgentAction('check', 0, confidence=strength)
        
        return AgentAction('fold', 0, confidence=0.3)
    
    def _always_fold(self, legal_actions: List[str]) -> AgentAction:
        """Always fold strategy"""
        if 'fold' in legal_actions:
            return AgentAction('fold', 0, confidence=1.0)
        return AgentAction('check' if 'check' in legal_actions else 'fold', 0)
    
    def _always_call(self, legal_actions: List[str], current_bet: float, stack: float) -> AgentAction:
        """Always call strategy"""
        if 'call' in legal_actions:
            return AgentAction('call', min(current_bet, stack), confidence=1.0)
        elif 'check' in legal_actions:
            return AgentAction('check', 0, confidence=1.0)
        return AgentAction('fold', 0, confidence=0.5)
    
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """Make decision based on strategy"""
        parsed = self._parse_state(state)
        
        # Evaluate hand strength
        strength = self._evaluate_hand_strength(
            parsed['hole_cards'],
            parsed['board_cards']
        )
        
        # Apply strategy
        if self.strategy == 'always_fold':
            return self._always_fold(parsed['legal_actions'])
        
        elif self.strategy == 'always_call':
            return self._always_call(parsed['legal_actions'], parsed['current_bet'], parsed['agent_stack'])
        
        elif self.strategy == 'conservative':
            return self._conservative_strategy(strength, parsed['legal_actions'], 
                                               parsed['pot'], parsed['current_bet'], parsed['agent_stack'])
        
        elif self.strategy == 'aggressive':
            return self._aggressive_strategy(strength, parsed['legal_actions'],
                                            parsed['pot'], parsed['current_bet'], parsed['agent_stack'])
        
        elif self.strategy == 'tight':
            return self._tight_strategy(strength, parsed['legal_actions'],
                                       parsed['pot'], parsed['current_bet'], parsed['agent_stack'])
        
        elif self.strategy == 'loose':
            return self._loose_strategy(strength, parsed['legal_actions'],
                                       parsed['pot'], parsed['current_bet'], parsed['agent_stack'])
        
        else:
            # Default conservative
            return self._conservative_strategy(strength, parsed['legal_actions'],
                                               parsed['pot'], parsed['current_bet'], parsed['agent_stack'])
    
    def reset(self):
        """Reset agent"""
        super().reset()


class RandomAgent(BaseAgent):
    """Random agent - chooses random legal actions"""
    
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
    
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """Choose random legal action"""
        parsed = self._parse_state(state)
        legal_actions = parsed['legal_actions']
        
        if not legal_actions:
            return AgentAction('fold', 0, confidence=0.0)
        
        action = random.choice(legal_actions)
        
        # Calculate bet size for betting actions
        bet_size = 0.0
        if action in ['bet', 'raise']:
            pot = parsed['pot']
            bet_size = random.uniform(pot * 0.25, min(pot * 1.0, parsed['agent_stack']))
        elif action == 'call':
            bet_size = parsed['current_bet']
        elif action == 'all_in':
            bet_size = parsed['agent_stack']
        
        return AgentAction(action, bet_size, confidence=0.5)
    
    def reset(self):
        pass