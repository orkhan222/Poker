"""
Game state management - Track all poker game state
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from copy import deepcopy


class Street(Enum):
    """Betting rounds in Texas Hold'em"""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


class Action(Enum):
    """Possible player actions"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5


@dataclass
class PlayerState:
    """State of a single player in the game"""
    position: int
    stack: float
    current_bet: float = 0.0
    total_bet: float = 0.0
    hole_cards: List[str] = field(default_factory=list)
    is_active: bool = True
    is_all_in: bool = False
    has_acted: bool = False
    nickname: str = ""
    
    def reset_for_new_hand(self, new_stack: float):
        """Reset player state for new hand"""
        self.stack = new_stack
        self.current_bet = 0.0
        self.total_bet = 0.0
        self.hole_cards = []
        self.is_active = True
        self.is_all_in = False
        self.has_acted = False
    
    def can_act(self) -> bool:
        """Check if player can act"""
        return self.is_active and not self.is_all_in
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'position': self.position,
            'stack': self.stack,
            'current_bet': self.current_bet,
            'total_bet': self.total_bet,
            'hole_cards': self.hole_cards,
            'is_active': self.is_active,
            'is_all_in': self.is_all_in,
            'has_acted': self.has_acted,
            'nickname': self.nickname
        }


@dataclass
class GameState:
    """Complete poker game state"""
    
    # Game info
    hand_id: str = ""
    hand_number: int = 0
    
    # Players
    players: List[PlayerState] = field(default_factory=list)
    current_player_idx: int = 0
    dealer_idx: int = 0
    
    # Betting
    pot: float = 0.0
    current_bet: float = 0.0
    min_raise: float = 0.0
    last_raise: float = 0.0
    
    # Cards
    board_cards: List[str] = field(default_factory=list)
    
    # Round info
    street: Street = Street.PREFLOP
    betting_round: int = 0
    actions_this_round: int = 0
    
    # Timing (for agent)
    opponent_waiting_time: Optional[float] = None
    time_since_agent_bet: Optional[float] = None
    
    # Terminal
    is_terminal: bool = False
    winners: List[int] = field(default_factory=list)
    winner_amounts: List[float] = field(default_factory=list)
    
    # History
    action_history: List[Dict] = field(default_factory=list)
    
    @property
    def num_players(self) -> int:
        return len(self.players)
    
    @property
    def num_active_players(self) -> int:
        return sum(1 for p in self.players if p.is_active)
    
    @property
    def current_player(self) -> Optional[PlayerState]:
        if 0 <= self.current_player_idx < len(self.players):
            return self.players[self.current_player_idx]
        return None
    
    @property
    def is_heads_up(self) -> bool:
        return self.num_active_players == 2
    
    @property
    def is_preflop(self) -> bool:
        return self.street == Street.PREFLOP
    
    @property
    def is_flop(self) -> bool:
        return self.street == Street.FLOP
    
    @property
    def is_turn(self) -> bool:
        return self.street == Street.TURN
    
    @property
    def is_river(self) -> bool:
        return self.street == Street.RIVER
    
    def get_active_players(self) -> List[PlayerState]:
        """Get list of active players"""
        return [p for p in self.players if p.is_active]
    
    def get_player_by_position(self, position: int) -> Optional[PlayerState]:
        """Get player by position index"""
        for p in self.players:
            if p.position == position:
                return p
        return None
    
    def get_legal_actions(self, player_idx: int = None) -> List[Action]:
        """Get legal actions for a player"""
        if player_idx is None:
            player_idx = self.current_player_idx
        
        player = self.players[player_idx]
        
        if not player.can_act():
            return []
        
        legal_actions = []
        
        # Fold is always legal if there's a bet to call
        if self.current_bet > player.current_bet:
            legal_actions.append(Action.FOLD)
        
        # Check if no bet to call
        if self.current_bet == player.current_bet:
            legal_actions.append(Action.CHECK)
        
        # Call if there's a bet and player has enough chips
        call_amount = self.current_bet - player.current_bet
        if call_amount > 0 and player.stack >= call_amount:
            legal_actions.append(Action.CALL)
        
        # Bet if no current bet and player has chips
        if self.current_bet == 0 and player.stack >= self.min_raise:
            legal_actions.append(Action.BET)
        
        # Raise if player has enough chips
        raise_amount = self.current_bet + self.min_raise - player.current_bet
        if raise_amount > 0 and player.stack >= raise_amount:
            legal_actions.append(Action.RAISE)
        
        # All-in if player has chips but not enough for min raise/call
        if player.stack > 0:
            if (self.current_bet > player.current_bet and player.stack < call_amount) or \
               (self.current_bet == 0 and player.stack < self.min_raise):
                legal_actions.append(Action.ALL_IN)
        
        return legal_actions
    
    def get_bet_amount(self, action: Action) -> float:
        """Get the amount to bet for a given action"""
        player = self.current_player
        
        if action == Action.CALL:
            return self.current_bet - player.current_bet
        
        elif action == Action.BET:
            return self.min_raise
        
        elif action == Action.RAISE:
            return self.current_bet + self.min_raise - player.current_bet
        
        elif action == Action.ALL_IN:
            return player.stack
        
        return 0.0
    
    def apply_action(self, action: Action, bet_amount: float = None) -> float:
        """
        Apply an action and return the amount bet
        
        Args:
            action: Action to apply
            bet_amount: Optional custom bet amount
            
        Returns:
            Amount of chips added to pot
        """
        player = self.current_player
        if bet_amount is None:
            bet_amount = self.get_bet_amount(action)
        
        # Record action
        self.action_history.append({
            'hand_id': self.hand_id,
            'street': self.street.name,
            'player_idx': self.current_player_idx,
            'player_position': player.position,
            'action': action.name,
            'amount': bet_amount,
            'stack_before': player.stack,
            'pot_before': self.pot,
            'timestamp': len(self.action_history)
        })
        
        amount_paid = 0.0
        
        if action == Action.FOLD:
            player.is_active = False
            self.pot += player.current_bet
            amount_paid = player.current_bet
        
        elif action == Action.CHECK:
            pass
        
        elif action == Action.CALL:
            call_amount = self.current_bet - player.current_bet
            player.stack -= call_amount
            player.current_bet = self.current_bet
            player.total_bet += call_amount
            self.pot += call_amount
            amount_paid = call_amount
        
        elif action == Action.BET:
            self.current_bet = bet_amount
            self.last_raise = bet_amount
            self.min_raise = bet_amount
            player.stack -= bet_amount
            player.current_bet = bet_amount
            player.total_bet += bet_amount
            self.pot += bet_amount
            amount_paid = bet_amount
        
        elif action == Action.RAISE:
            total_bet = self.current_bet + bet_amount - player.current_bet
            raise_size = bet_amount - self.current_bet
            player.stack -= total_bet
            player.current_bet = bet_amount
            player.total_bet += total_bet
            self.pot += total_bet
            self.current_bet = bet_amount
            self.last_raise = raise_size
            self.min_raise = raise_size
            amount_paid = total_bet
        
        elif action == Action.ALL_IN:
            all_in_amount = player.stack
            player.stack = 0
            player.current_bet += all_in_amount
            player.total_bet += all_in_amount
            player.is_all_in = True
            self.pot += all_in_amount
            
            # Update current bet if all-in is larger
            if player.current_bet > self.current_bet:
                self.current_bet = player.current_bet
            
            amount_paid = all_in_amount
        
        player.has_acted = True
        
        return amount_paid
    
    def advance_to_next_player(self) -> int:
        """Move to next active player, return next player index"""
        if self.num_active_players <= 1:
            self.is_terminal = True
            return -1
        
        next_idx = (self.current_player_idx + 1) % self.num_players
        
        # Skip inactive players
        while next_idx != self.current_player_idx:
            if self.players[next_idx].is_active and not self.players[next_idx].is_all_in:
                self.current_player_idx = next_idx
                return next_idx
            next_idx = (next_idx + 1) % self.num_players
        
        # No more players to act - move to next street
        return -1
    
    def is_betting_round_complete(self) -> bool:
        """Check if current betting round is complete"""
        if self.num_active_players <= 1:
            return True
        
        # All active players have acted
        for player in self.players:
            if player.is_active and not player.is_all_in and not player.has_acted:
                return False
        
        # All bets are equal
        for player in self.players:
            if player.is_active and not player.is_all_in:
                if player.current_bet != self.current_bet:
                    return False
        
        return True
    
    def next_street(self):
        """Move to next betting street"""
        if self.street == Street.PREFLOP:
            self.street = Street.FLOP
        elif self.street == Street.FLOP:
            self.street = Street.TURN
        elif self.street == Street.TURN:
            self.street = Street.RIVER
        else:
            self.street = Street.SHOWDOWN
            self.is_terminal = True
        
        # Reset for new betting round
        self.current_bet = 0
        self.min_raise = 0
        self.last_raise = 0
        self.actions_this_round = 0
        
        for player in self.players:
            player.current_bet = 0
            player.has_acted = False
        
        # Find first active player after dealer
        start_idx = (self.dealer_idx + 1) % self.num_players
        while not self.players[start_idx].is_active or self.players[start_idx].is_all_in:
            start_idx = (start_idx + 1) % self.num_players
            if start_idx == (self.dealer_idx + 1) % self.num_players:
                break
        
        self.current_player_idx = start_idx
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for agent input"""
        return {
            'hand_id': self.hand_id,
            'hand_number': self.hand_number,
            'street': self.street.name,
            'pot': self.pot,
            'current_bet': self.current_bet,
            'min_raise': self.min_raise,
            'board_cards': self.board_cards.copy(),
            'current_player_idx': self.current_player_idx,
            'players': [p.to_dict() for p in self.players],
            'legal_actions': [a.name for a in self.get_legal_actions()],
            'opponent_waiting_time': self.opponent_waiting_time,
            'time_since_agent_bet': self.time_since_agent_bet,
            'is_terminal': self.is_terminal,
            'num_active_players': self.num_active_players
        }
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state"""
        return deepcopy(self)