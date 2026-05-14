"""
Poker rules and betting structures
"""

from typing import Tuple, List, Dict
from .game_state import Street, Action


class BlindStructure:
    """Blind structure for tournament play"""
    
    def __init__(self, initial_sb: float = 5, initial_bb: float = 10):
        self.initial_sb = initial_sb
        self.initial_bb = initial_bb
        self.level = 0
        self.sb = initial_sb
        self.bb = initial_bb
    
    def next_level(self):
        """Move to next blind level"""
        self.level += 1
        self.sb = self.initial_sb * (2 ** (self.level // 10))  # Double every 10 levels
        self.bb = self.initial_bb * (2 ** (self.level // 10))
        return self.sb, self.bb
    
    def reset(self):
        """Reset to initial blinds"""
        self.level = 0
        self.sb = self.initial_sb
        self.bb = self.initial_bb
    
    def get_blinds(self) -> Tuple[float, float]:
        return self.sb, self.bb


class BettingRules:
    """Betting rules and limits"""
    
    def __init__(self, min_bet: float = 10, max_bet: float = None):
        self.min_bet = min_bet
        self.max_bet = max_bet
    
    def get_min_raise(self, current_bet: float, last_raise: float) -> float:
        """
        Calculate minimum raise amount
        
        In No-Limit Hold'em, min raise is at least the last raise amount
        """
        if last_raise == 0:
            return current_bet + max(self.min_bet, current_bet)
        return current_bet + last_raise
    
    def get_max_bet(self, player_stack: float, pot: float) -> float:
        """Maximum bet is player's stack (all-in)"""
        return player_stack
    
    def get_min_bet(self, pot: float) -> float:
        """Minimum bet is the current min_bet or the big blind"""
        return max(self.min_bet, pot / 2 if pot > 0 else self.min_bet)
    
    def is_valid_bet_size(self, bet_amount: float, current_bet: float, 
                          player_stack: float, min_raise: float) -> Tuple[bool, str]:
        """Check if bet amount is valid"""
        
        if bet_amount <= 0:
            return False, "Bet amount must be positive"
        
        if bet_amount > player_stack:
            return False, "Bet amount exceeds player stack (would be all-in)"
        
        # Bet must be at least min raise
        if bet_amount < min_raise:
            return False, f"Bet amount below minimum raise ({min_raise})"
        
        # Bet must be a raise (greater than current bet)
        if bet_amount <= current_bet:
            return False, f"Bet must be greater than current bet ({current_bet})"
        
        # Check max bet
        if self.max_bet and bet_amount > self.max_bet:
            return False, f"Bet exceeds maximum bet ({self.max_bet})"
        
        return True, "Valid"


class PokerRules:
    """Main poker rules class"""
    
    @staticmethod
    def get_button_position(num_players: int, hand_number: int) -> int:
        """Get button position for a given hand number"""
        return hand_number % num_players
    
    @staticmethod
    def get_blind_positions(button_pos: int, num_players: int) -> Tuple[int, int]:
        """Get small blind and big blind positions"""
        sb_pos = (button_pos + 1) % num_players
        bb_pos = (button_pos + 2) % num_players
        return sb_pos, bb_pos
    
    @staticmethod
    def post_blinds(state, blind_structure: BlindStructure) -> float:
        """Post blinds and return total pot"""
        sb, bb = blind_structure.get_blinds()
        sb_pos, bb_pos = PokerRules.get_blind_positions(
            state.dealer_idx, state.num_players
        )
        
        total_posted = 0.0
        
        # Small blind
        sb_player = state.players[sb_pos]
        if sb_player.stack >= sb:
            sb_player.stack -= sb
            sb_player.current_bet = sb
            sb_player.total_bet = sb
            total_posted += sb
        else:
            # Short stack - all in
            total_posted += sb_player.stack
            sb_player.current_bet = sb_player.stack
            sb_player.total_bet = sb_player.stack
            sb_player.stack = 0
            sb_player.is_all_in = True
        
        # Big blind
        bb_player = state.players[bb_pos]
        if bb_player.stack >= bb:
            bb_player.stack -= bb
            bb_player.current_bet = bb
            bb_player.total_bet = bb
            total_posted += bb
        else:
            total_posted += bb_player.stack
            bb_player.current_bet = bb_player.stack
            bb_player.total_bet = bb_player.stack
            bb_player.stack = 0
            bb_player.is_all_in = True
        
        state.pot += total_posted
        state.current_bet = bb
        state.min_raise = bb
        
        return total_posted
    
    @staticmethod
    def deal_cards(deck: List[str], num_cards: int = 2) -> List[str]:
        """Deal cards from deck"""
        if len(deck) < num_cards:
            raise ValueError("Not enough cards in deck")
        
        dealt = deck[:num_cards]
        remaining = deck[num_cards:]
        return dealt, remaining
    
    @staticmethod
    def deal_flop(deck: List[str]) -> Tuple[List[str], List[str]]:
        """Deal flop (3 cards)"""
        if len(deck) < 3:
            raise ValueError("Not enough cards for flop")
        return deck[:3], deck[3:]
    
    @staticmethod
    def deal_turn(deck: List[str]) -> Tuple[str, List[str]]:
        """Deal turn (1 card)"""
        if len(deck) < 1:
            raise ValueError("Not enough cards for turn")
        return deck[0], deck[1:]
    
    @staticmethod
    def deal_river(deck: List[str]) -> Tuple[str, List[str]]:
        """Deal river (1 card)"""
        if len(deck) < 1:
            raise ValueError("Not enough cards for river")
        return deck[0], deck[1:]
    
    @staticmethod
    def create_deck() -> List[str]:
        """Create a standard 52-card deck"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        return [f"{r}{s}" for r in ranks for s in suits]
    
    @staticmethod
    def shuffle_deck(deck: List[str], seed: int = None) -> List[str]:
        """Shuffle the deck"""
        import random
        if seed is not None:
            random.seed(seed)
        shuffled = deck.copy()
        random.shuffle(shuffled)
        return shuffled
    
    @staticmethod
    def get_hand_ranking_name(rank: int) -> str:
        """Get hand ranking name from rank value"""
        rankings = {
            9: "Royal Flush",
            8: "Straight Flush",
            7: "Four of a Kind",
            6: "Full House",
            5: "Flush",
            4: "Straight",
            3: "Three of a Kind",
            2: "Two Pair",
            1: "One Pair",
            0: "High Card"
        }
        return rankings.get(rank, "Unknown")
    
    @staticmethod
    def get_action_from_string(action_str: str) -> Action:
        """Convert string to Action enum"""
        action_map = {
            'fold': Action.FOLD,
            'check': Action.CHECK,
            'call': Action.CALL,
            'bet': Action.BET,
            'raise': Action.RAISE,
            'all_in': Action.ALL_IN,
            'all-in': Action.ALL_IN
        }
        return action_map.get(action_str.lower(), Action.FOLD)