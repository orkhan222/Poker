"""
Poker-specific utility functions
"""

from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
import itertools
import random

from .constants import RANK_VALUES, SUIT_NAMES


class CardUtils:
    """Utility class for card operations"""
    
    @staticmethod
    def is_valid_card(card: str) -> bool:
        """Check if a card string is valid"""
        if len(card) < 2:
            return False
        
        rank = card[0]
        suit = card[1] if len(card) == 2 else card[2]
        
        return rank in RANK_VALUES and suit in SUIT_NAMES
    
    @staticmethod
    def normalize_card(card: str) -> str:
        """Normalize card string to standard format"""
        if not card:
            return ""
        
        rank = card[0].upper()
        suit = card[1].lower() if len(card) >= 2 else ""
        
        if rank in RANK_VALUES and suit in SUIT_NAMES:
            return f"{rank}{suit}"
        return ""
    
    @staticmethod
    def get_rank(card: str) -> int:
        """Get numeric rank of a card"""
        if not card or len(card) < 1:
            return 0
        rank = card[0]
        return RANK_VALUES.get(rank.upper(), 0)
    
    @staticmethod
    def get_suit(card: str) -> str:
        """Get suit of a card"""
        if not card or len(card) < 2:
            return ""
        suit = card[1] if len(card) == 2 else card[2]
        return suit.lower()
    
    @staticmethod
    def card_to_string(rank: int, suit: str) -> str:
        """Convert rank value and suit to card string"""
        rank_map = {v: k for k, v in RANK_VALUES.items()}
        rank_str = rank_map.get(rank, str(rank))
        return f"{rank_str}{suit}"
    
    @staticmethod
    def create_deck() -> List[str]:
        """Create a standard 52-card deck"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        return [f"{r}{s}" for r in ranks for s in suits]
    
    @staticmethod
    def shuffle_deck(deck: List[str], seed: Optional[int] = None) -> List[str]:
        """Shuffle the deck"""
        if seed is not None:
            random.seed(seed)
        shuffled = deck.copy()
        random.shuffle(shuffled)
        return shuffled
    
    @staticmethod
    def deal_cards(deck: List[str], num_cards: int) -> Tuple[List[str], List[str]]:
        """Deal cards from deck"""
        if len(deck) < num_cards:
            raise ValueError("Not enough cards in deck")
        return deck[:num_cards], deck[num_cards:]


class HandEvaluatorUtils:
    """Utility class for hand evaluation"""
    
    @staticmethod
    def is_pair(ranks: List[int]) -> bool:
        """Check if ranks contain a pair"""
        rank_counts = Counter(ranks)
        return 2 in rank_counts.values()
    
    @staticmethod
    def is_two_pair(ranks: List[int]) -> bool:
        """Check if ranks contain two pairs"""
        rank_counts = Counter(ranks)
        return list(rank_counts.values()).count(2) >= 2
    
    @staticmethod
    def is_three_of_a_kind(ranks: List[int]) -> bool:
        """Check if ranks contain three of a kind"""
        rank_counts = Counter(ranks)
        return 3 in rank_counts.values()
    
    @staticmethod
    def is_straight(ranks: List[int]) -> Tuple[bool, int]:
        """
        Check if ranks form a straight
        Returns: (is_straight, high_card)
        """
        unique_ranks = sorted(set(ranks), reverse=True)
        
        if len(unique_ranks) < 5:
            return False, 0
        
        # Check for Ace-low straight
        if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
            return True, 5
        
        # Check normal straight
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                return True, unique_ranks[i]
        
        return False, 0
    
    @staticmethod
    def is_flush(suits: List[str]) -> bool:
        """Check if suits form a flush"""
        suit_counts = Counter(suits)
        return max(suit_counts.values()) >= 5
    
    @staticmethod
    def is_full_house(ranks: List[int]) -> bool:
        """Check if ranks form a full house"""
        rank_counts = Counter(ranks)
        values = sorted(rank_counts.values(), reverse=True)
        return values[0] >= 3 and values[1] >= 2
    
    @staticmethod
    def is_four_of_a_kind(ranks: List[int]) -> bool:
        """Check if ranks contain four of a kind"""
        rank_counts = Counter(ranks)
        return 4 in rank_counts.values()


def normalize_card(card: str) -> str:
    """Convenience function to normalize a card"""
    return CardUtils.normalize_card(card)


def is_valid_card(card: str) -> bool:
    """Convenience function to check if card is valid"""
    return CardUtils.is_valid_card(card)


def get_card_rank_value(card: str) -> int:
    """Convenience function to get card rank value"""
    return CardUtils.get_rank(card)


def get_card_suit(card: str) -> str:
    """Convenience function to get card suit"""
    return CardUtils.get_suit(card)


def compare_hands_utils(hand1: Tuple[int, List[int]], hand2: Tuple[int, List[int]]) -> int:
    """
    Compare two hands
    Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    rank1, kickers1 = hand1
    rank2, kickers2 = hand2
    
    if rank1 > rank2:
        return 1
    if rank1 < rank2:
        return -1
    
    for k1, k2 in zip(kickers1, kickers2):
        if k1 > k2:
            return 1
        if k1 < k2:
            return -1
    
    return 0


def calculate_pot_odds(pot: float, call_amount: float) -> float:
    """
    Calculate pot odds
    
    Args:
        pot: Current pot size
        call_amount: Amount to call
        
    Returns:
        Pot odds ratio (0-1)
    """
    if call_amount <= 0:
        return 1.0
    
    total_pot = pot + call_amount
    return call_amount / total_pot


def calculate_equity(hand_strength: float, num_opponents: int = 1) -> float:
    """
    Estimate equity based on hand strength
    
    Args:
        hand_strength: Estimated hand strength (0-1)
        num_opponents: Number of opponents
        
    Returns:
        Estimated equity (0-1)
    """
    # Simplified equity calculation
    # In practice, you'd use Monte Carlo simulation
    base_equity = hand_strength
    opponent_factor = 1 / (num_opponents + 1)
    
    equity = base_equity * (1 - opponent_factor) + opponent_factor * 0.5
    return min(1.0, max(0.0, equity))


def get_hand_description(hole_cards: List[str], board_cards: List[str]) -> str:
    """
    Get description of current hand
    
    Args:
        hole_cards: Player's hole cards
        board_cards: Community cards
        
    Returns:
        Hand description string
    """
    if len(hole_cards) < 2:
        return "No cards"
    
    ranks = [c[0] for c in hole_cards[:2]]
    suits = [c[1] if len(c) == 2 else c[2] for c in hole_cards[:2]]
    
    rank_names = {'A': 'Ace', 'K': 'King', 'Q': 'Queen', 'J': 'Jack',
                  'T': 'Ten', '9': 'Nine', '8': 'Eight', '7': 'Seven',
                  '6': 'Six', '5': 'Five', '4': 'Four', '3': 'Three', '2': 'Two'}
    
    rank1 = rank_names.get(ranks[0], ranks[0])
    rank2 = rank_names.get(ranks[1], ranks[1])
    
    if ranks[0] == ranks[1]:
        return f"Pocket {rank1}s"
    elif suits[0] == suits[1]:
        return f"{rank1}-{rank2} suited"
    else:
        return f"{rank1}-{rank2} offsuit"