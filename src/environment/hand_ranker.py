"""
Hand ranking and evaluation for Texas Hold'em
"""

from typing import List, Tuple, Dict, Optional
from collections import Counter
from enum import IntEnum


class HandRank(IntEnum):
    """Hand ranking values (higher is better)"""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class HandStrength:
    """Represents a hand's strength with ranking and kickers"""
    
    def __init__(self, rank: HandRank, kickers: List[int]):
        self.rank = rank
        self.kickers = kickers
    
    def __lt__(self, other: 'HandStrength') -> bool:
        if self.rank != other.rank:
            return self.rank < other.rank
        for k1, k2 in zip(self.kickers, other.kickers):
            if k1 != k2:
                return k1 < k2
        return False
    
    def __gt__(self, other: 'HandStrength') -> bool:
        if self.rank != other.rank:
            return self.rank > other.rank
        for k1, k2 in zip(self.kickers, other.kickers):
            if k1 != k2:
                return k1 > k2
        return False
    
    def __eq__(self, other: 'HandStrength') -> bool:
        if self.rank != other.rank:
            return False
        return self.kickers == other.kickers
    
    def to_tuple(self) -> Tuple[int, List[int]]:
        return (self.rank.value, self.kickers)


class HandEvaluator:
    """Evaluate poker hand strengths"""
    
    # Card rank values
    RANK_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    
    SUITS = ['h', 'd', 'c', 's']
    
    @classmethod
    def parse_cards(cls, cards: List[str]) -> Tuple[List[int], List[str]]:
        """Parse cards into ranks and suits"""
        ranks = []
        suits = []
        
        for card in cards:
            if len(card) >= 2:
                rank_char = card[0]
                suit = card[1] if len(card) == 2 else card[2]
                
                if rank_char in cls.RANK_VALUES:
                    ranks.append(cls.RANK_VALUES[rank_char])
                    suits.append(suit)
        
        return ranks, suits
    
    @classmethod
    def evaluate_hand(cls, hole_cards: List[str], board_cards: List[str]) -> HandStrength:
        """
        Evaluate the best 5-card hand from hole and board cards
        
        Args:
            hole_cards: Player's 2 hole cards
            board_cards: Community cards (0-5 cards)
            
        Returns:
            HandStrength object with rank and kickers
        """
        all_cards = hole_cards + board_cards
        
        if len(all_cards) < 5:
            # Not enough cards for full evaluation
            return HandStrength(HandRank.HIGH_CARD, [0])
        
        ranks, suits = cls.parse_cards(all_cards)
        
        if len(ranks) < 5:
            return HandStrength(HandRank.HIGH_CARD, [0])
        
        # Get all possible 5-card combinations
        from itertools import combinations
        best_hand = None
        
        for combo in combinations(zip(ranks, suits), 5):
            combo_ranks = [r for r, _ in combo]
            combo_suits = [s for _, s in combo]
            hand = cls._evaluate_5_cards(combo_ranks, combo_suits)
            
            if best_hand is None or hand > best_hand:
                best_hand = hand
        
        return best_hand if best_hand else HandStrength(HandRank.HIGH_CARD, [0])
    
    @classmethod
    def _evaluate_5_cards(cls, ranks: List[int], suits: List[str]) -> HandStrength:
        """Evaluate exactly 5 cards"""
        
        # Sort ranks descending
        ranks_sorted = sorted(ranks, reverse=True)
        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(set(ranks), reverse=True)
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        is_straight = cls._is_straight(unique_ranks)
        
        # Royal Flush (10-A straight flush)
        if is_flush and is_straight and max(ranks) == 14 and min(ranks) == 10:
            return HandStrength(HandRank.ROYAL_FLUSH, [14])
        
        # Straight Flush
        if is_flush and is_straight:
            straight_high = cls._get_straight_high(unique_ranks)
            return HandStrength(HandRank.STRAIGHT_FLUSH, [straight_high])
        
        # Four of a Kind
        if 4 in counts:
            four_rank = cls._get_rank_by_count(rank_counts, 4)
            kicker = cls._get_rank_by_count(rank_counts, 1)
            return HandStrength(HandRank.FOUR_OF_A_KIND, [four_rank, kicker])
        
        # Full House
        if 3 in counts and 2 in counts:
            three_rank = cls._get_rank_by_count(rank_counts, 3)
            two_rank = cls._get_rank_by_count(rank_counts, 2)
            return HandStrength(HandRank.FULL_HOUSE, [three_rank, two_rank])
        
        # Flush
        if is_flush:
            return HandStrength(HandRank.FLUSH, ranks_sorted[:5])
        
        # Straight
        if is_straight:
            straight_high = cls._get_straight_high(unique_ranks)
            return HandStrength(HandRank.STRAIGHT, [straight_high])
        
        # Three of a Kind
        if 3 in counts:
            three_rank = cls._get_rank_by_count(rank_counts, 3)
            kickers = cls._get_kickers(rank_counts, [three_rank], 2)
            return HandStrength(HandRank.THREE_OF_A_KIND, [three_rank] + kickers)
        
        # Two Pair
        if counts.count(2) >= 2:
            pairs = cls._get_ranks_by_count(rank_counts, 2, 2)
            kicker = cls._get_rank_by_count(rank_counts, 1)
            return HandStrength(HandRank.TWO_PAIR, pairs + [kicker])
        
        # One Pair
        if 2 in counts:
            pair_rank = cls._get_rank_by_count(rank_counts, 2)
            kickers = cls._get_kickers(rank_counts, [pair_rank], 3)
            return HandStrength(HandRank.ONE_PAIR, [pair_rank] + kickers)
        
        # High Card
        return HandStrength(HandRank.HIGH_CARD, ranks_sorted[:5])
    
    @classmethod
    def _is_straight(cls, unique_ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        if len(unique_ranks) < 5:
            return False
        
        # Check for Ace-low straight (A,2,3,4,5)
        if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
            return True
        
        # Check normal straight
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                return True
        
        return False
    
    @classmethod
    def _get_straight_high(cls, unique_ranks: List[int]) -> int:
        """Get the high card of a straight"""
        # Check for Ace-low straight
        if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
            return 5
        
        # Find normal straight
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                return unique_ranks[i]
        
        return 0
    
    @classmethod
    def _get_rank_by_count(cls, rank_counts: Counter, count: int) -> int:
        """Get rank that appears exactly 'count' times"""
        for rank, c in rank_counts.items():
            if c == count:
                return rank
        return 0
    
    @classmethod
    def _get_ranks_by_count(cls, rank_counts: Counter, count: int, num: int) -> List[int]:
        """Get 'num' ranks that appear exactly 'count' times"""
        ranks = [rank for rank, c in rank_counts.items() if c == count]
        return sorted(ranks, reverse=True)[:num]
    
    @classmethod
    def _get_kickers(cls, rank_counts: Counter, exclude_ranks: List[int], num: int) -> List[int]:
        """Get kicker ranks excluding certain ranks"""
        kickers = [rank for rank in rank_counts.keys() if rank not in exclude_ranks]
        return sorted(kickers, reverse=True)[:num]
    
    @classmethod
    def compare_hands(cls, hand1: HandStrength, hand2: HandStrength) -> int:
        """
        Compare two hands
        Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        if hand1 > hand2:
            return 1
        elif hand2 > hand1:
            return -1
        return 0
    
    @classmethod
    def get_hand_description(cls, hand_strength: HandStrength) -> str:
        """Get human-readable hand description"""
        rank_names = {
            HandRank.HIGH_CARD: "High Card",
            HandRank.ONE_PAIR: "One Pair",
            HandRank.TWO_PAIR: "Two Pair",
            HandRank.THREE_OF_A_KIND: "Three of a Kind",
            HandRank.STRAIGHT: "Straight",
            HandRank.FLUSH: "Flush",
            HandRank.FULL_HOUSE: "Full House",
            HandRank.FOUR_OF_A_KIND: "Four of a Kind",
            HandRank.STRAIGHT_FLUSH: "Straight Flush",
            HandRank.ROYAL_FLUSH: "Royal Flush"
        }
        
        return rank_names.get(hand_strength.rank, "Unknown")


class HandRanker:
    """Wrapper class for hand evaluation"""
    
    @staticmethod
    def evaluate(hole_cards: List[str], board_cards: List[str]) -> Tuple[int, List[int]]:
        """Evaluate hand and return (rank, kickers) tuple"""
        strength = HandEvaluator.evaluate_hand(hole_cards, board_cards)
        return strength.to_tuple()
    
    @staticmethod
    def compare(hand1: Tuple[int, List[int]], hand2: Tuple[int, List[int]]) -> int:
        """Compare two hands"""
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
    
    @staticmethod
    def get_win_probability(hole_cards: List[str], board_cards: List[str], 
                           num_opponents: int = 1, num_simulations: int = 1000) -> float:
        """
        Estimate win probability using Monte Carlo simulation
        """
        import random
        
        wins = 0
        ties = 0
        
        for _ in range(num_simulations):
            # Simulate opponent hands and remaining board cards
            # Simplified implementation
            pass
        
        return wins / num_simulations