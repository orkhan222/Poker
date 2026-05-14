"""
Tests for poker rules
"""

import unittest
from src.environment.rules import PokerRules, BlindStructure, BettingRules
from src.environment.game_state import Street, Action


class TestBlindStructure(unittest.TestCase):
    """Test cases for blind structure"""
    
    def test_initial_blinds(self):
        blinds = BlindStructure(initial_sb=5, initial_bb=10)
        sb, bb = blinds.get_blinds()
        self.assertEqual(sb, 5)
        self.assertEqual(bb, 10)
    
    def test_next_level(self):
        blinds = BlindStructure(initial_sb=5, initial_bb=10)
        
        sb, bb = blinds.next_level()
        self.assertEqual(sb, 10)
        self.assertEqual(bb, 20)
        
        sb, bb = blinds.next_level()
        self.assertEqual(sb, 20)
        self.assertEqual(bb, 40)
    
    def test_reset(self):
        blinds = BlindStructure(initial_sb=5, initial_bb=10)
        blinds.next_level()
        blinds.reset()
        
        sb, bb = blinds.get_blinds()
        self.assertEqual(sb, 5)
        self.assertEqual(bb, 10)


class TestBettingRules(unittest.TestCase):
    """Test cases for betting rules"""
    
    def setUp(self):
        self.rules = BettingRules(min_bet=10)
    
    def test_get_min_raise(self):
        # First raise
        min_raise = self.rules.get_min_raise(50, 0)
        self.assertEqual(min_raise, 60)  # 50 + 10
        
        # After a raise
        min_raise = self.rules.get_min_raise(100, 50)
        self.assertEqual(min_raise, 150)  # 100 + 50
    
    def test_get_max_bet(self):
        max_bet = self.rules.get_max_bet(500, 100)
        self.assertEqual(max_bet, 500)
    
    def test_is_valid_bet_size(self):
        # Valid bet
        valid, msg = self.rules.is_valid_bet_size(100, 50, 1000, 75)
        self.assertTrue(valid)
        
        # Too small
        valid, msg = self.rules.is_valid_bet_size(60, 50, 1000, 75)
        self.assertFalse(valid)
        
        # Too large
        valid, msg = self.rules.is_valid_bet_size(2000, 50, 1000, 75)
        self.assertFalse(valid)


class TestPokerRules(unittest.TestCase):
    """Test cases for poker rules"""
    
    def test_get_button_position(self):
        self.assertEqual(PokerRules.get_button_position(6, 0), 0)
        self.assertEqual(PokerRules.get_button_position(6, 1), 1)
        self.assertEqual(PokerRules.get_button_position(6, 6), 0)
    
    def test_get_blind_positions(self):
        # 6 players, button at position 0
        sb, bb = PokerRules.get_blind_positions(0, 6)
        self.assertEqual(sb, 1)
        self.assertEqual(bb, 2)
        
        # Button at position 4
        sb, bb = PokerRules.get_blind_positions(4, 6)
        self.assertEqual(sb, 5)
        self.assertEqual(bb, 0)
    
    def test_create_deck(self):
        deck = PokerRules.create_deck()
        self.assertEqual(len(deck), 52)
        
        # Check all suits and ranks are present
        ranks = set(c[0] for c in deck)
        suits = set(c[1] for c in deck)
        
        self.assertEqual(len(ranks), 13)
        self.assertEqual(len(suits), 4)
    
    def test_shuffle_deck(self):
        deck = PokerRules.create_deck()
        shuffled = PokerRules.shuffle_deck(deck, seed=42)
        
        self.assertEqual(len(shuffled), 52)
        self.assertNotEqual(deck, shuffled)
        
        # Same seed produces same shuffle
        shuffled2 = PokerRules.shuffle_deck(deck, seed=42)
        self.assertEqual(shuffled, shuffled2)
    
    def test_deal_cards(self):
        deck = PokerRules.create_deck()
        dealt, remaining = PokerRules.deal_cards(deck, 2)
        
        self.assertEqual(len(dealt), 2)
        self.assertEqual(len(remaining), 50)
    
    def test_deal_flop(self):
        deck = PokerRules.create_deck()
        flop, remaining = PokerRules.deal_flop(deck)
        
        self.assertEqual(len(flop), 3)
        self.assertEqual(len(remaining), 49)
    
    def test_deal_turn(self):
        deck = PokerRules.create_deck()
        turn, remaining = PokerRules.deal_turn(deck)
        
        self.assertIsInstance(turn, str)
        self.assertEqual(len(remaining), 51)
    
    def test_get_action_from_string(self):
        self.assertEqual(PokerRules.get_action_from_string('fold'), Action.FOLD)
        self.assertEqual(PokerRules.get_action_from_string('check'), Action.CHECK)
        self.assertEqual(PokerRules.get_action_from_string('call'), Action.CALL)
        self.assertEqual(PokerRules.get_action_from_string('bet'), Action.BET)
        self.assertEqual(PokerRules.get_action_from_string('raise'), Action.RAISE)
        self.assertEqual(PokerRules.get_action_from_string('all_in'), Action.ALL_IN)
        
        # Unknown action defaults to fold
        self.assertEqual(PokerRules.get_action_from_string('unknown'), Action.FOLD)
    
    def test_get_hand_ranking_name(self):
        self.assertEqual(PokerRules.get_hand_ranking_name(9), "Royal Flush")
        self.assertEqual(PokerRules.get_hand_ranking_name(8), "Straight Flush")
        self.assertEqual(PokerRules.get_hand_ranking_name(0), "High Card")
        self.assertEqual(PokerRules.get_hand_ranking_name(99), "Unknown")


if __name__ == '__main__':
    unittest.main()