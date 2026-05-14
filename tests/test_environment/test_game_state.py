"""
Tests for game state management
"""

import unittest
from src.environment.game_state import GameState, PlayerState, Street, Action


class TestPlayerState(unittest.TestCase):
    """Test cases for PlayerState"""
    
    def test_player_initialization(self):
        player = PlayerState(position=0, stack=1000)
        self.assertEqual(player.position, 0)
        self.assertEqual(player.stack, 1000)
        self.assertEqual(player.current_bet, 0)
        self.assertTrue(player.is_active)
        self.assertFalse(player.is_all_in)
    
    def test_can_act(self):
        player = PlayerState(position=0, stack=1000)
        self.assertTrue(player.can_act())
        
        player.is_active = False
        self.assertFalse(player.can_act())
        
        player.is_active = True
        player.is_all_in = True
        self.assertFalse(player.can_act())
    
    def test_reset_for_new_hand(self):
        player = PlayerState(position=0, stack=1000)
        player.current_bet = 500
        player.total_bet = 500
        player.hole_cards = ['Ah', 'Kd']
        
        player.reset_for_new_hand(1500)
        
        self.assertEqual(player.stack, 1500)
        self.assertEqual(player.current_bet, 0)
        self.assertEqual(player.total_bet, 0)
        self.assertEqual(player.hole_cards, [])
        self.assertTrue(player.is_active)
        self.assertFalse(player.is_all_in)


class TestGameState(unittest.TestCase):
    """Test cases for GameState"""
    
    def setUp(self):
        self.state = GameState()
        self.state.players = [
            PlayerState(position=0, stack=1000),
            PlayerState(position=1, stack=1000),
            PlayerState(position=2, stack=1000)
        ]
        self.state.current_player_idx = 0
        self.state.current_bet = 0
    
    def test_num_players(self):
        self.assertEqual(self.state.num_players, 3)
    
    def test_num_active_players(self):
        self.assertEqual(self.state.num_active_players, 3)
        self.state.players[0].is_active = False
        self.assertEqual(self.state.num_active_players, 2)
    
    def test_current_player(self):
        player = self.state.current_player
        self.assertEqual(player.position, 0)
        
        self.state.current_player_idx = 1
        player = self.state.current_player
        self.assertEqual(player.position, 1)
    
    def test_is_preflop(self):
        self.assertTrue(self.state.is_preflop)
        self.state.street = Street.FLOP
        self.assertFalse(self.state.is_preflop)
        self.assertTrue(self.state.is_flop)
    
    def test_get_active_players(self):
        self.state.players[1].is_active = False
        active = self.state.get_active_players()
        self.assertEqual(len(active), 2)
        self.assertEqual(active[0].position, 0)
        self.assertEqual(active[1].position, 2)
    
    def test_get_legal_actions_check(self):
        """Test legal actions when no bet to call"""
        self.state.current_bet = 0
        legal = self.state.get_legal_actions(0)
        
        self.assertIn(Action.CHECK, legal)
        self.assertIn(Action.BET, legal)
        # Fold not legal when no bet
        self.assertNotIn(Action.FOLD, legal)
    
    def test_get_legal_actions_call(self):
        """Test legal actions when there's a bet"""
        self.state.current_bet = 50
        self.state.players[0].current_bet = 0
        legal = self.state.get_legal_actions(0)
        
        self.assertIn(Action.CALL, legal)
        self.assertIn(Action.FOLD, legal)
        self.assertIn(Action.RAISE, legal)
    
    def test_apply_action_fold(self):
        """Test fold action"""
        player = self.state.players[0]
        self.state.current_bet = 50
        player.current_bet = 50
        
        amount = self.state.apply_action(Action.FOLD)
        
        self.assertFalse(player.is_active)
        self.assertEqual(amount, 50)
    
    def test_apply_action_check(self):
        """Test check action"""
        player = self.state.players[0]
        self.state.current_bet = 0
        
        amount = self.state.apply_action(Action.CHECK)
        
        self.assertTrue(player.is_active)
        self.assertEqual(amount, 0)
    
    def test_apply_action_call(self):
        """Test call action"""
        player = self.state.players[0]
        self.state.current_bet = 50
        player.current_bet = 0
        player.stack = 1000
        
        amount = self.state.apply_action(Action.CALL)
        
        self.assertEqual(player.stack, 950)
        self.assertEqual(player.current_bet, 50)
        self.assertEqual(self.state.pot, 50)
        self.assertEqual(amount, 50)
    
    def test_apply_action_bet(self):
        """Test bet action"""
        player = self.state.players[0]
        self.state.current_bet = 0
        self.state.min_raise = 100
        player.stack = 1000
        
        amount = self.state.apply_action(Action.BET, 100)
        
        self.assertEqual(player.stack, 900)
        self.assertEqual(player.current_bet, 100)
        self.assertEqual(self.state.current_bet, 100)
        self.assertEqual(amount, 100)
    
    def test_is_betting_round_complete(self):
        """Test betting round completion"""
        # All players have acted
        for player in self.state.players:
            player.has_acted = True
        self.assertTrue(self.state.is_betting_round_complete())
        
        # One player hasn't acted
        self.state.players[1].has_acted = False
        self.assertFalse(self.state.is_betting_round_complete())
    
    def test_next_street(self):
        """Test moving to next street"""
        self.assertEqual(self.state.street, Street.PREFLOP)
        
        self.state.next_street()
        self.assertEqual(self.state.street, Street.FLOP)
        
        self.state.next_street()
        self.assertEqual(self.state.street, Street.TURN)
        
        self.state.next_street()
        self.assertEqual(self.state.street, Street.RIVER)
        
        self.state.next_street()
        self.assertEqual(self.state.street, Street.SHOWDOWN)
        self.assertTrue(self.state.is_terminal)
    
    def test_to_dict(self):
        """Test state to dictionary conversion"""
        state_dict = self.state.to_dict()
        
        self.assertIn('hand_id', state_dict)
        self.assertIn('street', state_dict)
        self.assertIn('pot', state_dict)
        self.assertIn('players', state_dict)
        self.assertIn('legal_actions', state_dict)
    
    def test_copy(self):
        """Test deep copy"""
        original = self.state
        copied = original.copy()
        
        self.assertEqual(original.num_players, copied.num_players)
        self.assertEqual(original.current_bet, copied.current_bet)
        
        # Modify copy, original should remain unchanged
        copied.current_bet = 100
        self.assertEqual(original.current_bet, 0)


if __name__ == '__main__':
    unittest.main()