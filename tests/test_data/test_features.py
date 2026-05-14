"""
Tests for feature extraction
"""

import unittest
import torch
import numpy as np

from src.data.features import PokerFeatureExtractor, PokerState, create_state_tensor


class TestPokerFeatureExtractor(unittest.TestCase):
    """Test cases for feature extraction"""
    
    def setUp(self):
        self.extractor = PokerFeatureExtractor(max_opponents=5)
    
    def test_card_to_one_hot(self):
        """Test card to one-hot conversion"""
        cards = ['Ah', 'Kd']
        one_hot = self.extractor._cards_to_one_hot(cards)
        
        self.assertEqual(len(one_hot), 52)
        # Check that exactly 2 positions are 1
        self.assertEqual(sum(one_hot), 2)
        
        # Test with empty cards
        empty_one_hot = self.extractor._cards_to_one_hot([])
        self.assertEqual(sum(empty_one_hot), 0)
    
    def test_street_to_one_hot(self):
        """Test street to one-hot conversion"""
        preflop = self.extractor._street_to_one_hot('preflop')
        self.assertEqual(preflop[0], 1.0)
        self.assertEqual(sum(preflop), 1.0)
        
        flop = self.extractor._street_to_one_hot('flop')
        self.assertEqual(flop[1], 1.0)
        
        turn = self.extractor._street_to_one_hot('turn')
        self.assertEqual(turn[2], 1.0)
        
        river = self.extractor._street_to_one_hot('river')
        self.assertEqual(river[3], 1.0)
        
        # Unknown street defaults to preflop
        unknown = self.extractor._street_to_one_hot('unknown')
        self.assertEqual(unknown[0], 1.0)
    
    def test_legal_actions_to_vector(self):
        """Test legal actions to vector conversion"""
        legal = ['fold', 'call', 'raise']
        vector = self.extractor._legal_actions_to_vector(legal)
        
        self.assertEqual(len(vector), 6)
        self.assertEqual(vector[0], 1.0)  # fold
        self.assertEqual(vector[2], 1.0)  # call
        self.assertEqual(vector[4], 1.0)  # raise
        self.assertEqual(vector[1], 0.0)  # check
        self.assertEqual(vector[3], 0.0)  # bet
        self.assertEqual(vector[5], 0.0)  # all_in
    
    def test_state_to_tensor(self):
        """Test full state to tensor conversion"""
        state = PokerState(
            hole_cards=['Ah', 'Kd'],
            board_cards=['2h', '3c', '4d'],
            agent_stack=1000,
            pot=150,
            current_bet=50,
            street='flop',
            legal_actions=['fold', 'call', 'raise'],
            opponent_stacks={'opp1': 800, 'opp2': 900}
        )
        
        tensor = self.extractor.state_to_tensor(state)
        
        # Check tensor properties
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dim(), 1)
        self.assertEqual(tensor.shape[0], self.extractor.feature_dim)
        
        # Check values are normalized
        self.assertTrue(torch.all(tensor >= 0))
        self.assertTrue(torch.all(tensor <= 1))
    
    def test_batch_states_to_tensor(self):
        """Test batch conversion"""
        states = [
            PokerState(
                hole_cards=['Ah', 'Kd'],
                board_cards=[],
                agent_stack=1000,
                pot=30,
                current_bet=10,
                street='preflop',
                legal_actions=['fold', 'call', 'raise']
            ),
            PokerState(
                hole_cards=['2c', '7d'],
                board_cards=['As', 'Ks', 'Qs'],
                agent_stack=500,
                pot=200,
                current_bet=100,
                street='river',
                legal_actions=['fold', 'call']
            )
        ]
        
        batch_tensor = self.extractor.batch_states_to_tensor(states)
        
        self.assertIsInstance(batch_tensor, torch.Tensor)
        self.assertEqual(batch_tensor.dim(), 2)
        self.assertEqual(batch_tensor.shape[0], 2)
        self.assertEqual(batch_tensor.shape[1], self.extractor.feature_dim)
    
    def test_create_state_tensor_function(self):
        """Test convenience function"""
        tensor = create_state_tensor(
            hole_cards=['Ah', 'Kd'],
            board_cards=['2h', '3c', '4d'],
            agent_stack=1000,
            pot=150,
            current_bet=50,
            street='flop',
            legal_actions=['fold', 'call', 'raise']
        )
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dim(), 1)


if __name__ == '__main__':
    unittest.main()