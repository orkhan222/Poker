"""
Tests for policy network
"""

import unittest
import torch

from src.models.policy_net import PolicyNetwork, PolicyNetworkConfig, MultiHeadPolicyNetwork, LSTMPolicyNetwork
from src.models.value_net import ValueNetwork, DualHeadNetwork
from src.models.utils import count_parameters, initialize_weights


class TestPolicyNetwork(unittest.TestCase):
    """Test cases for policy network"""
    
    def setUp(self):
        self.config = PolicyNetworkConfig(
            input_dim=115,
            hidden_dims=[256, 128],
            output_dim=6,
            dropout_rate=0.3
        )
        self.model = PolicyNetwork(self.config)
    
    def test_forward_pass(self):
        batch_size = 4
        x = torch.randn(batch_size, 115)
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, 6))
    
    def test_get_action_probs(self):
        x = torch.randn(1, 115)
        probs = self.model.get_action_probs(x)
        
        self.assertEqual(probs.shape, (1, 6))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
    
    def test_get_action(self):
        x = torch.randn(1, 115)
        action, prob = self.model.get_action(x, deterministic=True)
        
        self.assertIsInstance(action, torch.Tensor)
        self.assertIsInstance(prob, torch.Tensor)
        self.assertTrue(0 <= action.item() < 6)
    
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertGreater(param_count['trainable'], 0)
        self.assertEqual(param_count['trainable'], param_count['total'])


class TestMultiHeadPolicyNetwork(unittest.TestCase):
    """Test cases for multi-head policy network"""
    
    def setUp(self):
        self.model = MultiHeadPolicyNetwork(
            input_dim=115,
            hidden_dims=[256, 128],
            num_actions=6
        )
    
    def test_forward_pass(self):
        x = torch.randn(4, 115)
        action_logits, bet_pred = self.model(x)
        
        self.assertEqual(action_logits.shape, (4, 6))
        self.assertEqual(bet_pred.shape, (4, 1))
    
    def test_get_action_and_bet(self):
        x = torch.randn(1, 115)
        action_idx, bet_size, confidence = self.model.get_action_and_bet(x, max_bet=1000)
        
        self.assertTrue(0 <= action_idx < 6)
        self.assertTrue(0 <= bet_size <= 1000)
        self.assertTrue(0 <= confidence <= 1)


class TestLSTMPolicyNetwork(unittest.TestCase):
    """Test cases for LSTM policy network"""
    
    def setUp(self):
        self.model = LSTMPolicyNetwork(
            input_dim=115,
            hidden_dim=128,
            num_layers=2,
            output_dim=6
        )
    
    def test_forward_pass(self):
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 115)
        
        output, hidden = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, 6))
        self.assertEqual(len(hidden), 2)  # h and c
        self.assertEqual(hidden[0].shape, (2, batch_size, 128))
    
    def test_init_hidden(self):
        batch_size = 4
        h = self.model.init_hidden(batch_size, torch.device('cpu'))
        
        self.assertEqual(len(h), 2)
        self.assertEqual(h[0].shape, (2, batch_size, 128))


class TestValueNetwork(unittest.TestCase):
    """Test cases for value network"""
    
    def setUp(self):
        self.model = ValueNetwork()
    
    def test_forward_pass(self):
        x = torch.randn(4, 115)
        value = self.model(x)
        
        self.assertEqual(value.shape, (4,))
    
    def test_get_value(self):
        x = torch.randn(1, 115)
        value = self.model.get_value(x)
        
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(value.shape, (1,))


class TestDualHeadNetwork(unittest.TestCase):
    """Test cases for dual head network"""
    
    def setUp(self):
        self.model = DualHeadNetwork(
            input_dim=115,
            hidden_dims=[256, 128],
            num_actions=6
        )
    
    def test_forward_pass(self):
        x = torch.randn(4, 115)
        policy_logits, value = self.model(x)
        
        self.assertEqual(policy_logits.shape, (4, 6))
        self.assertEqual(value.shape, (4,))
    
    def test_get_action_and_value(self):
        x = torch.randn(1, 115)
        action_idx, value, prob = self.model.get_action_and_value(x, deterministic=True)
        
        self.assertTrue(0 <= action_idx < 6)
        self.assertIsInstance(value, float)
        self.assertTrue(0 <= prob <= 1)


class TestWeightInitialization(unittest.TestCase):
    """Test cases for weight initialization"""
    
    def test_initialize_weights(self):
        model = PolicyNetwork()
        initialize_weights(model, 'xavier_uniform')
        
        # Check that weights were initialized (not all zeros)
        for param in model.parameters():
            if param.dim() >= 2:
                self.assertFalse(torch.all(param == 0))


if __name__ == '__main__':
    unittest.main()