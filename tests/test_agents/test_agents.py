"""
Tests for poker agents
"""

import unittest
from unittest.mock import Mock, patch
import torch

from src.agents.base_agent import BaseAgent, AgentAction
from src.agents.policy_agent import PolicyAgent
from src.agents.rule_agent import RuleBasedAgent, RandomAgent
from src.agents.llm_agent import LLMAgent, LLMAgentConfig


class TestBaseAgent(unittest.TestCase):
    """Test cases for base agent"""
    
    def test_agent_action_creation(self):
        action = AgentAction(action='raise', bet_size=100, confidence=0.8)
        self.assertEqual(action.action, 'raise')
        self.assertEqual(action.bet_size, 100)
        self.assertEqual(action.confidence, 0.8)
    
    def test_agent_action_to_tuple(self):
        action = AgentAction(action='call', bet_size=50)
        action_tuple = action.to_tuple()
        self.assertEqual(action_tuple, ('call', 50))
    
    def test_base_agent_abstract(self):
        """BaseAgent should be abstract"""
        with self.assertRaises(TypeError):
            BaseAgent()  # Cannot instantiate abstract class


class TestRuleBasedAgent(unittest.TestCase):
    """Test cases for rule-based agent"""
    
    def setUp(self):
        self.agent = RuleBasedAgent(strategy='conservative')
        
        self.test_state = {
            'hole_cards': ['As', 'Ks'],
            'board_cards': [],
            'agent_stack': 1000,
            'pot': 30,
            'current_bet': 10,
            'street': 'preflop',
            'legal_actions': ['fold', 'call', 'raise', 'bet']
        }
    
    def test_agent_initialization(self):
        self.assertEqual(self.agent.name, 'RuleAgent')
        self.assertEqual(self.agent.strategy, 'conservative')
    
    def test_agent_act_returns_action(self):
        action = self.agent.act(self.test_state)
        self.assertIsInstance(action, AgentAction)
        self.assertIn(action.action, ['fold', 'check', 'call', 'bet', 'raise', 'all_in'])
    
    def test_conservative_strategy_strong_hand(self):
        """Conservative should raise with strong hand"""
        state = self.test_state.copy()
        state['hole_cards'] = ['As', 'Ks']  # Very strong
        
        action = self.agent.act(state)
        # Should raise or bet with strong hand
        self.assertIn(action.action, ['raise', 'bet'])
    
    def test_always_fold_strategy(self):
        agent = RuleBasedAgent(strategy='always_fold')
        action = agent.act(self.test_state)
        self.assertEqual(action.action, 'fold')
    
    def test_always_call_strategy(self):
        agent = RuleBasedAgent(strategy='always_call')
        state = self.test_state.copy()
        state['current_bet'] = 50
        
        action = agent.act(state)
        self.assertEqual(action.action, 'call')
        self.assertEqual(action.bet_size, 50)
    
    def test_reset(self):
        self.agent.reset()
        self.assertEqual(self.agent.action_history, [])


class TestRandomAgent(unittest.TestCase):
    """Test cases for random agent"""
    
    def setUp(self):
        self.agent = RandomAgent()
        self.test_state = {
            'hole_cards': ['Ah', 'Kd'],
            'board_cards': [],
            'agent_stack': 1000,
            'pot': 30,
            'current_bet': 10,
            'street': 'preflop',
            'legal_actions': ['fold', 'call', 'raise']
        }
    
    def test_random_agent_act(self):
        action = self.agent.act(self.test_state)
        self.assertIsInstance(action, AgentAction)
        self.assertIn(action.action, self.test_state['legal_actions'])
    
    def test_random_agent_multiple_actions(self):
        actions = set()
        for _ in range(100):
            action = self.agent.act(self.test_state)
            actions.add(action.action)
        
        # Should see multiple action types
        self.assertGreater(len(actions), 1)


class TestPolicyAgent(unittest.TestCase):
    """Test cases for policy agent"""
    
    def setUp(self):
        self.agent = PolicyAgent(name="TestPolicy", device='cpu')
        self.test_state = {
            'hole_cards': ['Ah', 'Kd'],
            'board_cards': ['2h', '3c', '4d'],
            'agent_stack': 1000,
            'pot': 150,
            'current_bet': 50,
            'street': 'flop',
            'legal_actions': ['fold', 'call', 'raise', 'bet'],
            'min_raise': 100
        }
    
    def test_policy_agent_initialization(self):
        self.assertEqual(self.agent.name, 'TestPolicy')
        self.assertIsNotNone(self.agent.model)
    
    def test_policy_agent_act_returns_action(self):
        action = self.agent.act(self.test_state)
        self.assertIsInstance(action, AgentAction)
        self.assertIn(action.action, ['fold', 'check', 'call', 'bet', 'raise', 'all_in'])
    
    def test_policy_agent_save_load_model(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            self.agent.save_model(f.name)
            
            # Load into new agent
            new_agent = PolicyAgent(device='cpu')
            new_agent.load_model(f.name)
            
            # Both should have same state dict keys
            self.assertEqual(
                set(self.agent.model.state_dict().keys()),
                set(new_agent.model.state_dict().keys())
            )
    
    def test_policy_agent_reset(self):
        self.agent.reset()
        self.assertEqual(self.agent.action_history, [])
    
    def test_policy_agent_training_mode(self):
        self.agent.set_training_mode(True)
        self.assertTrue(self.agent.model.training)
        
        self.agent.set_training_mode(False)
        self.assertFalse(self.agent.model.training)


class TestLLMAgent(unittest.TestCase):
    """Test cases for LLM agent"""
    
    def setUp(self):
        self.config = LLMAgentConfig(
            use_local=True,  # Use local fallback to avoid API calls
            few_shot_examples=False
        )
        self.agent = LLMAgent(name="TestLLM", config=self.config)
        self.test_state = {
            'hole_cards': ['Ah', 'Kd'],
            'board_cards': [],
            'agent_stack': 1000,
            'pot': 30,
            'current_bet': 10,
            'street': 'preflop',
            'legal_actions': ['fold', 'call', 'raise']
        }
    
    def test_llm_agent_initialization(self):
        self.assertEqual(self.agent.name, 'TestLLM')
    
    def test_llm_agent_act_fallback(self):
        # Should use fallback when no API
        action = self.agent.act(self.test_state)
        self.assertIsInstance(action, AgentAction)
        self.assertIn(action.action, self.test_state['legal_actions'])
    
    def test_build_prompt(self):
        prompt = self.agent._build_prompt(self.test_state)
        self.assertIsInstance(prompt, str)
        self.assertIn('Ah Kd', prompt)
        self.assertIn('preflop', prompt)
    
    def test_describe_hand(self):
        desc = self.agent._describe_hand(['As', 'Ks'])
        self.assertIn('suited', desc)
        
        desc = self.agent._describe_hand(['Ah', 'Ad'])
        self.assertIn('Pocket', desc)
        
        desc = self.agent._describe_hand(['2c', '7d'])
        self.assertIn('offsuit', desc)
    
    def test_parse_llm_response(self):
        response = "Reasoning: Strong hand\nAction: raise, bet_size: 100"
        action = self.agent._parse_llm_response(response, ['fold', 'call', 'raise'])
        
        self.assertEqual(action.action, 'raise')
        self.assertEqual(action.bet_size, 100)
    
    def test_fallback_response(self):
        response = self.agent._fallback_response()
        self.assertIsInstance(response, str)
        self.assertIn('fold', response.lower())


if __name__ == '__main__':
    unittest.main()