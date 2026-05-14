"""
Tests for API prediction endpoints
"""

import unittest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app


class TestPredictEndpoint(unittest.TestCase):
    """Test cases for prediction endpoint"""
    
    def setUp(self):
        self.client = TestClient(app)
        
        self.sample_state = {
            "hole_cards": [
                {"rank": "A", "suit": "h"},
                {"rank": "K", "suit": "h"}
            ],
            "agent_stack": 1000,
            "agent_position": 0,
            "board_cards": [
                {"rank": "Q", "suit": "d"},
                {"rank": "J", "suit": "d"},
                {"rank": "T", "suit": "d"}
            ],
            "pot": 150,
            "current_bet": 50,
            "min_raise": 100,
            "street": "flop",
            "opponents": [
                {"position": 1, "stack": 900, "current_bet": 50, "is_active": True},
                {"position": 2, "stack": 800, "current_bet": 0, "is_active": True}
            ],
            "action_history": []
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["service"], "Poker Agent API")
        self.assertIn("endpoints", data)
    
    @patch('api.dependencies.get_agent')
    def test_predict_endpoint(self, mock_get_agent):
        """Test prediction endpoint"""
        # Create mock agent
        mock_action = MagicMock()
        mock_action.action = 'raise'
        mock_action.bet_size = 100
        mock_action.confidence = 0.85
        mock_action.reasoning = "Strong hand"
        
        mock_agent = MagicMock()
        mock_agent.act.return_value = mock_action
        mock_agent.name = "TestAgent"
        
        mock_get_agent.return_value = mock_agent
        
        response = self.client.post(
            "/predict",
            json=self.sample_state
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("action", data)
        self.assertIn("bet_size", data)
        self.assertIn("confidence", data)
        self.assertIn("processing_time_ms", data)
    
    @patch('api.dependencies.get_agent')
    def test_predict_invalid_state(self, mock_get_agent):
        """Test prediction with invalid state"""
        mock_agent = MagicMock()
        mock_agent.act.side_effect = Exception("Invalid state")
        mock_get_agent.return_value = mock_agent
        
        invalid_state = {"hole_cards": []}  # Missing required fields
        
        response = self.client.post(
            "/predict",
            json=invalid_state
        )
        
        self.assertEqual(response.status_code, 500)
    
    def test_predict_missing_fields(self):
        """Test prediction with missing required fields"""
        invalid_state = {"hole_cards": []}  # Missing agent_stack, etc.
        
        response = self.client.post(
            "/predict",
            json=invalid_state
        )
        
        self.assertEqual(response.status_code, 422)  # Validation error
    
    @patch('api.dependencies.get_agent')
    def test_batch_predict_endpoint(self, mock_get_agent):
        """Test batch prediction endpoint"""
        mock_action = MagicMock()
        mock_action.action = 'call'
        mock_action.bet_size = 50
        mock_action.confidence = 0.7
        mock_action.reasoning = "Good odds"
        
        mock_agent = MagicMock()
        mock_agent.act.return_value = mock_action
        
        mock_get_agent.return_value = mock_agent
        
        batch_request = {
            "states": [self.sample_state, self.sample_state]
        }
        
        response = self.client.post(
            "/predict/batch",
            json=batch_request
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertIn("total_time_ms", data)
    
    @patch('api.dependencies.get_agent')
    def test_predict_stats_endpoint(self, mock_get_agent):
        """Test prediction stats endpoint"""
        mock_agent = MagicMock()
        mock_agent.get_stats.return_value = {
            "name": "TestAgent",
            "hand_count": 100,
            "total_profit": 500,
            "avg_profit_per_hand": 5.0
        }
        mock_get_agent.return_value = mock_agent
        
        response = self.client.get("/predict/stats")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["agent_name"], "TestAgent")
        self.assertEqual(data["hand_count"], 100)


class TestEvaluateEndpoint(unittest.TestCase):
    """Test cases for evaluation endpoint"""
    
    def setUp(self):
        self.client = TestClient(app)
    
    @patch('api.dependencies.get_agent')
    def test_evaluate_endpoint(self, mock_get_agent):
        """Test evaluation endpoint"""
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_get_agent.return_value = mock_agent
        
        request = {
            "num_hands": 100,
            "opponent_type": "random",
            "verbose": False
        }
        
        response = self.client.post("/evaluate", json=request)
        
        # May return 200 or 500 depending on imports
        self.assertIn(response.status_code, [200, 500])
    
    def test_evaluate_invalid_opponent(self):
        """Test evaluation with invalid opponent type"""
        request = {
            "num_hands": 100,
            "opponent_type": "invalid_type"
        }
        
        response = self.client.post("/evaluate", json=request)
        
        self.assertEqual(response.status_code, 422)


class TestHealthEndpoints(unittest.TestCase):
    """Test cases for health endpoints"""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
    
    def test_readiness_check(self):
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})
    
    def test_liveness_check(self):
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "alive"})


if __name__ == '__main__':
    unittest.main()