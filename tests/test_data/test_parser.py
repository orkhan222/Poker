"""
Tests for JSONL parser
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import pandas as pd

from src.data.jsonl_parser import JSONLParser


class TestJSONLParser(unittest.TestCase):
    """Test cases for JSONL parser"""
    
    def setUp(self):
        """Create temporary test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / 'raw'
        self.processed_dir = Path(self.temp_dir) / 'processed'
        self.raw_dir.mkdir(parents=True)
        
        # Create sample JSONL data
        self.sample_data = [
            {
                "hand_id": "test_hand_001",
                "hand_index": 1,
                "board_cards": ["As", "Ks", "Qs"],
                "players": [
                    {
                        "position": "SB",
                        "nickname": "player1",
                        "cards": ["Jh", "Jd"],
                        "starting_stack": 1000,
                        "ending_stack": 1200,
                        "stack_delta": 200
                    },
                    {
                        "position": "BB",
                        "nickname": "player2",
                        "cards": ["2c", "7d"],
                        "starting_stack": 1000,
                        "ending_stack": 800,
                        "stack_delta": -200
                    }
                ],
                "actions": [
                    {
                        "frame_id": 1,
                        "player_position": "SB",
                        "action": "call",
                        "street": "preflop"
                    },
                    {
                        "frame_id": 2,
                        "player_position": "BB",
                        "action": "raise",
                        "street": "preflop"
                    }
                ],
                "stack_events": [
                    {
                        "frame_id": 1,
                        "player_position": "SB",
                        "event": "update_stack",
                        "stack": 900,
                        "diff": -100
                    }
                ]
            },
            {
                "hand_id": "test_hand_002",
                "hand_index": 2,
                "board_cards": ["2h", "3d", "4c"],
                "players": [
                    {
                        "position": "UTG",
                        "nickname": "player3",
                        "cards": ["Ac", "Ad"],
                        "starting_stack": 1500,
                        "ending_stack": 1700,
                        "stack_delta": 200
                    }
                ],
                "actions": [
                    {
                        "frame_id": 1,
                        "player_position": "UTG",
                        "action": "raise",
                        "street": "preflop"
                    }
                ],
                "stack_events": []
            }
        ]
        
        # Write sample data to JSONL file
        self.test_file = self.raw_dir / 'test_data.jsonl'
        with open(self.test_file, 'w') as f:
            for hand in self.sample_data:
                f.write(json.dumps(hand) + '\n')
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_parser_initialization(self):
        """Test parser initialization"""
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        self.assertEqual(parser.raw_dir, self.raw_dir)
        self.assertEqual(parser.processed_dir, self.processed_dir)
        self.assertTrue(self.processed_dir.exists())
    
    def test_parse_file(self):
        """Test parsing a single JSONL file"""
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        result = parser._parse_file(str(self.test_file))
        
        # Check hands
        self.assertEqual(len(result['hands']), 2)
        self.assertEqual(result['hands'][0]['hand_id'], 'test_hand_001')
        self.assertEqual(result['hands'][1]['hand_id'], 'test_hand_002')
        
        # Check players
        self.assertEqual(len(result['players']), 3)
        
        # Check actions
        self.assertEqual(len(result['actions']), 3)
        
        # Check stack events
        self.assertEqual(len(result['stack_events']), 1)
    
    def test_extract_hand(self):
        """Test hand extraction"""
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        hand_data = self.sample_data[0]
        
        hand = parser._extract_hand(hand_data, 'test_file.jsonl', 0)
        
        self.assertEqual(hand['hand_id'], 'test_hand_001')
        self.assertEqual(hand['source_file'], 'test_file.jsonl')
        self.assertEqual(hand['board_cards'], 'As Ks Qs')
        self.assertEqual(hand['total_actions'], 2)
        self.assertEqual(hand['total_stack_events'], 1)
    
    def test_extract_player(self):
        """Test player extraction"""
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        player_data = self.sample_data[0]['players'][0]
        
        player = parser._extract_player(player_data, 'test_hand_001')
        
        self.assertEqual(player['hand_id'], 'test_hand_001')
        self.assertEqual(player['position'], 'SB')
        self.assertEqual(player['cards'], 'Jh Jd')
        self.assertEqual(player['starting_stack'], 1000)
        self.assertEqual(player['stack_delta'], 200)
    
    def test_extract_action(self):
        """Test action extraction"""
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        action_data = self.sample_data[0]['actions'][0]
        
        action = parser._extract_action(action_data, 'test_hand_001')
        
        self.assertEqual(action['hand_id'], 'test_hand_001')
        self.assertEqual(action['frame_id'], 1)
        self.assertEqual(action['player_position'], 'SB')
        self.assertEqual(action['action'], 'call')
        self.assertEqual(action['street'], 'preflop')
    
    def test_parse_all(self):
        """Test parsing all files"""
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        parser.parse_all()
        
        # Check if CSV files were created
        self.assertTrue((self.processed_dir / 'hands.csv').exists())
        self.assertTrue((self.processed_dir / 'players.csv').exists())
        self.assertTrue((self.processed_dir / 'actions.csv').exists())
        self.assertTrue((self.processed_dir / 'stack_events.csv').exists())
        
        # Check content
        hands_df = pd.read_csv(self.processed_dir / 'hands.csv')
        self.assertEqual(len(hands_df), 2)
    
    def test_empty_file_handling(self):
        """Test handling of empty files"""
        empty_file = self.raw_dir / 'empty.jsonl'
        empty_file.touch()
        
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        # Should not raise exception
        parser.parse_all()
    
    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        malformed_file = self.raw_dir / 'malformed.jsonl'
        with open(malformed_file, 'w') as f:
            f.write('{"invalid": json}\n')
            f.write('{"hand_id": "valid", "players": []}\n')
        
        parser = JSONLParser(self.raw_dir, self.processed_dir)
        # Should skip malformed line
        parser.parse_all()


if __name__ == '__main__':
    unittest.main()