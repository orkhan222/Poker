import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import glob

class JSONLParser:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_all(self):
        all_hands = []
        all_players = []
        all_actions = []
        all_stack_events = []
        
        jsonl_files = glob.glob(f"{self.raw_dir}/*.jsonl")
        print(f"Found {len(jsonl_files)} JSONL files")
        
        for file_path in jsonl_files:
            print(f"Parsing {file_path}...")
            result = self._parse_file(file_path)
            all_hands.extend(result['hands'])
            all_players.extend(result['players'])
            all_actions.extend(result['actions'])
            all_stack_events.extend(result['stack_events'])
        
        # Save to CSV
        pd.DataFrame(all_hands).to_csv(self.processed_dir / 'hands.csv', index=False)
        pd.DataFrame(all_players).to_csv(self.processed_dir / 'players.csv', index=False)
        pd.DataFrame(all_actions).to_csv(self.processed_dir / 'actions.csv', index=False)
        pd.DataFrame(all_stack_events).to_csv(self.processed_dir / 'stack_events.csv', index=False)
        
        print(f"✅ Saved: {len(all_hands)} hands, {len(all_players)} players, {len(all_actions)} actions, {len(all_stack_events)} stack events")
    
    def _parse_file(self, file_path: str) -> Dict[str, List]:
        hands, players, actions, stack_events = [], [], [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    hand_data = json.loads(line)
                except:
                    continue
                
                hand_id = hand_data.get('hand_id', f"{Path(file_path).stem}_{line_num}")
                
                # Extract hand
                hands.append({
                    'hand_id': hand_id,
                    'hand_index': hand_data.get('hand_index', line_num),
                    'local_hand_index': line_num,
                    'source_file': Path(file_path).name,
                    'board_cards': ' '.join(hand_data.get('board_cards', [])),
                    'total_actions': len(hand_data.get('actions', [])),
                    'total_stack_events': len(hand_data.get('stack_events', [])),
                    'pot_from_stacks': hand_data.get('pot_from_stacks', 0),
                    'pot_from_recognition': hand_data.get('pot_from_recognition', 0)
                })
                
                # Extract players
                for player in hand_data.get('players', []):
                    players.append({
                        'hand_id': hand_id,
                        'hand_index': hand_data.get('hand_index', line_num),
                        'local_hand_index': line_num,
                        'source_file': Path(file_path).name,
                        'position': player.get('position', ''),
                        'nickname': player.get('nickname', ''),
                        'cards': ' '.join(player.get('cards', [])),
                        'starting_stack': player.get('starting_stack', 0),
                        'ending_stack': player.get('ending_stack', 0),
                        'stack_delta': player.get('stack_delta', 0)
                    })
                
                # Extract actions
                for action in hand_data.get('actions', []):
                    actions.append({
                        'hand_id': hand_id,
                        'hand_index': hand_data.get('hand_index', line_num),
                        'local_hand_index': line_num,
                        'source_file': Path(file_path).name,
                        'frame_id': action.get('frame_id', 0),
                        'player_position': action.get('player_position', ''),
                        'player_nickname': action.get('player_nickname', ''),
                        'action': action.get('action', ''),
                        'street': action.get('street', '')
                    })
                
                # Extract stack events
                for event in hand_data.get('stack_events', []):
                    stack_events.append({
                        'hand_id': hand_id,
                        'hand_index': hand_data.get('hand_index', line_num),
                        'local_hand_index': line_num,
                        'source_file': Path(file_path).name,
                        'frame_id': event.get('frame_id', 0),
                        'player_position': event.get('player_position', ''),
                        'event': event.get('event', ''),
                        'stack': event.get('stack', 0),
                        'diff': event.get('diff', 0),
                        'stack_after_event': event.get('stack_after_event', 0)
                    })
        
        return {'hands': hands, 'players': players, 'actions': actions, 'stack_events': stack_events}