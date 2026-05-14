"""
LLM-based poker agent using language models
"""

import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import random
import openai
from .base_agent import BaseAgent, AgentAction


@dataclass
class LLMAgentConfig:
    """Configuration for LLM agent"""
    model: str = "gpt-3.5-turbo"  # or "llama2", "claude", etc.
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    use_local: bool = False
    temperature: float = 0.7
    max_tokens: int = 150
    few_shot_examples: bool = True
    include_reasoning: bool = True


class LLMAgent(BaseAgent):
    """
    LLM-based poker agent that uses language models to make decisions.
    Supports both API-based (OpenAI, Anthropic) and local models.
    """
    
    def __init__(self, name: str = "LLMAgent", config: LLMAgentConfig = None):
        super().__init__(name)
        self.config = config or LLMAgentConfig()
        self.client = None
        self._init_client()
        
        # Few-shot examples
        self.examples = self._get_few_shot_examples()
    
    def _init_client(self):
        """Initialize the LLM client"""
        if not self.config.use_local:
            try:
                if "gpt" in self.config.model.lower():
                    openai.api_key = self.config.api_key
                    self.client = "openai"
                elif "claude" in self.config.model.lower():
                    # Anthropic client
                    self.client = "anthropic"
                else:
                    self.client = "generic"
            except ImportError:
                print("Warning: LLM library not installed. Using fallback.")
                self.client = None
        else:
            # Local model (e.g., llama.cpp, Ollama)
            self.client = "local"
    
    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for prompting"""
        return """
Example 1:
State: Preflop, Hole cards: As Ks, Stack: 1000, Pot: 30, Current bet: 10
Legal actions: fold, call, raise
Reasoning: I have a strong starting hand (AK suited). I should raise to build the pot.
Action: raise, bet_size: 30

Example 2:
State: Flop, Board: 2h 7d Qc, Hole cards: 9h 9d, Stack: 800, Pot: 100, Current bet: 0
Legal actions: check, bet
Reasoning: I have middle pair (nines). With no overcards on the flop, I should bet for value.
Action: bet, bet_size: 50

Example 3:
State: Turn, Board: Ah Kh Qc Jd, Hole cards: Tc 2h, Stack: 500, Pot: 200, Current bet: 100
Legal actions: fold, call, raise
Reasoning: I have a straight draw but need a 9. Pot odds are good but calling is risky with low confidence.
Action: call, bet_size: 100

Example 4:
State: River, Board: As Ks Qs Js Ts, Hole cards: 2d 3h, Stack: 300, Pot: 150, Current bet: 150
Legal actions: fold, call, raise
Reasoning: There's a royal flush possible on board. I have nothing. I should fold.
Action: fold, bet_size: 0
"""
    
    def _build_prompt(self, state: Dict) -> str:
        """Build prompt for LLM"""
        parsed = self._parse_state(state)
        
        # Get card strength description
        hand_desc = self._describe_hand(parsed['hole_cards'])
        board_desc = self._describe_board(parsed['board_cards'])
        
        prompt = f"""You are a professional Texas Hold'em poker player. Analyze the situation and decide the best action.

Current Game State:
- Your hole cards: {parsed['hole_cards']} ({hand_desc})
- Community cards: {board_desc if parsed['board_cards'] else 'Not yet dealt'}
- Current street: {parsed['street'].upper()}
- Your stack: ${parsed['agent_stack']:.0f}
- Pot size: ${parsed['pot']:.0f}
- Current bet to call: ${parsed['current_bet']:.0f}
- Minimum raise: ${parsed['min_raise']:.0f}
- Legal actions: {', '.join(parsed['legal_actions'])}

"""
        
        # Add recent action history
        if parsed['action_history']:
            prompt += "Recent actions:\n"
            for action in parsed['action_history'][-5:]:
                prompt += f"- {action.get('player', 'Unknown')}: {action.get('action', '')} ${action.get('amount', 0):.0f}\n"
            prompt += "\n"
        
        # Add few-shot examples
        if self.config.few_shot_examples:
            prompt += self.examples
            prompt += "\n"
        
        prompt += """Now analyze the current situation and respond in the following format:
Reasoning: [Your reasoning here]
Action: [action_name], bet_size: [amount]

Choose the best action based on optimal poker strategy.
"""
        
        return prompt
    
    def _describe_hand(self, cards: List[str]) -> str:
        """Describe hole cards strength"""
        if len(cards) < 2:
            return "Unknown"
        
        ranks = [c[0] for c in cards]
        suits = [c[1] if len(c) == 2 else c[2] for c in cards]
        
        is_pair = ranks[0] == ranks[1]
        is_suited = suits[0] == suits[1]
        
        rank_map = {'A': 'Ace', 'K': 'King', 'Q': 'Queen', 'J': 'Jack', 
                    'T': 'Ten', '9': 'Nine', '8': 'Eight', '7': 'Seven',
                    '6': 'Six', '5': 'Five', '4': 'Four', '3': 'Three', '2': 'Two'}
        
        rank_names = [rank_map.get(r, r) for r in ranks]
        
        if is_pair:
            return f"Pocket {rank_names[0]}s"
        elif is_suited:
            return f"{rank_names[0]}-{rank_names[1]} suited"
        else:
            return f"{rank_names[0]}-{rank_names[1]} offsuit"
    
    def _describe_board(self, cards: List[str]) -> str:
        """Describe board cards"""
        if not cards:
            return "No community cards yet"
        
        if len(cards) == 3:
            return f"Flop: {', '.join(cards)}"
        elif len(cards) == 4:
            return f"Turn: {', '.join(cards)}"
        elif len(cards) == 5:
            return f"River: {', '.join(cards)}"
        return ', '.join(cards)
    
    def _parse_llm_response(self, response: str, legal_actions: List[str]) -> AgentAction:
        """Parse LLM response into AgentAction"""
        action = "fold"
        bet_size = 0.0
        reasoning = ""
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\nAction:|$)', response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r'Action:\s*(\w+)\s*(?:,?\s*bet_size:\s*(\d+(?:\.\d+)?))?', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).lower()
            if action_match.group(2):
                bet_size = float(action_match.group(2))
        
        # Validate action
        if action not in legal_actions:
            # Find closest legal action
            if 'fold' in legal_actions:
                action = 'fold'
            elif 'check' in legal_actions:
                action = 'check'
            elif 'call' in legal_actions:
                action = 'call'
            else:
                action = legal_actions[0] if legal_actions else 'fold'
        
        return AgentAction(
            action=action,
            bet_size=bet_size,
            confidence=0.7,
            reasoning=reasoning
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API or local model"""
        if self.client == "openai":
            try:
                import openai
                response = openai.ChatCompletion.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return self._fallback_response()
        
        elif self.client == "local":
            # Placeholder for local model
            return self._fallback_response()
        
        else:
            # Fallback to rule-based
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Fallback response when LLM is unavailable"""
        return """Reasoning: Cannot access LLM, using conservative play.
Action: fold, bet_size: 0"""
    
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """Make decision using LLM"""
        parsed = self._parse_state(state)
        
        # Build prompt
        prompt = self._build_prompt(state)
        
        # Call LLM
        try:
            response = self._call_llm(prompt)
            action = self._parse_llm_response(response, parsed['legal_actions'])
        except Exception as e:
            print(f"LLM error: {e}")
            # Fallback to random legal action
            legal = parsed['legal_actions']
            action_name = random.choice(legal) if legal else 'fold'
            action = AgentAction(action=action_name, bet_size=0, confidence=0.3)
        
        # Record action
        self.action_history.append({
            'state': parsed,
            'action': action.action,
            'bet_size': action.bet_size,
            'reasoning': action.reasoning
        })
        
        return action
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.action_history = []


class SimpleLLMAgent(LLMAgent):
    """Simpler LLM agent with basic prompting"""
    
    def _build_prompt(self, state: Dict) -> str:
        """Simplified prompt"""
        parsed = self._parse_state(state)
        
        prompt = f"""Poker decision:
Cards: {parsed['hole_cards']}
Board: {parsed['board_cards']}
Street: {parsed['street']}
Stack: ${parsed['agent_stack']:.0f}
Pot: ${parsed['pot']:.0f}
To call: ${parsed['current_bet']:.0f}
Legal: {parsed['legal_actions']}

Choose best action (fold/call/raise/check). If raise, specify amount.
Response format: ACTION [amount]
"""
        return prompt
    
    def _parse_llm_response(self, response: str, legal_actions: List[str]) -> AgentAction:
        """Simpler parsing"""
        response = response.lower().strip()
        
        # Check for raise with amount
        if 'raise' in response:
            import re
            amounts = re.findall(r'\d+(?:\.\d+)?', response)
            bet_size = float(amounts[0]) if amounts else 50
            return AgentAction(action='raise', bet_size=bet_size, confidence=0.6)
        
        # Check for other actions
        for action in ['fold', 'check', 'call']:
            if action in response:
                return AgentAction(action=action, bet_size=0, confidence=0.6)
        
        # Default
        return AgentAction(action='fold', bet_size=0, confidence=0.3)