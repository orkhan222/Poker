"""
Prediction endpoint for poker agent
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import logging

from api.dependencies import get_agent, get_agent_stats

logger = logging.getLogger("poker-api")

router = APIRouter()


# Request/Response Models
class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class CardModel(BaseModel):
    """Card representation"""
    rank: str = Field(..., description="Card rank (2-9, T, J, Q, K, A)")
    suit: str = Field(..., description="Card suit (h, d, c, s)")
    
    def to_string(self) -> str:
        return f"{self.rank}{self.suit}"


class PlayerStateModel(BaseModel):
    """Player state in the game"""
    position: int = Field(..., description="Player position/seat index")
    stack: float = Field(..., description="Current stack size in chips")
    current_bet: float = Field(0.0, description="Current bet amount in this round")
    is_active: bool = Field(True, description="Whether player is still in the hand")
    nickname: Optional[str] = Field(None, description="Player nickname")


class GameStateModel(BaseModel):
    """Complete game state for prediction"""
    hand_id: Optional[str] = Field(None, description="Unique hand identifier")
    
    # Agent info
    hole_cards: List[CardModel] = Field(..., description="Agent's hole cards", min_items=2, max_items=2)
    agent_stack: float = Field(..., description="Agent's remaining stack")
    agent_position: int = Field(..., description="Agent's position at table")
    
    # Game info
    board_cards: List[CardModel] = Field(default=[], description="Community cards")
    pot: float = Field(..., description="Current pot size")
    current_bet: float = Field(0.0, description="Current bet amount to call")
    min_raise: float = Field(0.0, description="Minimum raise amount")
    street: str = Field(..., description="Current street: preflop, flop, turn, river")
    
    # Opponents
    opponents: List[PlayerStateModel] = Field(default=[], description="Opponent states")
    
    # Action history
    action_history: Optional[List[Dict]] = Field(default=[], description="Previous actions in this hand")
    
    # Timing info (optional)
    opponent_waiting_time: Optional[float] = Field(None, description="Time opponent waited before acting")
    time_since_agent_bet: Optional[float] = Field(None, description="Time since agent's last bet")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hole_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "h"}],
                "agent_stack": 1000,
                "agent_position": 0,
                "board_cards": [{"rank": "Q", "suit": "d"}, {"rank": "J", "suit": "d"}, {"rank": "T", "suit": "d"}],
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
        }


class PredictResponse(BaseModel):
    """Prediction response"""
    action: ActionType = Field(..., description="Recommended action")
    bet_size: float = Field(0.0, description="Bet/raise amount (if applicable)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Reasoning for action (if available)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Action probabilities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    states: List[GameStateModel] = Field(..., description="List of game states")


class BatchPredictResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictResponse] = Field(..., description="List of predictions")
    total_time_ms: float = Field(..., description="Total processing time")


def convert_card_model_to_string(cards: List[CardModel]) -> List[str]:
    """Convert CardModel list to string list"""
    return [c.to_string() for c in cards]


def convert_game_state_to_dict(state: GameStateModel) -> Dict[str, Any]:
    """Convert GameStateModel to dictionary for agent"""
    return {
        'hand_id': state.hand_id,
        'hole_cards': convert_card_model_to_string(state.hole_cards),
        'board_cards': convert_card_model_to_string(state.board_cards),
        'agent_stack': state.agent_stack,
        'agent_position': state.agent_position,
        'pot': state.pot,
        'current_bet': state.current_bet,
        'min_raise': state.min_raise,
        'street': state.street,
        'opponents': [p.dict() for p in state.opponents],
        'action_history': state.action_history,
        'opponent_waiting_time': state.opponent_waiting_time,
        'time_since_agent_bet': state.time_since_agent_bet,
        'legal_actions': ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
    }


@router.post("/predict", response_model=PredictResponse)
async def predict(
    state: GameStateModel,
    agent = Depends(get_agent)
):
    """
    Get action prediction for current game state
    
    Args:
        state: Current game state
        
    Returns:
        Predicted action and metadata
    """
    start_time = time.time()
    
    try:
        # Convert state to agent format
        state_dict = convert_game_state_to_dict(state)
        
        # Get action from agent
        agent_action = agent.act(state_dict)
        
        # Get action probabilities if available
        probabilities = None
        if hasattr(agent, 'get_action_probs'):
            probs = agent.get_action_probs(state_dict)
            if probs is not None:
                action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
                probabilities = {name: float(probs[i]) for i, name in enumerate(action_names)}
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return PredictResponse(
            action=ActionType(agent_action.action),
            bet_size=agent_action.bet_size,
            confidence=agent_action.confidence,
            reasoning=agent_action.reasoning,
            probabilities=probabilities,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    agent = Depends(get_agent)
):
    """
    Get predictions for multiple game states (batch mode)
    
    Args:
        request: Batch of game states
        
    Returns:
        List of predictions
    """
    start_time = time.time()
    predictions = []
    
    for state in request.states:
        try:
            state_dict = convert_game_state_to_dict(state)
            agent_action = agent.act(state_dict)
            
            predictions.append(PredictResponse(
                action=ActionType(agent_action.action),
                bet_size=agent_action.bet_size,
                confidence=agent_action.confidence,
                reasoning=agent_action.reasoning,
                processing_time_ms=0  # Will be updated
            ))
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            predictions.append(PredictResponse(
                action=ActionType.FOLD,
                bet_size=0,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                processing_time_ms=0
            ))
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictResponse(
        predictions=predictions,
        total_time_ms=total_time
    )


@router.get("/predict/stats")
async def get_prediction_stats(stats = Depends(get_agent_stats)):
    """
    Get agent prediction statistics
    
    Returns:
        Statistics about agent's predictions
    """
    return {
        "agent_name": stats.get('name', 'Unknown'),
        "hand_count": stats.get('hand_count', 0),
        "total_profit": stats.get('total_profit', 0),
        "avg_profit_per_hand": stats.get('avg_profit_per_hand', 0)
    }