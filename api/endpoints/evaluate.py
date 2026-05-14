"""
Evaluation endpoint for testing agent performance
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import time
import logging

from api.dependencies import get_agent
from src.environment.simulator import MultiAgentSimulator
from src.agents.rule_agent import RuleBasedAgent, RandomAgent

logger = logging.getLogger("poker-api")

router = APIRouter()


class EvaluateRequest(BaseModel):
    """Evaluation request"""
    num_hands: int = Field(100, ge=1, le=10000, description="Number of hands to evaluate")
    opponent_type: str = Field("random", description="Opponent type: random, conservative, aggressive, tight, loose")
    verbose: bool = Field(False, description="Print detailed results")


class EvaluateResponse(BaseModel):
    """Evaluation response"""
    agent_name: str
    num_hands: int
    opponent_type: str
    win_rate: float
    loss_rate: float
    tie_rate: float
    avg_profit: float
    total_profit: float
    final_stack: float
    processing_time_seconds: float
    details: Optional[Dict] = None


class CompareRequest(BaseModel):
    """Agent comparison request"""
    agent_names: List[str] = Field(..., description="Names of agents to compare")
    num_hands: int = Field(500, ge=1, le=5000, description="Number of hands per pair")


class CompareResponse(BaseModel):
    """Comparison response"""
    results: Dict[str, Dict[str, float]]
    summary: Dict[str, Any]
    processing_time_seconds: float


def get_opponent_agent(opponent_type: str, name: str = "Opponent"):
    """Get opponent agent by type"""
    opponents = {
        "random": RandomAgent(name=name),
        "conservative": RuleBasedAgent(name=name, strategy="conservative"),
        "aggressive": RuleBasedAgent(name=name, strategy="aggressive"),
        "tight": RuleBasedAgent(name=name, strategy="tight"),
        "loose": RuleBasedAgent(name=name, strategy="loose"),
        "always_fold": RuleBasedAgent(name=name, strategy="always_fold"),
        "always_call": RuleBasedAgent(name=name, strategy="always_call")
    }
    
    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    return opponents[opponent_type]


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_agent(
    request: EvaluateRequest,
    agent = Depends(get_agent)
):
    """
    Evaluate agent against a specified opponent
    
    Args:
        request: Evaluation parameters
        
    Returns:
        Evaluation results
    """
    start_time = time.time()
    
    try:
        # Get opponent
        opponent = get_opponent_agent(request.opponent_type, "Opponent")
        
        # Create simulator
        simulator = MultiAgentSimulator(
            num_players=2,
            starting_stack=1000,
            verbose=request.verbose
        )
        
        # Run head-to-head
        simulator.agents = {0: agent, 1: opponent}
        simulator.agent_names = {0: agent.name, 1: opponent.name}
        
        # Run tournament
        tournament = simulator.run_tournament(
            num_hands=request.num_hands,
            verbose=request.verbose
        )
        
        win_rate = tournament.get_win_rate(agent.name)
        loss_rate = tournament.get_win_rate(opponent.name)
        tie_rate = 1 - win_rate - loss_rate
        avg_profit = tournament.get_average_profit(agent.name)
        total_profit = tournament.total_profit.get(agent.name, 0)
        final_stack = tournament.final_stacks.get(agent.name, 1000)
        
        processing_time = time.time() - start_time
        
        return EvaluateResponse(
            agent_name=agent.name,
            num_hands=request.num_hands,
            opponent_type=request.opponent_type,
            win_rate=win_rate,
            loss_rate=loss_rate,
            tie_rate=tie_rate,
            avg_profit=avg_profit,
            total_profit=total_profit,
            final_stack=final_stack,
            processing_time_seconds=processing_time,
            details=tournament.summary()
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/baseline")
async def evaluate_against_baselines(
    num_hands: int = 500,
    agent = Depends(get_agent)
):
    """
    Evaluate agent against all baseline opponents
    
    Args:
        num_hands: Number of hands per opponent
        
    Returns:
        Results against all baselines
    """
    start_time = time.time()
    results = {}
    
    opponent_types = ["random", "conservative", "aggressive", "tight", "loose", "always_fold", "always_call"]
    
    for opp_type in opponent_types:
        try:
            opponent = get_opponent_agent(opp_type, opp_type.capitalize())
            
            simulator = MultiAgentSimulator(num_players=2, starting_stack=1000, verbose=False)
            simulator.agents = {0: agent, 1: opponent}
            simulator.agent_names = {0: agent.name, 1: opponent.name}
            
            tournament = simulator.run_tournament(num_hands=num_hands, verbose=False)
            
            results[opp_type] = {
                "win_rate": tournament.get_win_rate(agent.name),
                "avg_profit": tournament.get_average_profit(agent.name),
                "total_profit": tournament.total_profit.get(agent.name, 0)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating against {opp_type}: {e}")
            results[opp_type] = {"error": str(e)}
    
    processing_time = time.time() - start_time
    
    # Calculate average win rate
    win_rates = [r["win_rate"] for r in results.values() if "win_rate" in r]
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
    
    return {
        "agent_name": agent.name,
        "num_hands": num_hands,
        "results": results,
        "average_win_rate": avg_win_rate,
        "best_against": max(results.items(), key=lambda x: x[1].get("win_rate", 0))[0] if results else None,
        "worst_against": min(results.items(), key=lambda x: x[1].get("win_rate", 1))[0] if results else None,
        "processing_time_seconds": processing_time
    }


@router.post("/evaluate/compare", response_model=CompareResponse)
async def compare_agents(
    request: CompareRequest
):
    """
    Compare multiple agents against each other
    
    Args:
        request: Comparison parameters
        
    Returns:
        Comparison matrix
    """
    start_time = time.time()
    
    # This would load different agent models
    # For now, return placeholder
    # In production, you'd load models from checkpoints
    
    results = {}
    for name in request.agent_names:
        results[name] = {}
        for other in request.agent_names:
            if name == other:
                results[name][other] = 0.5
            else:
                # Simulate comparison (replace with actual evaluation)
                results[name][other] = 0.5 + (hash(name) - hash(other)) % 20 / 100
    
    # Calculate summary
    avg_win_rates = {name: sum(results[name].values()) / len(results[name]) for name in request.agent_names}
    ranking = sorted(avg_win_rates.items(), key=lambda x: x[1], reverse=True)
    
    processing_time = time.time() - start_time
    
    return CompareResponse(
        results=results,
        summary={
            "average_win_rates": avg_win_rates,
            "ranking": [{"position": i+1, "name": name, "score": score} 
                       for i, (name, score) in enumerate(ranking)]
        },
        processing_time_seconds=processing_time
    )