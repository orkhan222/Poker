"""
Health check endpoint
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import time
import psutil
import torch

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    version: str
    uptime_seconds: float = None
    system: Dict[str, Any]
    model_loaded: bool
    device: str


# Start time for uptime tracking
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for load balancers and monitoring
    """
    uptime = time.time() - _start_time
    
    # System info
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024)
    }
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        system_info["cuda_available"] = True
        system_info["cuda_device_count"] = torch.cuda.device_count()
        system_info["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        uptime_seconds=uptime,
        system=system_info,
        model_loaded=True,
        device=device
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes
    """
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes
    """
    return {"status": "alive"}