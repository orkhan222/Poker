"""
FastAPI application main entry point
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
from typing import Dict, Any
import uvicorn
from api.endpoints import predict, health, evaluate
from api.dependencies import get_agent, get_model, initialize_agent, shutdown_agent
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("poker-api", log_dir="experiments/logs")

# Agent instance (global)
_agent = None
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("Starting Poker Agent API...")
    global _agent, _model
    _agent = initialize_agent()
    _model = get_model()
    logger.info("Poker Agent API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Poker Agent API...")
    shutdown_agent()
    logger.info("Poker Agent API shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Poker Agent API",
        description="API for poker agent decision making",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log request
        logger.debug(f"{request.method} {request.url.path} - {process_time:.4f}s")
        
        return response
    
    # Add exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)}
        )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, tags=["prediction"])
    app.include_router(evaluate.router, tags=["evaluation"])
    
    return app


# Create app instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Poker Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "evaluate": "/evaluate"
        }
    }


if __name__ == "__main__":

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )