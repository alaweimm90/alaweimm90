"""
HELIOS REST API - FastAPI Application

Provides comprehensive REST API with:
- Domain and algorithm information
- Hypothesis validation
- Performance leaderboards
- Real-time updates via WebSockets
- Data export functionality
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .server import create_app

__all__ = ['create_app']
