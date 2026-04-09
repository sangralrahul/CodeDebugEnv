"""
server/app.py — re-exports the main FastAPI app for openenv compatibility.
"""
from app import app, main

__all__ = ["app", "main"]
