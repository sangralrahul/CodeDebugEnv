"""
server/app.py — OpenEnv compatible server entry point.
"""
import uvicorn
import sys
import os

# Add parent directory to path so we can import app and environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860, workers=2)


if __name__ == "__main__":
    main()
