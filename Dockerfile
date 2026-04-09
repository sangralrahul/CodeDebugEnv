FROM python:3.11-slim

# HF Spaces metadata
# title: CodeDebugEnv
# emoji: 🐛
# colorFrom: blue
# colorTo: green
# sdk: docker
# pinned: false

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY environment.py app.py inference.py openenv.yaml ./

# HF Spaces runs on port 7860
EXPOSE 7860

# Run with multiple workers for concurrency
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "2", \
     "--log-level", "info"]
