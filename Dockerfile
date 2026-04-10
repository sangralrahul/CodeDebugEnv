FROM python:3.11-slim

# Force rebuild v3
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY environment.py app.py inference.py openenv.yaml ./
COPY server/ ./server/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2", "--log-level", "info"]
