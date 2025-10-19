FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- MLOps: Model Configuration and Caching ---
# Set environment variable for the model name
ENV MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english

# Pre-download and cache the model during the build process
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='${MODEL_NAME}')"

COPY train.py api.py ./

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]