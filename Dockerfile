# --- Customer Churn Prediction API ---
FROM python:3.11-slim

WORKDIR /code

# Install dependencies first (better layer caching)
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained model artifacts and app code
COPY models/ ./models/
COPY app/main.py ./app/main.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
