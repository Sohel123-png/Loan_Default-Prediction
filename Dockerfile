# =============================================================
# Loan Default Prediction — Deployment Dockerfile
# =============================================================
# Yeh ek hi container me app.py (FastAPI backend) + static/
# (frontend) + artifacts/ (trained model) sab kuch chalata hai.
# Alag se frontend deploy karne ki zaroorat NAHI hai, kyunki
# app.py khud StaticFiles se index.html serve karta hai.
#
# Build:
#   docker build -t loan-default-app .
# Run locally:
#   docker run -p 8000:8000 loan-default-app
# =============================================================

FROM python:3.11-slim

WORKDIR /app

# System deps jo lightgbm/xgboost/catboost ko chahiye ho sakte hain
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pehle sirf requirements copy karo -> Docker layer caching se rebuild fast hota hai
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ab poora app code copy karo
COPY webapp/ ./webapp/
COPY artifacts/ ./artifacts/

WORKDIR /app/webapp

EXPOSE 8000

# Production me --reload NAHI use karte (sirf development ke liye tha)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
