# MLOps Week 2-3: Containerize the serving layer so it runs identically
# anywhere - your laptop, CI, or a deployment host like Render/Railway.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer -> Docker caches this unless
# requirements change, so rebuilds after a code-only change are fast)
COPY requirements.txt .
COPY mlops/requirements-mlops.txt ./mlops/requirements-mlops.txt
RUN pip install --no-cache-dir -r requirements.txt -r mlops/requirements-mlops.txt

# Now copy the actual application code
COPY . .

# Train the model at build time so the image ships with a ready artifact.
# (For a larger real system you'd instead pull a versioned artifact from
# an MLflow model registry / S3 bucket - baking it in at build time is the
# right call at this project's scale.)
RUN python mlops/train.py

EXPOSE 8000

CMD ["uvicorn", "mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
