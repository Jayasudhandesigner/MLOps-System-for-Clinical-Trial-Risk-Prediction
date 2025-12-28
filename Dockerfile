# =============================================================================
# PRODUCTION DOCKERFILE - Clinical Trial Dropout Prediction API
# =============================================================================
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Production Runtime
FROM python:3.10-slim

# Metadata
LABEL maintainer="Clinical Trial ML Team"
LABEL version="1.0.0"
LABEL description="FastAPI service for clinical trial dropout prediction"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set PATH for user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY api/ /app/api/
COPY src/core/ /app/src/core/
COPY src/utils/ /app/src/utils/
COPY src/__init__.py /app/src/

# Copy data artifacts (preprocessor)
COPY data/processed/preprocessor_dropout_v3_causal.pkl /app/data/processed/

# Copy MLflow database (for model registry access)
COPY mlflow.db /app/

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Change ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables (can be overridden at runtime)
ENV MODEL_VERSION=v3_causal \
    MODEL_STAGE=production \
    DECISION_THRESHOLD=0.20 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Run uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
