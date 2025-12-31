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
COPY src/ /app/src/
COPY monitoring/ /app/monitoring/

# Copy model files (from artifacts or local)
COPY models/ /app/models/

# Copy data artifacts (optional - may not exist in all environments)
COPY data/processed/ /app/data/processed/

# Copy MLflow database (optional)
COPY mlflow.db* /app/

# Create logs directory with proper permissions
RUN mkdir -p /app/logs /app/monitoring && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV MODEL_VERSION=v3_causal \
    MODEL_STAGE=production \
    DECISION_THRESHOLD=0.20 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check (increased start period)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
