# =============================================================================
# PRODUCTION DOCKERFILE - Clinical Trial Dropout Prediction API
# =============================================================================
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.10-slim AS builder

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

# Create necessary directories
RUN mkdir -p /app/api /app/src /app/monitoring /app/models /app/data/processed /app/logs

# Copy application code (required)
COPY api/ /app/api/
COPY src/ /app/src/

# Copy everything else that might exist - using ADD with wildcards
# Note: If these don't exist, the build will fail. Use CI to ensure they exist.
COPY . /tmp/context/

# Copy optional files if they exist using shell
RUN cp -r /tmp/context/monitoring/*.py /app/monitoring/ 2>/dev/null || mkdir -p /app/monitoring && \
    cp /tmp/context/models/*.pkl /app/models/ 2>/dev/null || echo "No models to copy" && \
    cp /tmp/context/data/processed/*.pkl /app/data/processed/ 2>/dev/null || echo "No preprocessors to copy" && \
    cp /tmp/context/data/processed/*.csv /app/data/processed/ 2>/dev/null || echo "No processed data to copy" && \
    cp /tmp/context/mlflow.db /app/ 2>/dev/null || echo "No mlflow.db to copy" && \
    rm -rf /tmp/context

# Set ownership
RUN chown -R appuser:appuser /app

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
