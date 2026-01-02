# =============================================================================
# PRODUCTION DOCKERFILE - Clinical Trial Dropout Prediction API
# =============================================================================
# OPTIMIZED FOR AWS FREE TIER (t2.micro, 8GB storage)
# Target image size: < 600MB
# Trains model during build for sklearn version compatibility

# Stage 1: Builder + Model Training
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install MINIMAL production requirements
COPY requirements.prod.txt .
RUN pip install --no-cache-dir --user -r requirements.prod.txt

# Copy training script and train model (ensures sklearn compatibility)
COPY scripts/ ./scripts/
RUN python scripts/train_production_model.py

# =============================================================================
# Stage 2: Production Runtime (Minimal)
FROM python:3.10-slim

# Metadata
LABEL maintainer="Clinical Trial ML Team"
LABEL version="2.3.0-risk-consistency"
LABEL description="FastAPI service for clinical trial dropout prediction (AWS Free Tier)"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy trained model from builder
COPY --from=builder /build/models/production_model.pkl /app/models/

# Set PATH for user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p /app/logs /app/monitoring

# Copy application code (only what's needed)
COPY api/__init__.py api/config.py api/main.py api/prediction_logger.py api/risk_bands.py /app/api/

# Copy monitoring module (minimal)
COPY monitoring/__init__.py monitoring/prediction_monitor.py /app/monitoring/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV MODEL_PATH=models/production_model.pkl \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
