# Multi-stage production build for withoutbg web application
# Build context should be the repository root: docker build -f apps/web/Dockerfile -t withoutbg:latest .

# ============================================================================
# Stage 1: Build Frontend (React + Vite)
# ============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app

# Copy package files
COPY apps/web/frontend/package.json apps/web/frontend/package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY apps/web/frontend/ ./

# Build the application
RUN npm run build

# ============================================================================
# Stage 2: Build Backend (Python + FastAPI)
# ============================================================================
FROM python:3.12-slim AS backend-builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy and install the core withoutbg package
COPY packages/python/pyproject.toml packages/python/uv.lock packages/python/README.md packages/python/
COPY packages/python/src/ packages/python/src/
RUN cd packages/python && uv sync --frozen --no-dev

# Copy and install backend dependencies
COPY apps/web/backend/pyproject.toml apps/web/backend/uv.lock apps/web/backend/README.md apps/web/backend/
RUN cd apps/web/backend && uv sync --frozen --no-dev

# ============================================================================
# Stage 3: Production Runtime
# ============================================================================
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the complete withoutbg package structure (for editable install)
COPY --from=backend-builder /app/packages/python /app/packages/python

# Copy virtual environment from backend builder
COPY --from=backend-builder /app/apps/web/backend/.venv /app/apps/web/backend/.venv

# Install uv in runtime stage for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Add both virtual environments to PATH
ENV PATH="/app/apps/web/backend/.venv/bin:/app/packages/python/.venv/bin:$PATH"

# Reinstall withoutbg from the local source to ensure correct version
RUN cd /app/packages/python && \
    /usr/local/bin/uv pip install --python /app/apps/web/backend/.venv/bin/python -e .

# Copy backend application code
COPY apps/web/backend/ ./backend/

# Download ONNX model checkpoints from Hugging Face
RUN mkdir -p /app/checkpoints && \
    curl -fL --retry 3 -o /app/checkpoints/depth_anything_v2_vits_slim.onnx \
        "https://huggingface.co/withoutbg/focus/resolve/main/depth_anything_v2_vits_slim.onnx" && \
    curl -fL --retry 3 -o /app/checkpoints/isnet.onnx \
        "https://huggingface.co/withoutbg/focus/resolve/main/isnet.onnx" && \
    curl -fL --retry 3 -o /app/checkpoints/focus_matting_1.0.0.onnx \
        "https://huggingface.co/withoutbg/focus/resolve/main/focus_matting_1.0.0.onnx" && \
    curl -fL --retry 3 -o /app/checkpoints/focus_refiner_1.0.0.onnx \
        "https://huggingface.co/withoutbg/focus/resolve/main/focus_refiner_1.0.0.onnx"

# Copy built frontend from frontend builder to static directory
COPY --from=frontend-builder /app/dist ./static

# Set environment variables for backend
ENV PYTHONUNBUFFERED=1
ENV WITHOUTBG_DEPTH_MODEL_PATH=/app/checkpoints/depth_anything_v2_vits_slim.onnx
ENV WITHOUTBG_ISNET_MODEL_PATH=/app/checkpoints/isnet.onnx
ENV WITHOUTBG_MATTING_MODEL_PATH=/app/checkpoints/focus_matting_1.0.0.onnx
ENV WITHOUTBG_REFINER_MODEL_PATH=/app/checkpoints/focus_refiner_1.0.0.onnx

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:80/api/health || exit 1

# Run uvicorn directly
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "80"]

