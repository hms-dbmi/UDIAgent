FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy application code and data
COPY src/ ./src/
COPY data/ ./data/

# Install the project itself
RUN uv sync --frozen

# Create logs directory and non-root user
RUN mkdir -p /app/logs && \
    addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app

ENV UV_CACHE_DIR=/tmp/uv-cache

USER app

EXPOSE 80

CMD ["uv", "run", "fastapi", "run", "./src/udi_api.py", "--port", "80", "--host", "0.0.0.0"]
