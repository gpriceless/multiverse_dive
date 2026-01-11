# =============================================================================
# API Service Image for Multiverse Dive
# =============================================================================
# Full-featured API server with FastAPI and agent orchestration.
# Exposes REST endpoints for event analysis and pipeline management.
# =============================================================================

FROM multiverse-dive-base:latest AS api

LABEL maintainer="Multiverse Dive Team"
LABEL description="API service image with FastAPI and agent orchestration"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy all required modules
COPY --chown=mdive:mdive core/ /app/core/
COPY --chown=mdive:mdive openspec/ /app/openspec/
COPY --chown=mdive:mdive agents/ /app/agents/
COPY --chown=mdive:mdive api/ /app/api/

# Copy configuration files
COPY --chown=mdive:mdive examples/ /app/examples/

# Ensure proper permissions
RUN chown -R mdive:mdive /app

# Switch back to non-root user
USER mdive

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# API configuration
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_WORKERS=4
ENV LOG_LEVEL=info

# Expose the API port
EXPOSE 8000

# Healthcheck for API service
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Start the API server with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
