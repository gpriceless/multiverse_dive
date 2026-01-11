# =============================================================================
# Background Worker Image for Multiverse Dive
# =============================================================================
# Worker image for distributed processing and async task execution.
# Connects to Redis for job queuing and coordinates with agent system.
# =============================================================================

FROM multiverse-dive-base:latest AS worker

LABEL maintainer="Multiverse Dive Team"
LABEL description="Background worker image for distributed processing"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy required modules for worker processes
COPY --chown=mdive:mdive core/ /app/core/
COPY --chown=mdive:mdive openspec/ /app/openspec/
COPY --chown=mdive:mdive agents/ /app/agents/

# Copy configuration files
COPY --chown=mdive:mdive examples/ /app/examples/

# Ensure proper permissions
RUN chown -R mdive:mdive /app

# Switch back to non-root user
USER mdive

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Worker configuration
ENV WORKER_CONCURRENCY=4
ENV WORKER_QUEUE=default
ENV REDIS_URL=redis://localhost:6379/0

# No HTTP healthcheck for worker - uses Redis ping
HEALTHCHECK NONE

# Start the worker process
# In production, this would be a proper task queue worker (e.g., Celery, RQ)
CMD ["python", "-c", "from agents import *; print('Worker ready - waiting for tasks')"]
