# =============================================================================
# Core Processing Image for Multiverse Dive
# =============================================================================
# Lightweight image for batch processing and analysis pipelines.
# Does not include API dependencies - optimized for compute workloads.
# =============================================================================

FROM multiverse-dive-base:latest AS core

LABEL maintainer="Multiverse Dive Team"
LABEL description="Core processing image for batch analysis"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy core processing modules
COPY --chown=mdive:mdive core/ /app/core/
COPY --chown=mdive:mdive openspec/ /app/openspec/

# Copy configuration and schema files
COPY --chown=mdive:mdive examples/ /app/examples/

# Ensure proper permissions
RUN chown -R mdive:mdive /app

# Switch back to non-root user
USER mdive

# Set Python path to include app directory
ENV PYTHONPATH=/app:${PYTHONPATH}

# No healthcheck needed for batch processing
HEALTHCHECK NONE

# Default command runs a simple validation
CMD ["python", "-c", "from core import *; from openspec import *; print('Core modules loaded successfully')"]
