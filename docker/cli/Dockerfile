# =============================================================================
# CLI Tools Image for Multiverse Dive
# =============================================================================
# Lightweight image for command-line interface operations.
# Provides the 'mdive' CLI tool for local analysis and pipeline execution.
# =============================================================================

FROM multiverse-dive-base:latest AS cli

LABEL maintainer="Multiverse Dive Team"
LABEL description="CLI tools image for multiverse-dive"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy CLI and core modules (minimal footprint)
COPY --chown=mdive:mdive cli/ /app/cli/
COPY --chown=mdive:mdive core/ /app/core/
COPY --chown=mdive:mdive openspec/ /app/openspec/

# Copy configuration files
COPY --chown=mdive:mdive examples/ /app/examples/

# Ensure proper permissions
RUN chown -R mdive:mdive /app

# Switch back to non-root user
USER mdive

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# No healthcheck for CLI tool
HEALTHCHECK NONE

# Set entrypoint to the CLI tool
# Users can run: docker run mdive-cli analyze --help
ENTRYPOINT ["python", "-m", "cli"]

# Default command shows help
CMD ["--help"]
