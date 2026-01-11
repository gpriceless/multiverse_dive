# =============================================================================
# Base Image for Multiverse Dive
# =============================================================================
# Provides geospatial dependencies (GDAL, etc.) and Python environment.
# All other service images extend from this base.
# =============================================================================

FROM python:3.11-slim AS base

LABEL maintainer="Multiverse Dive Team"
LABEL description="Base image with geospatial dependencies for multiverse-dive"
LABEL version="0.1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for geospatial processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GDAL and geospatial libraries
    gdal-bin \
    libgdal-dev \
    # Proj for coordinate transformations
    proj-bin \
    libproj-dev \
    # GEOS for geometry operations
    libgeos-dev \
    # HDF5 for scientific data formats
    libhdf5-dev \
    # NetCDF for climate/weather data
    libnetcdf-dev \
    # Build tools for Python packages with C extensions
    build-essential \
    gcc \
    g++ \
    # Git for potential VCS-based dependencies
    git \
    # curl for healthchecks
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 mdive \
    && useradd --uid 1000 --gid mdive --shell /bin/bash --create-home mdive

# Set working directory
WORKDIR /app

# Copy dependency specification
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
# Using pip install . instead of -e for production images
RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# Create directories for data and cache
RUN mkdir -p /app/data /app/cache /app/output \
    && chown -R mdive:mdive /app

# Set default user (can be overridden in derived images)
USER mdive

# Default healthcheck (can be overridden)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# No default CMD - derived images should specify their own
