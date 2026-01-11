#!/bin/bash
# Multiverse Dive - Local Image Build Script
#
# Builds all Docker images locally for development and testing.
#
# Usage:
#   ./scripts/build-images.sh              # Build all images
#   ./scripts/build-images.sh --no-cache   # Build without cache
#   ./scripts/build-images.sh api          # Build specific image
#   ./scripts/build-images.sh --edge       # Build edge images only
#   ./scripts/build-images.sh --help       # Show help

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_PREFIX="${IMAGE_PREFIX:-multiverse-dive}"
VERSION="${VERSION:-latest}"
BUILD_ARGS=""
IMAGES_TO_BUILD=""
BUILD_EDGE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Multiverse Dive Docker Image Builder

Usage: $(basename "$0") [OPTIONS] [IMAGE...]

Options:
    --no-cache      Build images without using cache
    --edge          Build edge deployment images only
    --version VER   Tag images with specific version (default: latest)
    --push          Push images after building
    -h, --help      Show this help message

Images:
    base            Base image with core dependencies
    api             API server image
    worker          Background worker image
    edge-arm64      Lightweight ARM64 image
    edge-jetson     NVIDIA Jetson GPU image

Examples:
    $(basename "$0")                    # Build all standard images
    $(basename "$0") api worker         # Build only api and worker
    $(basename "$0") --edge             # Build edge images only
    $(basename "$0") --version v1.2.3   # Tag with version v1.2.3
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            BUILD_ARGS="--no-cache"
            shift
            ;;
        --edge)
            BUILD_EDGE=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --push)
            PUSH_AFTER_BUILD=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            IMAGES_TO_BUILD="$IMAGES_TO_BUILD $1"
            shift
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Build functions for each image
build_base() {
    log_info "Building base image..."
    docker build $BUILD_ARGS \
        -t "${IMAGE_PREFIX}/base:${VERSION}" \
        -f docker/base/Dockerfile \
        .
    log_info "Base image built: ${IMAGE_PREFIX}/base:${VERSION}"
}

build_api() {
    log_info "Building API image..."
    docker build $BUILD_ARGS \
        -t "${IMAGE_PREFIX}/api:${VERSION}" \
        -f docker/api/Dockerfile \
        --build-arg BASE_IMAGE="${IMAGE_PREFIX}/base:${VERSION}" \
        .
    log_info "API image built: ${IMAGE_PREFIX}/api:${VERSION}"
}

build_worker() {
    log_info "Building worker image..."
    docker build $BUILD_ARGS \
        -t "${IMAGE_PREFIX}/worker:${VERSION}" \
        -f docker/worker/Dockerfile \
        --build-arg BASE_IMAGE="${IMAGE_PREFIX}/base:${VERSION}" \
        .
    log_info "Worker image built: ${IMAGE_PREFIX}/worker:${VERSION}"
}

build_edge_arm64() {
    log_info "Building edge ARM64 image..."

    # Check if buildx is available for cross-platform builds
    if docker buildx version &> /dev/null; then
        docker buildx build $BUILD_ARGS \
            --platform linux/arm64 \
            -t "${IMAGE_PREFIX}/edge-arm64:${VERSION}" \
            -f deploy/edge/arm64/Dockerfile.lightweight \
            --load \
            .
    else
        log_warn "Docker buildx not available, building for native platform only"
        docker build $BUILD_ARGS \
            -t "${IMAGE_PREFIX}/edge-arm64:${VERSION}" \
            -f deploy/edge/arm64/Dockerfile.lightweight \
            .
    fi
    log_info "Edge ARM64 image built: ${IMAGE_PREFIX}/edge-arm64:${VERSION}"
}

build_edge_jetson() {
    log_info "Building edge Jetson image..."
    log_warn "Jetson images should ideally be built on Jetson hardware"

    docker build $BUILD_ARGS \
        -t "${IMAGE_PREFIX}/edge-jetson:${VERSION}" \
        -f deploy/edge/nvidia-jetson/Dockerfile.gpu \
        .
    log_info "Edge Jetson image built: ${IMAGE_PREFIX}/edge-jetson:${VERSION}"
}

# Check Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    exit 1
fi

log_info "Starting Multiverse Dive image build"
log_info "Project root: $PROJECT_ROOT"
log_info "Image prefix: $IMAGE_PREFIX"
log_info "Version: $VERSION"

# Determine which images to build
if [ "$BUILD_EDGE" = true ]; then
    # Build edge images only
    log_info "Building edge images only"
    build_edge_arm64
    build_edge_jetson
elif [ -n "$IMAGES_TO_BUILD" ]; then
    # Build specified images
    for image in $IMAGES_TO_BUILD; do
        case $image in
            base)
                build_base
                ;;
            api)
                build_api
                ;;
            worker)
                build_worker
                ;;
            edge-arm64)
                build_edge_arm64
                ;;
            edge-jetson)
                build_edge_jetson
                ;;
            *)
                log_error "Unknown image: $image"
                exit 1
                ;;
        esac
    done
else
    # Build all standard images
    log_info "Building all standard images"
    build_base
    build_api
    build_worker
fi

# Show built images
log_info "Build complete. Images:"
docker images --filter "reference=${IMAGE_PREFIX}/*:${VERSION}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Push if requested
if [ "$PUSH_AFTER_BUILD" = true ]; then
    log_info "Pushing images..."
    "$SCRIPT_DIR/push-images.sh" --version "$VERSION"
fi

log_info "Done!"
