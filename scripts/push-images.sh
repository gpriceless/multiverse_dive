#!/bin/bash
# Multiverse Dive - Image Push Script
#
# Pushes Docker images to container registries.
#
# Usage:
#   ./scripts/push-images.sh                        # Push all images to default registry
#   ./scripts/push-images.sh --version v1.2.3       # Push specific version
#   ./scripts/push-images.sh --registry ghcr.io     # Push to specific registry
#   ./scripts/push-images.sh api worker             # Push specific images only
#   ./scripts/push-images.sh --help                 # Show help

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_PREFIX="${IMAGE_PREFIX:-multiverse-dive}"
VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-ghcr.io}"
REGISTRY_NAMESPACE="${REGISTRY_NAMESPACE:-}"
IMAGES_TO_PUSH=""
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
Multiverse Dive Docker Image Push Script

Usage: $(basename "$0") [OPTIONS] [IMAGE...]

Options:
    --registry REG      Target registry (default: ghcr.io)
    --namespace NS      Registry namespace/organization
    --version VER       Image version tag (default: latest)
    --dry-run           Show what would be pushed without pushing
    -h, --help          Show this help message

Images:
    base            Base image with core dependencies
    api             API server image
    worker          Background worker image
    edge-arm64      Lightweight ARM64 image
    edge-jetson     NVIDIA Jetson GPU image

Environment Variables:
    REGISTRY            Default registry (overridden by --registry)
    REGISTRY_NAMESPACE  Registry namespace (overridden by --namespace)
    IMAGE_PREFIX        Local image prefix (default: multiverse-dive)
    VERSION             Image version (overridden by --version)

Examples:
    $(basename "$0")                                    # Push all to default registry
    $(basename "$0") --registry docker.io --namespace myorg
    $(basename "$0") api worker --version v1.2.3
    $(basename "$0") --dry-run                          # Preview push
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            REGISTRY_NAMESPACE="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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
            IMAGES_TO_PUSH="$IMAGES_TO_PUSH $1"
            shift
            ;;
    esac
done

# Build remote image name
get_remote_tag() {
    local image_name=$1
    if [ -n "$REGISTRY_NAMESPACE" ]; then
        echo "${REGISTRY}/${REGISTRY_NAMESPACE}/${image_name}:${VERSION}"
    else
        echo "${REGISTRY}/${image_name}:${VERSION}"
    fi
}

# Push a single image
push_image() {
    local local_tag="${IMAGE_PREFIX}/$1:${VERSION}"
    local remote_tag=$(get_remote_tag "$1")

    log_info "Pushing $1..."
    log_info "  Local:  $local_tag"
    log_info "  Remote: $remote_tag"

    if [ "$DRY_RUN" = true ]; then
        log_warn "  [DRY RUN] Would push $remote_tag"
        return 0
    fi

    # Check if local image exists
    if ! docker image inspect "$local_tag" &> /dev/null; then
        log_error "Local image not found: $local_tag"
        log_error "Run ./scripts/build-images.sh $1 first"
        return 1
    fi

    # Tag for remote registry
    docker tag "$local_tag" "$remote_tag"

    # Push to registry
    docker push "$remote_tag"

    log_info "  Successfully pushed $remote_tag"

    # Also push latest tag if this is a versioned release
    if [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        local latest_tag=$(get_remote_tag "$1" | sed "s/:${VERSION}/:latest/")
        log_info "  Also tagging as latest: $latest_tag"

        if [ "$DRY_RUN" = false ]; then
            docker tag "$local_tag" "$latest_tag"
            docker push "$latest_tag"
        fi
    fi
}

# Check Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if logged in to registry
check_registry_auth() {
    log_info "Checking registry authentication..."

    # Try to get credentials for the registry
    if ! docker info 2>/dev/null | grep -q "Registry"; then
        log_warn "Unable to verify registry authentication"
    fi

    # For GHCR, check if we can access
    if [[ "$REGISTRY" == "ghcr.io" ]]; then
        if ! echo "" | docker login ghcr.io --username _ --password-stdin 2>&1 | grep -q "Login Succeeded"; then
            log_warn "You may need to login: docker login ghcr.io"
        fi
    fi
}

log_info "Multiverse Dive Image Push"
log_info "Registry: $REGISTRY"
log_info "Version: $VERSION"
[ -n "$REGISTRY_NAMESPACE" ] && log_info "Namespace: $REGISTRY_NAMESPACE"
[ "$DRY_RUN" = true ] && log_warn "DRY RUN MODE - No images will be pushed"

# Default images to push
DEFAULT_IMAGES="base api worker"

# Determine which images to push
if [ -z "$IMAGES_TO_PUSH" ]; then
    IMAGES_TO_PUSH="$DEFAULT_IMAGES"
fi

# Track failures
FAILED_IMAGES=""

# Push each image
for image in $IMAGES_TO_PUSH; do
    if push_image "$image"; then
        log_info "  $image: OK"
    else
        FAILED_IMAGES="$FAILED_IMAGES $image"
        log_error "  $image: FAILED"
    fi
done

# Summary
echo ""
log_info "Push Summary"
log_info "============"

if [ -z "$FAILED_IMAGES" ]; then
    log_info "All images pushed successfully!"
else
    log_error "Failed images:$FAILED_IMAGES"
    exit 1
fi

# Show pushed images
if [ "$DRY_RUN" = false ]; then
    echo ""
    log_info "Pushed images:"
    for image in $IMAGES_TO_PUSH; do
        echo "  $(get_remote_tag "$image")"
    done
fi

log_info "Done!"
