"""
Screenshot Generator for Image Validation.

Generates visual screenshots of validated raster images for debugging
and audit trails. Uses matplotlib for lightweight rendering without
browser dependencies.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.data.ingestion.validation.config import ValidationConfig

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False


class ScreenshotGenerator:
    """
    Generates screenshots of raster images for validation debugging.

    Features:
    - Individual band visualization
    - RGB/false-color composites
    - Metadata overlay
    - Validation status indicators

    Example:
        generator = ScreenshotGenerator(config)
        screenshot_path = generator.generate(
            raster_path=Path("/data/image.tif"),
            dataset_id="S2A_20250110",
            validation_result=result,
        )
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialize screenshot generator.

        Args:
            config: Validation configuration
        """
        self.config = config

        # Ensure output directory exists
        self.output_dir = config.screenshots.get_output_path()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        raster_path: Union[str, Path],
        dataset_id: str,
        validation_result: Optional[Any] = None,
        execution_id: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Generate screenshot of raster image.

        Args:
            raster_path: Path to raster file
            dataset_id: Dataset identifier
            validation_result: Optional ImageValidationResult
            execution_id: Optional execution identifier

        Returns:
            Path to generated screenshot, or None if generation failed
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not installed - cannot generate screenshots")
            return None

        if not HAS_RASTERIO:
            logger.warning("rasterio not installed - cannot generate screenshots")
            return None

        start_time = time.time()
        raster_path = Path(raster_path)

        try:
            # Generate output path
            output_path = self._get_output_path(dataset_id, execution_id)

            # Open raster and generate visualization
            with rasterio.open(raster_path) as dataset:
                fig = self._create_figure(dataset, dataset_id, validation_result)

                # Save figure
                fig.savefig(
                    output_path,
                    format=self.config.screenshots.format,
                    dpi=100,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )
                plt.close(fig)

            duration = time.time() - start_time
            logger.info(
                f"Generated screenshot for {dataset_id} at {output_path} ({duration:.2f}s)"
            )

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate screenshot for {dataset_id}: {e}")
            return None

    def _get_output_path(
        self,
        dataset_id: str,
        execution_id: Optional[str] = None,
    ) -> Path:
        """Generate output path for screenshot."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if execution_id:
            subdir = self.output_dir / execution_id
            subdir.mkdir(parents=True, exist_ok=True)
            filename = f"{dataset_id}_{timestamp}.{self.config.screenshots.format}"
            return subdir / filename
        else:
            filename = f"{dataset_id}_{timestamp}.{self.config.screenshots.format}"
            return self.output_dir / filename

    def _create_figure(
        self,
        dataset: Any,
        dataset_id: str,
        validation_result: Optional[Any],
    ) -> Any:
        """Create matplotlib figure with raster visualization."""
        n_bands = min(dataset.count, 6)  # Limit to 6 bands max

        # Determine figure layout
        if n_bands <= 1:
            fig_width, fig_height = 8, 8
            n_cols, n_rows = 1, 1
        elif n_bands <= 3:
            fig_width, fig_height = 12, 6
            n_cols, n_rows = n_bands, 1
        else:
            fig_width, fig_height = 12, 10
            n_cols, n_rows = 3, 2

        # Apply resolution constraint
        width, height = self.config.screenshots.resolution
        aspect = width / height
        if fig_width / fig_height > aspect:
            fig_height = fig_width / aspect
        else:
            fig_width = fig_height * aspect

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create grid
        gs = GridSpec(
            n_rows + 1, n_cols, figure=fig, height_ratios=[0.1] + [1.0] * n_rows
        )

        # Add title with validation status
        title_ax = fig.add_subplot(gs[0, :])
        self._add_title(title_ax, dataset_id, validation_result)

        # Add band visualizations
        for idx in range(n_bands):
            row = idx // n_cols + 1
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            self._plot_band(ax, dataset, idx + 1, validation_result)

        # Adjust layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    def _add_title(
        self,
        ax: Any,
        dataset_id: str,
        validation_result: Optional[Any],
    ) -> None:
        """Add title with validation status."""
        ax.axis("off")

        # Determine status
        if validation_result is not None:
            if validation_result.is_valid:
                status = "PASSED"
                color = "green"
            else:
                status = "FAILED"
                color = "red"
            title = f"{dataset_id} - Validation: {status}"
        else:
            title = dataset_id
            color = "black"

        ax.text(
            0.5,
            0.5,
            title,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )

        # Add timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        ax.text(
            0.99,
            0.5,
            timestamp,
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
            ha="right",
            va="center",
        )

    def _plot_band(
        self,
        ax: Any,
        dataset: Any,
        band_index: int,
        validation_result: Optional[Any],
    ) -> None:
        """Plot a single band."""
        try:
            # Read band data (subsample for large images)
            data = self._read_subsampled_band(dataset, band_index)

            # Get band name
            band_name = f"Band {band_index}"
            if dataset.descriptions and len(dataset.descriptions) >= band_index:
                desc = dataset.descriptions[band_index - 1]
                if desc:
                    band_name = desc

            # Handle NoData
            nodata = dataset.nodata
            if nodata is not None:
                data = np.ma.masked_equal(data, nodata)

            # Determine valid range for visualization
            valid_data = data.compressed() if hasattr(data, "compressed") else data.flatten()
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 2)
                vmax = np.percentile(valid_data, 98)
            else:
                vmin, vmax = 0, 1

            # Plot
            im = ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Get validation status for this band
            band_status = ""
            if validation_result and hasattr(validation_result, "band_results"):
                # Try to find matching band result
                for name, result in validation_result.band_results.items():
                    if result.band_index == band_index:
                        if result.is_valid:
                            band_status = " [OK]"
                        else:
                            band_status = " [FAIL]"
                        break

            # Set title with status
            title_color = "green" if "[OK]" in band_status else ("red" if "[FAIL]" in band_status else "black")
            ax.set_title(f"{band_name}{band_status}", fontsize=10, color=title_color)
            ax.axis("off")

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading band {band_index}:\n{str(e)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="red",
            )
            ax.axis("off")

    def _read_subsampled_band(
        self,
        dataset: Any,
        band_index: int,
        max_pixels: int = 2000,
    ) -> np.ndarray:
        """
        Read band with subsampling for large images.

        Args:
            dataset: Open rasterio dataset
            band_index: 1-indexed band number
            max_pixels: Maximum pixels per dimension

        Returns:
            Subsampled band data
        """
        height, width = dataset.height, dataset.width

        # Calculate subsample factor
        factor = max(1, max(height, width) // max_pixels)

        if factor == 1:
            return dataset.read(band_index)

        # Read with overview/decimation
        out_shape = (height // factor, width // factor)

        data = dataset.read(
            band_index,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.average,
        )

        return data

    def generate_rgb_composite(
        self,
        raster_path: Union[str, Path],
        dataset_id: str,
        band_indices: Tuple[int, int, int] = (4, 3, 2),
        execution_id: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Generate RGB composite screenshot.

        Args:
            raster_path: Path to raster file
            dataset_id: Dataset identifier
            band_indices: Band indices for R, G, B channels
            execution_id: Optional execution identifier

        Returns:
            Path to generated screenshot
        """
        if not HAS_MATPLOTLIB or not HAS_RASTERIO:
            return None

        try:
            output_path = self._get_output_path(f"{dataset_id}_rgb", execution_id)

            with rasterio.open(raster_path) as dataset:
                # Read RGB bands
                rgb_data = []
                for band_idx in band_indices:
                    if band_idx <= dataset.count:
                        data = self._read_subsampled_band(dataset, band_idx)
                        rgb_data.append(data)
                    else:
                        # Pad with zeros if band doesn't exist
                        shape = (dataset.height // 2, dataset.width // 2)
                        rgb_data.append(np.zeros(shape))

                # Stack and normalize
                rgb = np.stack(rgb_data, axis=-1).astype(np.float32)

                # Percentile stretch
                for i in range(3):
                    channel = rgb[:, :, i]
                    valid = channel[channel > 0]
                    if len(valid) > 0:
                        vmin = np.percentile(valid, 2)
                        vmax = np.percentile(valid, 98)
                        if vmax > vmin:
                            rgb[:, :, i] = np.clip((channel - vmin) / (vmax - vmin), 0, 1)

                # Create figure
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(rgb)
                ax.set_title(f"{dataset_id} - RGB Composite", fontsize=12)
                ax.axis("off")

                fig.savefig(
                    output_path,
                    format=self.config.screenshots.format,
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close(fig)

                return output_path

        except Exception as e:
            logger.error(f"Failed to generate RGB composite: {e}")
            return None

    def cleanup_old_screenshots(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """
        Clean up old temporary screenshots.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of files deleted
        """
        if not self.output_dir.exists():
            return 0

        deleted = 0
        now = datetime.now(timezone.utc)
        max_age_seconds = max_age_hours * 3600

        for file_path in self.output_dir.rglob(f"*.{self.config.screenshots.format}"):
            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                age_seconds = (now - mtime).total_seconds()

                if age_seconds > max_age_seconds:
                    file_path.unlink()
                    deleted += 1
                    logger.debug(f"Deleted old screenshot: {file_path}")

            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old screenshots")

        return deleted
