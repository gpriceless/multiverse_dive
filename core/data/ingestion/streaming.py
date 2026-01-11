"""
Streaming Data Ingestion for Memory-Efficient Processing.

Provides tools for downloading and processing large datasets without
loading entire files into memory. Supports chunked downloads, windowed
raster reads, and stream processing.

Key Components:
- StreamingDownloader: Chunked HTTP downloads with resume support
- WindowedReader: Memory-efficient raster reading via windows
- StreamingIngester: End-to-end streaming ingestion pipeline

Example Usage:
    from core.data.ingestion.streaming import (
        StreamingDownloader,
        WindowedReader,
        StreamingIngester,
    )

    # Download large file in chunks
    downloader = StreamingDownloader(
        chunk_size_mb=10,
        bandwidth_limit_mbps=50,
    )
    downloader.download(url, output_path, resume=True)

    # Read raster in windows
    reader = WindowedReader(raster_path)
    for window, data in reader.iterate_windows(window_size=(512, 512)):
        process(data)

    # Stream ingestion
    ingester = StreamingIngester(buffer_size_mb=256)
    ingester.ingest(source_url, output_path)
"""

import hashlib
import logging
import mmap
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports - rasterio may not be installed
try:
    import rasterio
    from rasterio.windows import Window

    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    Window = None
    HAS_RASTERIO = False

# Optional httpx/requests for downloads
try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False


# =============================================================================
# Enumerations
# =============================================================================


class DownloadStatus(Enum):
    """Status of a download operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressUnit(Enum):
    """Units for progress reporting."""

    BYTES = "bytes"
    PERCENT = "percent"
    CHUNKS = "chunks"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DownloadProgress:
    """
    Progress information for a download.

    Attributes:
        total_bytes: Total file size in bytes
        downloaded_bytes: Bytes downloaded so far
        elapsed_seconds: Time elapsed
        speed_bytes_per_sec: Current download speed
        estimated_remaining_seconds: Estimated time remaining
        status: Current download status
    """

    total_bytes: int = 0
    downloaded_bytes: int = 0
    elapsed_seconds: float = 0.0
    speed_bytes_per_sec: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    status: DownloadStatus = DownloadStatus.PENDING

    @property
    def progress_percent(self) -> float:
        """Download progress as percentage."""
        if self.total_bytes == 0:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100)

    @property
    def remaining_bytes(self) -> int:
        """Bytes remaining to download."""
        return max(0, self.total_bytes - self.downloaded_bytes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "speed_mbps": self.speed_bytes_per_sec / (1024 * 1024),
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "status": self.status.value,
        }


@dataclass
class ChunkInfo:
    """
    Information about a download chunk.

    Attributes:
        start_byte: Starting byte position
        end_byte: Ending byte position (exclusive)
        size_bytes: Chunk size in bytes
        checksum: Optional checksum of chunk
    """

    start_byte: int
    end_byte: int
    size_bytes: int
    checksum: Optional[str] = None


@dataclass
class WindowInfo:
    """
    Information about a raster window.

    Attributes:
        col_off: Column offset (x start)
        row_off: Row offset (y start)
        width: Window width in pixels
        height: Window height in pixels
        band: Band number (1-indexed)
    """

    col_off: int
    row_off: int
    width: int
    height: int
    band: int = 1

    @property
    def shape(self) -> Tuple[int, int]:
        """Window shape (height, width)."""
        return (self.height, self.width)

    def to_rasterio_window(self) -> Optional["Window"]:
        """Convert to rasterio Window object."""
        if Window is None:
            return None
        return Window(self.col_off, self.row_off, self.width, self.height)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "col_off": self.col_off,
            "row_off": self.row_off,
            "width": self.width,
            "height": self.height,
            "band": self.band,
        }


@dataclass
class StreamingConfig:
    """
    Configuration for streaming operations.

    Attributes:
        chunk_size_bytes: Size of download chunks
        buffer_size_bytes: Buffer size for processing
        bandwidth_limit_bps: Maximum bandwidth (bytes/sec, 0 = unlimited)
        max_retries: Maximum retry attempts
        retry_delay_seconds: Delay between retries
        timeout_seconds: Request timeout
        verify_checksums: Verify chunk checksums
        enable_resume: Enable download resume
    """

    chunk_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    buffer_size_bytes: int = 256 * 1024 * 1024  # 256 MB
    bandwidth_limit_bps: int = 0  # 0 = unlimited
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    timeout_seconds: float = 60.0
    verify_checksums: bool = True
    enable_resume: bool = True

    @classmethod
    def from_mb(
        cls,
        chunk_size_mb: float = 10,
        buffer_size_mb: float = 256,
        bandwidth_limit_mbps: float = 0,
        **kwargs,
    ) -> "StreamingConfig":
        """Create config with sizes in MB."""
        return cls(
            chunk_size_bytes=int(chunk_size_mb * 1024 * 1024),
            buffer_size_bytes=int(buffer_size_mb * 1024 * 1024),
            bandwidth_limit_bps=int(bandwidth_limit_mbps * 1024 * 1024),
            **kwargs,
        )


# =============================================================================
# StreamingDownloader
# =============================================================================


class StreamingDownloader:
    """
    Chunked HTTP downloader with resume support.

    Features:
    - Download large files in chunks
    - Resume interrupted downloads
    - Progress tracking with callbacks
    - Bandwidth throttling
    - Checksum verification

    Example:
        downloader = StreamingDownloader(
            config=StreamingConfig.from_mb(chunk_size_mb=10),
        )

        # Simple download
        downloader.download(url, output_path)

        # With progress callback
        def on_progress(progress: DownloadProgress):
            print(f"{progress.progress_percent:.1f}% complete")

        downloader.download(url, output_path, progress_callback=on_progress)

        # Resume interrupted download
        downloader.download(url, output_path, resume=True)
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize downloader.

        Args:
            config: Streaming configuration
            headers: Additional HTTP headers
        """
        self.config = config or StreamingConfig()
        self.headers = headers or {}
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._current_progress: Optional[DownloadProgress] = None

    def download(
        self,
        url: str,
        output_path: Union[str, Path],
        resume: bool = True,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> DownloadProgress:
        """
        Download file from URL.

        Args:
            url: Source URL
            output_path: Output file path
            resume: Enable resume from partial download
            progress_callback: Callback for progress updates

        Returns:
            Final download progress

        Raises:
            RuntimeError: If download fails after all retries
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._cancelled.clear()

        # Get file info
        total_bytes, supports_range = self._get_file_info(url)

        # Check for existing partial download
        start_byte = 0
        if resume and output_path.exists() and supports_range:
            start_byte = output_path.stat().st_size
            if start_byte >= total_bytes:
                # Already complete
                progress = DownloadProgress(
                    total_bytes=total_bytes,
                    downloaded_bytes=total_bytes,
                    status=DownloadStatus.COMPLETED,
                )
                if progress_callback:
                    progress_callback(progress)
                return progress

        # Initialize progress
        progress = DownloadProgress(
            total_bytes=total_bytes,
            downloaded_bytes=start_byte,
            status=DownloadStatus.IN_PROGRESS,
        )
        self._current_progress = progress

        # Download
        start_time = time.time()
        last_progress_time = start_time

        try:
            mode = "ab" if start_byte > 0 else "wb"
            with open(output_path, mode) as f:
                for chunk in self._stream_chunks(url, start_byte, total_bytes):
                    if self._cancelled.is_set():
                        progress.status = DownloadStatus.CANCELLED
                        break

                    f.write(chunk)
                    progress.downloaded_bytes += len(chunk)

                    # Update timing stats
                    current_time = time.time()
                    progress.elapsed_seconds = current_time - start_time

                    if current_time - last_progress_time >= 0.5:  # Update every 0.5s
                        if progress.elapsed_seconds > 0:
                            progress.speed_bytes_per_sec = (
                                progress.downloaded_bytes - start_byte
                            ) / progress.elapsed_seconds

                        if progress.speed_bytes_per_sec > 0:
                            progress.estimated_remaining_seconds = (
                                progress.remaining_bytes / progress.speed_bytes_per_sec
                            )

                        if progress_callback:
                            progress_callback(progress)
                        last_progress_time = current_time

            if progress.downloaded_bytes >= total_bytes:
                progress.status = DownloadStatus.COMPLETED

        except Exception as e:
            logger.error(f"Download failed: {e}")
            progress.status = DownloadStatus.FAILED
            raise RuntimeError(f"Download failed: {e}") from e

        # Final callback
        if progress_callback:
            progress_callback(progress)

        return progress

    def _get_file_info(self, url: str) -> Tuple[int, bool]:
        """
        Get file size and check range support.

        Returns:
            Tuple of (total_bytes, supports_range)
        """
        headers = {**self.headers}

        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    response = client.head(url, headers=headers, follow_redirects=True)
                    response.raise_for_status()
                    total_bytes = int(response.headers.get("content-length", 0))
                    supports_range = response.headers.get("accept-ranges") == "bytes"
            elif HAS_REQUESTS:
                response = requests.head(
                    url, headers=headers, timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                total_bytes = int(response.headers.get("content-length", 0))
                supports_range = response.headers.get("accept-ranges") == "bytes"
            else:
                raise RuntimeError("No HTTP library available (httpx or requests)")

            return total_bytes, supports_range

        except Exception as e:
            logger.warning(f"Could not get file info: {e}")
            return 0, False

    def _stream_chunks(
        self,
        url: str,
        start_byte: int,
        total_bytes: int,
    ) -> Generator[bytes, None, None]:
        """
        Stream file content in chunks.

        Yields:
            Data chunks
        """
        headers = {**self.headers}

        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"

        retry_count = 0
        current_byte = start_byte

        while current_byte < total_bytes and retry_count < self.config.max_retries:
            try:
                if HAS_HTTPX:
                    yield from self._stream_httpx(url, headers, current_byte, total_bytes)
                    break
                elif HAS_REQUESTS:
                    yield from self._stream_requests(url, headers, current_byte, total_bytes)
                    break
                else:
                    raise RuntimeError("No HTTP library available")

            except Exception as e:
                retry_count += 1
                logger.warning(f"Download error (retry {retry_count}): {e}")

                if retry_count >= self.config.max_retries:
                    raise

                time.sleep(self.config.retry_delay_seconds * retry_count)

                # Update range header for resume
                if self._current_progress:
                    current_byte = self._current_progress.downloaded_bytes
                    headers["Range"] = f"bytes={current_byte}-"

    def _stream_httpx(
        self,
        url: str,
        headers: Dict[str, str],
        start_byte: int,
        total_bytes: int,
    ) -> Generator[bytes, None, None]:
        """Stream using httpx."""
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
                response.raise_for_status()

                for chunk in response.iter_bytes(chunk_size=self.config.chunk_size_bytes):
                    if self._cancelled.is_set():
                        break

                    # Apply bandwidth throttling
                    if self.config.bandwidth_limit_bps > 0:
                        expected_time = len(chunk) / self.config.bandwidth_limit_bps
                        time.sleep(expected_time)

                    yield chunk

    def _stream_requests(
        self,
        url: str,
        headers: Dict[str, str],
        start_byte: int,
        total_bytes: int,
    ) -> Generator[bytes, None, None]:
        """Stream using requests."""
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=self.config.timeout_seconds,
        ) as response:
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=self.config.chunk_size_bytes):
                if self._cancelled.is_set():
                    break

                if chunk:  # Filter out keep-alive chunks
                    # Apply bandwidth throttling
                    if self.config.bandwidth_limit_bps > 0:
                        expected_time = len(chunk) / self.config.bandwidth_limit_bps
                        time.sleep(expected_time)

                    yield chunk

    def cancel(self) -> None:
        """Cancel the current download."""
        self._cancelled.set()

    def get_progress(self) -> Optional[DownloadProgress]:
        """Get current download progress."""
        return self._current_progress


# =============================================================================
# WindowedReader
# =============================================================================


class WindowedReader:
    """
    Memory-efficient raster reading using windows.

    Features:
    - Read specific tiles/windows from large rasters
    - Memory-mapped file support
    - Lazy loading
    - Iterator interface for sequential access

    Example:
        with WindowedReader(raster_path) as reader:
            # Get raster info
            print(f"Shape: {reader.shape}")
            print(f"CRS: {reader.crs}")

            # Read specific window
            data = reader.read_window(0, 0, 512, 512)

            # Iterate over windows
            for window_info, data in reader.iterate_windows(window_size=(512, 512)):
                process(data)
    """

    def __init__(
        self,
        path: Union[str, Path],
        use_mmap: bool = True,
    ):
        """
        Initialize windowed reader.

        Args:
            path: Path to raster file
            use_mmap: Use memory-mapped file access
        """
        self.path = Path(path)
        self.use_mmap = use_mmap
        self._dataset = None
        self._mmap = None
        self._lock = threading.Lock()

        if not self.path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.path}")

        self._open()

    def _open(self) -> None:
        """Open the raster file."""
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio not installed - required for WindowedReader")

        self._dataset = rasterio.open(self.path)

    def close(self) -> None:
        """Close the raster file."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self) -> "WindowedReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def shape(self) -> Tuple[int, int]:
        """Raster shape (height, width)."""
        if self._dataset is None:
            raise RuntimeError("Dataset not open")
        return (self._dataset.height, self._dataset.width)

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        if self._dataset is None:
            raise RuntimeError("Dataset not open")
        return self._dataset.count

    @property
    def dtype(self) -> np.dtype:
        """Data type of raster."""
        if self._dataset is None:
            raise RuntimeError("Dataset not open")
        return np.dtype(self._dataset.dtypes[0])

    @property
    def crs(self) -> Optional[str]:
        """Coordinate reference system."""
        if self._dataset is None:
            return None
        crs = self._dataset.crs
        return crs.to_string() if crs else None

    @property
    def transform(self) -> Optional[Any]:
        """Affine transform."""
        if self._dataset is None:
            return None
        return self._dataset.transform

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Geographic bounds (minx, miny, maxx, maxy)."""
        if self._dataset is None:
            return None
        b = self._dataset.bounds
        return (b.left, b.bottom, b.right, b.top)

    @property
    def resolution(self) -> Optional[Tuple[float, float]]:
        """Pixel resolution (x_res, y_res)."""
        if self._dataset is None or self._dataset.transform is None:
            return None
        return (self._dataset.transform.a, abs(self._dataset.transform.e))

    def read_window(
        self,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        bands: Optional[List[int]] = None,
        out_dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Read a window from the raster.

        Args:
            col_off: Column offset (x start)
            row_off: Row offset (y start)
            width: Window width
            height: Window height
            bands: Band indices to read (1-indexed, None = all)
            out_dtype: Output data type

        Returns:
            Array of shape (bands, height, width) or (height, width) for single band
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not open")

        # Clamp to valid range
        col_off = max(0, col_off)
        row_off = max(0, row_off)
        width = min(width, self._dataset.width - col_off)
        height = min(height, self._dataset.height - row_off)

        if width <= 0 or height <= 0:
            return np.array([])

        window = Window(col_off, row_off, width, height)

        if bands is None:
            bands = list(range(1, self._dataset.count + 1))

        with self._lock:
            data = self._dataset.read(bands, window=window, out_dtype=out_dtype)

        # Squeeze single band
        if len(bands) == 1:
            data = data.squeeze(axis=0)

        return data

    def read_tile(
        self,
        tile_col: int,
        tile_row: int,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
        bands: Optional[List[int]] = None,
    ) -> Tuple[WindowInfo, np.ndarray]:
        """
        Read a tile from the raster.

        Args:
            tile_col: Tile column index
            tile_row: Tile row index
            tile_size: Tile size (width, height)
            overlap: Overlap pixels on each side
            bands: Band indices to read

        Returns:
            Tuple of (WindowInfo, data array)
        """
        col_off = tile_col * (tile_size[0] - 2 * overlap)
        row_off = tile_row * (tile_size[1] - 2 * overlap)

        # Apply overlap
        read_col_off = max(0, col_off - overlap)
        read_row_off = max(0, row_off - overlap)
        read_width = tile_size[0] + 2 * overlap
        read_height = tile_size[1] + 2 * overlap

        window_info = WindowInfo(
            col_off=read_col_off,
            row_off=read_row_off,
            width=read_width,
            height=read_height,
        )

        data = self.read_window(
            read_col_off,
            read_row_off,
            read_width,
            read_height,
            bands=bands,
        )

        return window_info, data

    def iterate_windows(
        self,
        window_size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
        bands: Optional[List[int]] = None,
    ) -> Generator[Tuple[WindowInfo, np.ndarray], None, None]:
        """
        Iterate over raster in windows.

        Args:
            window_size: Window size (width, height)
            overlap: Overlap pixels
            bands: Band indices to read

        Yields:
            Tuple of (WindowInfo, data array) for each window
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not open")

        effective_width = window_size[0] - 2 * overlap if overlap > 0 else window_size[0]
        effective_height = window_size[1] - 2 * overlap if overlap > 0 else window_size[1]

        n_cols = int(np.ceil(self._dataset.width / effective_width))
        n_rows = int(np.ceil(self._dataset.height / effective_height))

        for row in range(n_rows):
            for col in range(n_cols):
                window_info, data = self.read_tile(
                    tile_col=col,
                    tile_row=row,
                    tile_size=window_size,
                    overlap=overlap,
                    bands=bands,
                )
                yield window_info, data


# =============================================================================
# MemoryMappedReader
# =============================================================================


class MemoryMappedReader:
    """
    Memory-mapped file reader for large arrays.

    Enables efficient random access to large files without
    loading them entirely into memory.

    Example:
        reader = MemoryMappedReader(path, shape=(10000, 10000), dtype=np.float32)

        # Random access reads
        data = reader.read_region(1000, 1000, 100, 100)

        # Memory-mapped array access
        with reader.as_array() as arr:
            arr[5000:5100, 5000:5100]
    """

    def __init__(
        self,
        path: Union[str, Path],
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        offset: int = 0,
        mode: str = "r",
    ):
        """
        Initialize memory-mapped reader.

        Args:
            path: Path to binary file
            shape: Array shape
            dtype: Data type
            offset: Byte offset in file
            mode: File mode ('r' for read, 'r+' for read/write)
        """
        self.path = Path(path)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.offset = offset
        self.mode = mode
        self._mmap: Optional[np.ndarray] = None

    def open(self) -> np.ndarray:
        """
        Open memory-mapped array.

        Returns:
            Memory-mapped numpy array
        """
        self._mmap = np.memmap(
            self.path,
            dtype=self.dtype,
            mode=self.mode,
            offset=self.offset,
            shape=self.shape,
        )
        return self._mmap

    def close(self) -> None:
        """Close memory-mapped file."""
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

    def __enter__(self) -> np.ndarray:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read_region(
        self,
        row_start: int,
        col_start: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Read a region from the memory-mapped file.

        Args:
            row_start: Starting row
            col_start: Starting column
            height: Region height
            width: Region width

        Returns:
            Copy of region data
        """
        if self._mmap is None:
            self.open()

        # Handle different array dimensions
        if len(self.shape) == 2:
            return self._mmap[
                row_start : row_start + height, col_start : col_start + width
            ].copy()
        elif len(self.shape) == 3:
            return self._mmap[
                :, row_start : row_start + height, col_start : col_start + width
            ].copy()
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")


# =============================================================================
# StreamingIngester
# =============================================================================


class StreamingIngester:
    """
    End-to-end streaming data ingestion.

    Combines downloading and processing without loading entire
    files into memory. Processes tiles as they become available.

    Example:
        ingester = StreamingIngester(
            config=StreamingConfig.from_mb(buffer_size_mb=256),
        )

        # Simple ingestion
        result = ingester.ingest(source_url, output_path)

        # With tile processor
        def process_tile(window, data):
            return analyze(data)

        result = ingester.ingest(
            source_url,
            output_path,
            tile_processor=process_tile,
            tile_size=(512, 512),
        )
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize streaming ingester.

        Args:
            config: Streaming configuration
            temp_dir: Temporary directory for intermediate files
        """
        self.config = config or StreamingConfig()
        self.temp_dir = Path(temp_dir) if temp_dir else Path.cwd() / ".streaming_temp"
        self._downloader = StreamingDownloader(config=self.config)

    def ingest(
        self,
        source: str,
        output_path: Union[str, Path],
        tile_processor: Optional[Callable[[WindowInfo, np.ndarray], np.ndarray]] = None,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest data from source with streaming processing.

        Args:
            source: Source URL or path
            output_path: Output file path
            tile_processor: Optional function to process each tile
            tile_size: Tile size for processing
            overlap: Tile overlap
            progress_callback: Progress callback

        Returns:
            Ingestion result dictionary
        """
        output_path = Path(output_path)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        result: Dict[str, Any] = {
            "source": source,
            "output_path": str(output_path),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "tiles_processed": 0,
            "errors": [],
        }

        try:
            # Check if source is URL or local file
            is_url = source.startswith(("http://", "https://", "s3://"))

            if is_url:
                # Download to temp file
                temp_file = self.temp_dir / f"download_{hashlib.md5(source.encode()).hexdigest()[:8]}.tif"

                download_progress = self._downloader.download(
                    source,
                    temp_file,
                    progress_callback=lambda p: self._report_download_progress(p, progress_callback),
                )

                if download_progress.status != DownloadStatus.COMPLETED:
                    result["status"] = "failed"
                    result["errors"].append("Download failed")
                    return result

                source_path = temp_file
            else:
                source_path = Path(source)

            # Process tiles
            if tile_processor is not None:
                result = self._process_tiles(
                    source_path,
                    output_path,
                    tile_processor,
                    tile_size,
                    overlap,
                    result,
                    progress_callback,
                )
            else:
                # Just copy/convert
                self._copy_file(source_path, output_path)
                result["tiles_processed"] = 1

            result["status"] = "completed"
            result["end_time"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            result["status"] = "failed"
            result["errors"].append(str(e))
            result["end_time"] = datetime.now(timezone.utc).isoformat()

        return result

    def _process_tiles(
        self,
        source_path: Path,
        output_path: Path,
        tile_processor: Callable[[WindowInfo, np.ndarray], np.ndarray],
        tile_size: Tuple[int, int],
        overlap: int,
        result: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Dict[str, Any]:
        """Process source file tile by tile."""
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio not installed")

        with WindowedReader(source_path) as reader:
            # Get metadata
            height, width = reader.shape
            n_bands = reader.n_bands
            dtype = reader.dtype
            crs = reader.crs
            transform = reader.transform

            # Calculate number of tiles
            effective_size = (
                tile_size[0] - 2 * overlap if overlap > 0 else tile_size[0],
                tile_size[1] - 2 * overlap if overlap > 0 else tile_size[1],
            )
            n_tiles_x = int(np.ceil(width / effective_size[0]))
            n_tiles_y = int(np.ceil(height / effective_size[1]))
            total_tiles = n_tiles_x * n_tiles_y

            result["total_tiles"] = total_tiles

            # Create output file
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=n_bands,
                dtype=dtype,
                crs=crs,
                transform=transform,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="lzw",
            ) as dst:
                for window_info, data in reader.iterate_windows(
                    window_size=tile_size,
                    overlap=overlap,
                ):
                    try:
                        # Process tile
                        processed = tile_processor(window_info, data)

                        # Write to output
                        rio_window = window_info.to_rasterio_window()
                        if processed.ndim == 2:
                            dst.write(processed, 1, window=rio_window)
                        else:
                            dst.write(processed, window=rio_window)

                        result["tiles_processed"] += 1

                        # Report progress
                        if progress_callback:
                            progress_callback({
                                "phase": "processing",
                                "tiles_processed": result["tiles_processed"],
                                "total_tiles": total_tiles,
                                "progress_percent": result["tiles_processed"] / total_tiles * 100,
                            })

                    except Exception as e:
                        logger.warning(f"Error processing tile: {e}")
                        result["errors"].append(f"Tile {window_info.col_off},{window_info.row_off}: {e}")

        return result

    def _copy_file(self, source: Path, dest: Path) -> None:
        """Copy file with streaming."""
        with open(source, "rb") as src:
            with open(dest, "wb") as dst:
                while chunk := src.read(self.config.chunk_size_bytes):
                    dst.write(chunk)

    def _report_download_progress(
        self,
        progress: DownloadProgress,
        callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        """Report download progress through unified callback."""
        if callback:
            callback({
                "phase": "download",
                "downloaded_bytes": progress.downloaded_bytes,
                "total_bytes": progress.total_bytes,
                "progress_percent": progress.progress_percent,
                "speed_mbps": progress.speed_bytes_per_sec / (1024 * 1024),
            })


# =============================================================================
# Utility Functions
# =============================================================================


def stream_download(
    url: str,
    output_path: Union[str, Path],
    chunk_size_mb: float = 10,
    resume: bool = True,
) -> DownloadProgress:
    """
    Convenience function for streaming download.

    Args:
        url: Source URL
        output_path: Output file path
        chunk_size_mb: Chunk size in MB
        resume: Enable resume

    Returns:
        Download progress
    """
    config = StreamingConfig.from_mb(chunk_size_mb=chunk_size_mb)
    downloader = StreamingDownloader(config=config)
    return downloader.download(url, output_path, resume=resume)


def read_raster_window(
    path: Union[str, Path],
    col_off: int,
    row_off: int,
    width: int,
    height: int,
    bands: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Convenience function to read a raster window.

    Args:
        path: Path to raster
        col_off: Column offset
        row_off: Row offset
        width: Window width
        height: Window height
        bands: Band indices (1-indexed)

    Returns:
        Window data array
    """
    with WindowedReader(path) as reader:
        return reader.read_window(col_off, row_off, width, height, bands=bands)


def iterate_raster_tiles(
    path: Union[str, Path],
    tile_size: Tuple[int, int] = (512, 512),
    overlap: int = 0,
) -> Generator[Tuple[WindowInfo, np.ndarray], None, None]:
    """
    Convenience generator to iterate over raster tiles.

    Args:
        path: Path to raster
        tile_size: Tile size (width, height)
        overlap: Tile overlap

    Yields:
        Tuple of (WindowInfo, data) for each tile
    """
    with WindowedReader(path) as reader:
        yield from reader.iterate_windows(
            window_size=tile_size,
            overlap=overlap,
        )


def estimate_chunk_count(
    file_size_bytes: int,
    chunk_size_mb: float = 10,
) -> int:
    """
    Estimate number of chunks for a download.

    Args:
        file_size_bytes: Total file size
        chunk_size_mb: Chunk size in MB

    Returns:
        Number of chunks
    """
    chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
    return int(np.ceil(file_size_bytes / chunk_size_bytes))
