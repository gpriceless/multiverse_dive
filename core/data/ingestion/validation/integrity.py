"""
Data Integrity Validation Module.

Provides comprehensive integrity checks for geospatial data:
- File format validation (COG, Zarr, GeoParquet)
- Header consistency verification
- Checksum computation and verification
- CRS and georeferencing validation
- Data type and range validation
- File structure validation (tiling, chunks, blocks)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class IntegrityCheckType(Enum):
    """Types of integrity checks."""

    FORMAT = "format"
    HEADER = "header"
    CHECKSUM = "checksum"
    CRS = "crs"
    DTYPE = "dtype"
    STRUCTURE = "structure"
    BOUNDS = "bounds"
    NODATA = "nodata"


class IntegritySeverity(Enum):
    """Severity levels for integrity issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class IntegrityIssue:
    """
    Represents a single integrity issue found during validation.

    Attributes:
        check_type: Type of check that found the issue
        severity: Severity level of the issue
        message: Human-readable description
        details: Additional context or technical details
        location: Where the issue was found (band, chunk, etc.)
    """

    check_type: IntegrityCheckType
    severity: IntegritySeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    location: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "location": self.location,
        }


@dataclass
class IntegrityResult:
    """
    Result of integrity validation.

    Attributes:
        is_valid: Whether the file passed all critical checks
        file_path: Path to validated file
        file_size_bytes: File size in bytes
        checksum_md5: MD5 checksum of file
        checksum_sha256: SHA256 checksum of file
        issues: List of integrity issues found
        metadata: Additional validation metadata
        duration_seconds: Time taken for validation
    """

    is_valid: bool
    file_path: Path
    file_size_bytes: int
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    issues: List[IntegrityIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == IntegritySeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == IntegritySeverity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "file_path": str(self.file_path),
            "file_size_bytes": self.file_size_bytes,
            "checksum_md5": self.checksum_md5,
            "checksum_sha256": self.checksum_sha256,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class IntegrityConfig:
    """
    Configuration for integrity validation.

    Attributes:
        compute_checksum: Whether to compute file checksums
        checksum_algorithms: Which checksum algorithms to use
        validate_crs: Whether to validate CRS information
        validate_bounds: Whether to validate spatial bounds
        validate_structure: Whether to validate internal structure
        expected_crs: Expected CRS (if known)
        expected_bounds: Expected bounds (minx, miny, maxx, maxy)
        expected_checksum: Expected checksum to verify
        max_file_size_bytes: Maximum allowed file size
        min_file_size_bytes: Minimum expected file size
        strict_mode: Treat warnings as errors
    """

    compute_checksum: bool = True
    checksum_algorithms: List[str] = field(default_factory=lambda: ["md5", "sha256"])
    validate_crs: bool = True
    validate_bounds: bool = True
    validate_structure: bool = True
    expected_crs: Optional[str] = None
    expected_bounds: Optional[Tuple[float, float, float, float]] = None
    expected_checksum: Optional[str] = None
    max_file_size_bytes: Optional[int] = None
    min_file_size_bytes: Optional[int] = None
    strict_mode: bool = False


class IntegrityValidator:
    """
    Validates data integrity for geospatial files.

    Supports validation of:
    - GeoTIFF/COG files
    - Zarr stores
    - GeoParquet files
    - NetCDF files

    Example:
        validator = IntegrityValidator(IntegrityConfig(compute_checksum=True))
        result = validator.validate_file("data.tif")
        if not result.is_valid:
            for issue in result.issues:
                print(f"{issue.severity}: {issue.message}")
    """

    def __init__(self, config: Optional[IntegrityConfig] = None):
        """
        Initialize integrity validator.

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or IntegrityConfig()

    def validate_file(self, path: Union[str, Path]) -> IntegrityResult:
        """
        Validate integrity of a geospatial file.

        Args:
            path: Path to file to validate

        Returns:
            IntegrityResult with validation details
        """
        import time

        start_time = time.time()
        path = Path(path)
        issues: List[IntegrityIssue] = []
        metadata: Dict[str, Any] = {}

        # Check file exists
        if not path.exists():
            return IntegrityResult(
                is_valid=False,
                file_path=path,
                file_size_bytes=0,
                issues=[
                    IntegrityIssue(
                        check_type=IntegrityCheckType.FORMAT,
                        severity=IntegritySeverity.ERROR,
                        message=f"File not found: {path}",
                    )
                ],
                duration_seconds=time.time() - start_time,
            )

        # Get file size
        file_size = path.stat().st_size
        metadata["file_size_bytes"] = file_size

        # Check file size constraints
        if self.config.max_file_size_bytes and file_size > self.config.max_file_size_bytes:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.ERROR,
                    message=f"File size {file_size} exceeds maximum {self.config.max_file_size_bytes}",
                    details={"file_size": file_size, "max_size": self.config.max_file_size_bytes},
                )
            )

        if self.config.min_file_size_bytes and file_size < self.config.min_file_size_bytes:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.WARNING,
                    message=f"File size {file_size} is below expected minimum {self.config.min_file_size_bytes}",
                    details={"file_size": file_size, "min_size": self.config.min_file_size_bytes},
                )
            )

        # Compute checksums
        checksum_md5 = None
        checksum_sha256 = None
        if self.config.compute_checksum:
            checksums = self._compute_checksums(path)
            checksum_md5 = checksums.get("md5")
            checksum_sha256 = checksums.get("sha256")
            metadata["checksums"] = checksums

            # Verify expected checksum if provided
            if self.config.expected_checksum:
                if checksum_sha256 != self.config.expected_checksum and checksum_md5 != self.config.expected_checksum:
                    issues.append(
                        IntegrityIssue(
                            check_type=IntegrityCheckType.CHECKSUM,
                            severity=IntegritySeverity.ERROR,
                            message="Checksum verification failed",
                            details={
                                "expected": self.config.expected_checksum,
                                "actual_md5": checksum_md5,
                                "actual_sha256": checksum_sha256,
                            },
                        )
                    )

        # Determine file type and run specific validation
        suffix = path.suffix.lower()
        if suffix in [".tif", ".tiff", ".geotiff"]:
            file_issues = self._validate_geotiff(path, metadata)
            issues.extend(file_issues)
        elif suffix == ".zarr" or path.is_dir():
            # Check if it's a Zarr store
            if (path / ".zarray").exists() or (path / ".zgroup").exists():
                file_issues = self._validate_zarr(path, metadata)
                issues.extend(file_issues)
        elif suffix == ".parquet" or suffix == ".geoparquet":
            file_issues = self._validate_geoparquet(path, metadata)
            issues.extend(file_issues)
        elif suffix in [".nc", ".nc4", ".netcdf"]:
            file_issues = self._validate_netcdf(path, metadata)
            issues.extend(file_issues)
        else:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.WARNING,
                    message=f"Unknown file format: {suffix}",
                    details={"suffix": suffix},
                )
            )

        # Determine validity
        error_count = sum(1 for i in issues if i.severity == IntegritySeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == IntegritySeverity.WARNING)

        if self.config.strict_mode:
            is_valid = error_count == 0 and warning_count == 0
        else:
            is_valid = error_count == 0

        duration = time.time() - start_time

        logger.info(
            f"Integrity validation for {path}: valid={is_valid}, "
            f"errors={error_count}, warnings={warning_count}"
        )

        return IntegrityResult(
            is_valid=is_valid,
            file_path=path,
            file_size_bytes=file_size,
            checksum_md5=checksum_md5,
            checksum_sha256=checksum_sha256,
            issues=issues,
            metadata=metadata,
            duration_seconds=duration,
        )

    def _compute_checksums(self, path: Path) -> Dict[str, str]:
        """Compute file checksums."""
        checksums = {}
        hash_funcs = {}

        if "md5" in self.config.checksum_algorithms:
            hash_funcs["md5"] = hashlib.md5()
        if "sha256" in self.config.checksum_algorithms:
            hash_funcs["sha256"] = hashlib.sha256()
        if "sha512" in self.config.checksum_algorithms:
            hash_funcs["sha512"] = hashlib.sha512()

        # Read file in chunks to handle large files
        chunk_size = 8192 * 1024  # 8MB chunks

        with open(path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                for hash_func in hash_funcs.values():
                    hash_func.update(data)

        for name, hash_func in hash_funcs.items():
            checksums[name] = hash_func.hexdigest()

        return checksums

    def _validate_geotiff(self, path: Path, metadata: Dict[str, Any]) -> List[IntegrityIssue]:
        """Validate GeoTIFF file integrity."""
        issues = []

        try:
            import rasterio
        except ImportError:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.WARNING,
                    message="rasterio not available for GeoTIFF validation",
                )
            )
            return issues

        try:
            with rasterio.open(path) as src:
                # Store metadata
                metadata["format"] = "GeoTIFF"
                metadata["driver"] = src.driver
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["band_count"] = src.count
                metadata["dtype"] = str(src.dtypes[0])
                metadata["crs"] = str(src.crs) if src.crs else None
                metadata["bounds"] = list(src.bounds) if src.bounds else None

                # Validate CRS
                if self.config.validate_crs:
                    if src.crs is None:
                        issues.append(
                            IntegrityIssue(
                                check_type=IntegrityCheckType.CRS,
                                severity=IntegritySeverity.WARNING,
                                message="No CRS defined",
                            )
                        )
                    elif self.config.expected_crs:
                        if str(src.crs) != self.config.expected_crs:
                            issues.append(
                                IntegrityIssue(
                                    check_type=IntegrityCheckType.CRS,
                                    severity=IntegritySeverity.WARNING,
                                    message=f"CRS mismatch: expected {self.config.expected_crs}, got {src.crs}",
                                    details={"expected": self.config.expected_crs, "actual": str(src.crs)},
                                )
                            )

                # Validate bounds
                if self.config.validate_bounds and self.config.expected_bounds:
                    bounds = src.bounds
                    expected = self.config.expected_bounds
                    if bounds:
                        # Check if bounds are within expected
                        if (
                            bounds.left < expected[0] - 0.01
                            or bounds.bottom < expected[1] - 0.01
                            or bounds.right > expected[2] + 0.01
                            or bounds.top > expected[3] + 0.01
                        ):
                            issues.append(
                                IntegrityIssue(
                                    check_type=IntegrityCheckType.BOUNDS,
                                    severity=IntegritySeverity.WARNING,
                                    message="Bounds outside expected range",
                                    details={
                                        "expected": expected,
                                        "actual": list(bounds),
                                    },
                                )
                            )

                # Validate structure for COG
                if self.config.validate_structure:
                    profile = src.profile

                    # Check if tiled
                    if not profile.get("tiled"):
                        issues.append(
                            IntegrityIssue(
                                check_type=IntegrityCheckType.STRUCTURE,
                                severity=IntegritySeverity.INFO,
                                message="File is not internally tiled (not COG optimized)",
                            )
                        )

                    # Check for overviews
                    if not src.overviews(1):
                        issues.append(
                            IntegrityIssue(
                                check_type=IntegrityCheckType.STRUCTURE,
                                severity=IntegritySeverity.INFO,
                                message="No overviews present",
                            )
                        )

                    # Record block structure
                    metadata["block_shapes"] = src.block_shapes
                    metadata["is_tiled"] = profile.get("tiled", False)
                    metadata["overviews"] = src.overviews(1) if src.count > 0 else []

                # Validate data can be read
                try:
                    # Read a small sample to verify data integrity
                    sample = src.read(
                        1,
                        window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)),
                    )

                    # Check for NaN/Inf issues
                    if np.any(np.isinf(sample)):
                        issues.append(
                            IntegrityIssue(
                                check_type=IntegrityCheckType.DTYPE,
                                severity=IntegritySeverity.WARNING,
                                message="Data contains infinite values",
                            )
                        )

                except Exception as e:
                    issues.append(
                        IntegrityIssue(
                            check_type=IntegrityCheckType.FORMAT,
                            severity=IntegritySeverity.ERROR,
                            message=f"Failed to read data: {e}",
                        )
                    )

        except rasterio.errors.RasterioError as e:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.ERROR,
                    message=f"Failed to open GeoTIFF: {e}",
                )
            )
        except Exception as e:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.ERROR,
                    message=f"Unexpected error validating GeoTIFF: {e}",
                )
            )

        return issues

    def _validate_zarr(self, path: Path, metadata: Dict[str, Any]) -> List[IntegrityIssue]:
        """Validate Zarr store integrity."""
        issues = []

        try:
            import zarr
        except ImportError:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.WARNING,
                    message="zarr not available for Zarr validation",
                )
            )
            return issues

        try:
            store = zarr.open(str(path), mode="r")
            metadata["format"] = "Zarr"

            if isinstance(store, zarr.Group):
                metadata["type"] = "group"
                metadata["arrays"] = list(store.array_keys())
                metadata["groups"] = list(store.group_keys())

                # Validate each array
                for array_name in store.array_keys():
                    arr = store[array_name]
                    array_issues = self._validate_zarr_array(arr, array_name)
                    issues.extend(array_issues)

            elif isinstance(store, zarr.Array):
                metadata["type"] = "array"
                array_issues = self._validate_zarr_array(store, "root")
                issues.extend(array_issues)

        except Exception as e:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.ERROR,
                    message=f"Failed to open Zarr store: {e}",
                )
            )

        return issues

    def _validate_zarr_array(self, arr, name: str) -> List[IntegrityIssue]:
        """Validate a single Zarr array."""
        issues = []

        try:
            # Check array properties
            if arr.size == 0:
                issues.append(
                    IntegrityIssue(
                        check_type=IntegrityCheckType.STRUCTURE,
                        severity=IntegritySeverity.WARNING,
                        message=f"Array '{name}' is empty",
                        location=name,
                    )
                )

            # Check chunks are valid
            if any(c <= 0 for c in arr.chunks):
                issues.append(
                    IntegrityIssue(
                        check_type=IntegrityCheckType.STRUCTURE,
                        severity=IntegritySeverity.ERROR,
                        message=f"Array '{name}' has invalid chunk sizes",
                        location=name,
                        details={"chunks": arr.chunks},
                    )
                )

            # Try to read a sample
            sample_slices = tuple(slice(0, min(10, s)) for s in arr.shape)
            sample = arr[sample_slices]

            # Check for invalid values
            if np.issubdtype(sample.dtype, np.floating):
                if np.any(np.isinf(sample)):
                    issues.append(
                        IntegrityIssue(
                            check_type=IntegrityCheckType.DTYPE,
                            severity=IntegritySeverity.WARNING,
                            message=f"Array '{name}' contains infinite values",
                            location=name,
                        )
                    )

        except Exception as e:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.ERROR,
                    message=f"Failed to validate array '{name}': {e}",
                    location=name,
                )
            )

        return issues

    def _validate_geoparquet(self, path: Path, metadata: Dict[str, Any]) -> List[IntegrityIssue]:
        """Validate GeoParquet file integrity."""
        issues = []

        try:
            import pyarrow.parquet as pq
        except ImportError:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.WARNING,
                    message="pyarrow not available for GeoParquet validation",
                )
            )
            return issues

        try:
            # Read parquet metadata
            parquet_file = pq.ParquetFile(path)
            meta = parquet_file.metadata
            schema = parquet_file.schema_arrow

            metadata["format"] = "GeoParquet"
            metadata["num_rows"] = meta.num_rows
            metadata["num_columns"] = meta.num_columns
            metadata["num_row_groups"] = meta.num_row_groups
            metadata["columns"] = [field.name for field in schema]

            # Check for geometry column
            geo_meta = None
            if schema.metadata and b"geo" in schema.metadata:
                import json

                geo_meta = json.loads(schema.metadata[b"geo"])
                metadata["geometry_columns"] = list(geo_meta.get("columns", {}).keys())
            else:
                issues.append(
                    IntegrityIssue(
                        check_type=IntegrityCheckType.STRUCTURE,
                        severity=IntegritySeverity.WARNING,
                        message="No GeoParquet geometry metadata found",
                    )
                )

            # Validate row groups
            if meta.num_row_groups == 0:
                issues.append(
                    IntegrityIssue(
                        check_type=IntegrityCheckType.STRUCTURE,
                        severity=IntegritySeverity.ERROR,
                        message="Parquet file has no row groups",
                    )
                )

            # Try to read a sample
            try:
                table = parquet_file.read_row_group(0)
                if table.num_rows == 0:
                    issues.append(
                        IntegrityIssue(
                            check_type=IntegrityCheckType.STRUCTURE,
                            severity=IntegritySeverity.WARNING,
                            message="First row group is empty",
                        )
                    )
            except Exception as e:
                issues.append(
                    IntegrityIssue(
                        check_type=IntegrityCheckType.FORMAT,
                        severity=IntegritySeverity.ERROR,
                        message=f"Failed to read row group: {e}",
                    )
                )

        except Exception as e:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.ERROR,
                    message=f"Failed to open GeoParquet file: {e}",
                )
            )

        return issues

    def _validate_netcdf(self, path: Path, metadata: Dict[str, Any]) -> List[IntegrityIssue]:
        """Validate NetCDF file integrity."""
        issues = []

        try:
            import xarray as xr
        except ImportError:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.WARNING,
                    message="xarray not available for NetCDF validation",
                )
            )
            return issues

        try:
            ds = xr.open_dataset(path)

            metadata["format"] = "NetCDF"
            metadata["dimensions"] = dict(ds.dims)
            metadata["variables"] = list(ds.data_vars)
            metadata["coords"] = list(ds.coords)
            metadata["attrs"] = dict(ds.attrs)

            # Check for common issues
            if not ds.data_vars:
                issues.append(
                    IntegrityIssue(
                        check_type=IntegrityCheckType.STRUCTURE,
                        severity=IntegritySeverity.WARNING,
                        message="No data variables found in NetCDF",
                    )
                )

            # Check CRS if available
            if self.config.validate_crs:
                crs_found = False
                for coord in ds.coords:
                    if "crs" in coord.lower() or ds.coords[coord].attrs.get("grid_mapping_name"):
                        crs_found = True
                        break

                if not crs_found and "crs" not in ds.attrs:
                    issues.append(
                        IntegrityIssue(
                            check_type=IntegrityCheckType.CRS,
                            severity=IntegritySeverity.INFO,
                            message="No CRS information found in NetCDF",
                        )
                    )

            ds.close()

        except Exception as e:
            issues.append(
                IntegrityIssue(
                    check_type=IntegrityCheckType.FORMAT,
                    severity=IntegritySeverity.ERROR,
                    message=f"Failed to open NetCDF file: {e}",
                )
            )

        return issues

    def verify_checksum(
        self,
        path: Union[str, Path],
        expected_checksum: str,
        algorithm: str = "sha256",
    ) -> bool:
        """
        Verify file matches expected checksum.

        Args:
            path: Path to file
            expected_checksum: Expected checksum value
            algorithm: Hash algorithm (md5, sha256, sha512)

        Returns:
            True if checksum matches
        """
        path = Path(path)
        if not path.exists():
            return False

        original_algorithms = self.config.checksum_algorithms
        self.config.checksum_algorithms = [algorithm]

        try:
            checksums = self._compute_checksums(path)
            actual = checksums.get(algorithm)
            return actual == expected_checksum
        finally:
            self.config.checksum_algorithms = original_algorithms


def validate_integrity(
    path: Union[str, Path],
    compute_checksum: bool = True,
    strict: bool = False,
) -> IntegrityResult:
    """
    Convenience function to validate file integrity.

    Args:
        path: Path to file to validate
        compute_checksum: Whether to compute checksums
        strict: Whether to treat warnings as errors

    Returns:
        IntegrityResult with validation details
    """
    config = IntegrityConfig(
        compute_checksum=compute_checksum,
        strict_mode=strict,
    )
    validator = IntegrityValidator(config)
    return validator.validate_file(path)
