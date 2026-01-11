"""
Forecast Data Ingestion Module.

Handles ingestion, parsing, and normalization of weather forecast data
from various providers (GFS, ECMWF, ERA5, etc.) into a unified format
for analysis pipeline consumption.

Key Capabilities:
- Multi-provider forecast data loading (GFS, ECMWF HRES, ECMWF ENS, ERA5)
- Forecast ensemble handling with member tracking
- Variable extraction and unit normalization
- Temporal/spatial subsetting for analysis area
- Forecast lead time management
- Quality metadata tracking for forecast data
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ForecastProvider(Enum):
    """Supported forecast data providers."""

    GFS = "gfs"                    # NOAA Global Forecast System
    ECMWF_HRES = "ecmwf_hres"      # ECMWF High Resolution
    ECMWF_ENS = "ecmwf_ens"        # ECMWF Ensemble
    ERA5 = "era5"                  # ECMWF ERA5 Reanalysis
    ERA5_LAND = "era5_land"        # ERA5-Land (higher resolution)
    NAM = "nam"                    # North American Mesoscale
    HRRR = "hrrr"                  # High-Resolution Rapid Refresh
    CUSTOM = "custom"              # User-provided forecast


class ForecastVariable(Enum):
    """Standard forecast variables."""

    # Temperature
    TEMPERATURE = "temperature"
    TEMPERATURE_2M = "temperature_2m"
    TEMPERATURE_MAX = "temperature_max"
    TEMPERATURE_MIN = "temperature_min"

    # Precipitation
    PRECIPITATION = "precipitation"
    PRECIPITATION_RATE = "precipitation_rate"
    ACCUMULATED_PRECIP = "accumulated_precip"
    CONVECTIVE_PRECIP = "convective_precip"

    # Wind
    WIND_U = "wind_u"
    WIND_V = "wind_v"
    WIND_SPEED = "wind_speed"
    WIND_GUST = "wind_gust"
    WIND_DIRECTION = "wind_direction"

    # Pressure
    PRESSURE = "pressure"
    PRESSURE_MSL = "pressure_msl"
    PRESSURE_SURFACE = "pressure_surface"

    # Humidity
    HUMIDITY = "humidity"
    RELATIVE_HUMIDITY = "relative_humidity"
    SPECIFIC_HUMIDITY = "specific_humidity"
    DEWPOINT = "dewpoint"

    # Cloud
    CLOUD_COVER = "cloud_cover"
    CLOUD_BASE = "cloud_base"

    # Radiation
    SOLAR_RADIATION = "solar_radiation"
    DOWNWARD_SW = "downward_sw"
    DOWNWARD_LW = "downward_lw"

    # Soil/Land
    SOIL_MOISTURE = "soil_moisture"
    SOIL_TEMPERATURE = "soil_temperature"
    SNOW_DEPTH = "snow_depth"
    SNOW_WATER_EQ = "snow_water_eq"

    # Severe Weather
    CAPE = "cape"  # Convective Available Potential Energy
    CIN = "cin"    # Convective Inhibition
    LIFTED_INDEX = "lifted_index"


class ForecastType(Enum):
    """Type of forecast data."""

    DETERMINISTIC = "deterministic"  # Single forecast
    ENSEMBLE = "ensemble"            # Multiple members
    REANALYSIS = "reanalysis"        # Historical analysis
    NOWCAST = "nowcast"              # Very short-term


@dataclass
class ForecastMetadata:
    """
    Metadata for a forecast dataset.

    Attributes:
        provider: Forecast data provider
        forecast_type: Type of forecast (deterministic, ensemble, etc.)
        initialization_time: When the forecast was initialized
        variables: Available forecast variables
        lead_times: Available forecast lead times
        spatial_resolution_m: Spatial resolution in meters
        temporal_resolution: Time step between forecast values
        ensemble_members: Number of ensemble members (if ensemble)
        model_version: Version of forecast model
        source_url: URL or path to source data
    """

    provider: ForecastProvider
    forecast_type: ForecastType
    initialization_time: datetime
    variables: List[ForecastVariable]
    lead_times: List[timedelta]
    spatial_resolution_m: float
    temporal_resolution: timedelta
    ensemble_members: Optional[int] = None
    model_version: Optional[str] = None
    source_url: Optional[str] = None
    crs: str = "EPSG:4326"
    bounds: Optional[Tuple[float, float, float, float]] = None  # (west, south, east, north)

    def __post_init__(self):
        """Validate metadata."""
        if self.initialization_time.tzinfo is None:
            self.initialization_time = self.initialization_time.replace(tzinfo=timezone.utc)

        if self.forecast_type == ForecastType.ENSEMBLE and not self.ensemble_members:
            logger.warning("Ensemble forecast without ensemble_members specified")

    @property
    def max_lead_time(self) -> timedelta:
        """Maximum forecast lead time."""
        return max(self.lead_times) if self.lead_times else timedelta(0)

    @property
    def forecast_horizon(self) -> datetime:
        """Latest valid time in forecast."""
        return self.initialization_time + self.max_lead_time

    def valid_times(self) -> List[datetime]:
        """Get all valid forecast times."""
        return [self.initialization_time + lt for lt in self.lead_times]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider.value,
            "forecast_type": self.forecast_type.value,
            "initialization_time": self.initialization_time.isoformat(),
            "variables": [v.value for v in self.variables],
            "lead_times_hours": [lt.total_seconds() / 3600 for lt in self.lead_times],
            "spatial_resolution_m": self.spatial_resolution_m,
            "temporal_resolution_hours": self.temporal_resolution.total_seconds() / 3600,
            "ensemble_members": self.ensemble_members,
            "model_version": self.model_version,
            "source_url": self.source_url,
            "crs": self.crs,
            "bounds": self.bounds,
        }


@dataclass
class ForecastTimestep:
    """
    A single timestep of forecast data.

    Attributes:
        valid_time: Valid time of this forecast step
        lead_time: Lead time from initialization
        data: Dictionary of variable name -> numpy array
        ensemble_member: Ensemble member ID (if applicable)
        quality_flags: Quality indicators per variable
    """

    valid_time: datetime
    lead_time: timedelta
    data: Dict[str, np.ndarray]
    ensemble_member: Optional[int] = None
    quality_flags: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timezone."""
        if self.valid_time.tzinfo is None:
            self.valid_time = self.valid_time.replace(tzinfo=timezone.utc)

    @property
    def lead_hours(self) -> float:
        """Lead time in hours."""
        return self.lead_time.total_seconds() / 3600

    def get_variable(self, var: Union[str, ForecastVariable]) -> Optional[np.ndarray]:
        """Get data for a variable."""
        key = var.value if isinstance(var, ForecastVariable) else var
        return self.data.get(key)

    def has_variable(self, var: Union[str, ForecastVariable]) -> bool:
        """Check if variable is present."""
        key = var.value if isinstance(var, ForecastVariable) else var
        return key in self.data


@dataclass
class ForecastDataset:
    """
    Complete forecast dataset with all timesteps and metadata.

    Attributes:
        metadata: Forecast metadata
        timesteps: List of forecast timesteps
        grid_lat: Latitude grid (1D or 2D array)
        grid_lon: Longitude grid (1D or 2D array)
        units: Unit specification per variable
    """

    metadata: ForecastMetadata
    timesteps: List[ForecastTimestep]
    grid_lat: np.ndarray
    grid_lon: np.ndarray
    units: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Sort timesteps by valid time."""
        self.timesteps.sort(key=lambda t: t.valid_time)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the spatial grid."""
        return self.grid_lat.shape

    @property
    def time_range(self) -> Tuple[datetime, datetime]:
        """Time range of forecast data."""
        if not self.timesteps:
            return (self.metadata.initialization_time, self.metadata.initialization_time)
        return (self.timesteps[0].valid_time, self.timesteps[-1].valid_time)

    @property
    def variables(self) -> List[str]:
        """Available variables across all timesteps."""
        all_vars = set()
        for ts in self.timesteps:
            all_vars.update(ts.data.keys())
        return sorted(all_vars)

    def get_timestep(self, valid_time: datetime) -> Optional[ForecastTimestep]:
        """Get timestep for a specific valid time."""
        for ts in self.timesteps:
            if ts.valid_time == valid_time:
                return ts
        return None

    def get_timestep_nearest(self, valid_time: datetime) -> Optional[ForecastTimestep]:
        """Get nearest timestep to valid time."""
        if not self.timesteps:
            return None
        return min(self.timesteps, key=lambda ts: abs((ts.valid_time - valid_time).total_seconds()))

    def slice_time(self, start: datetime, end: datetime) -> 'ForecastDataset':
        """Get a time slice of the dataset."""
        filtered = [ts for ts in self.timesteps if start <= ts.valid_time <= end]
        return ForecastDataset(
            metadata=self.metadata,
            timesteps=filtered,
            grid_lat=self.grid_lat,
            grid_lon=self.grid_lon,
            units=self.units,
        )

    def slice_spatial(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> 'ForecastDataset':
        """
        Get a spatial subset of the dataset.

        Args:
            west: Western bound (longitude)
            south: Southern bound (latitude)
            east: Eastern bound (longitude)
            north: Northern bound (latitude)

        Returns:
            New ForecastDataset with spatial subset
        """
        # Handle 1D vs 2D grids
        if self.grid_lat.ndim == 1:
            lat_mask = (self.grid_lat >= south) & (self.grid_lat <= north)
            lon_mask = (self.grid_lon >= west) & (self.grid_lon <= east)
            new_lat = self.grid_lat[lat_mask]
            new_lon = self.grid_lon[lon_mask]

            # Slice data
            new_timesteps = []
            for ts in self.timesteps:
                new_data = {}
                for var, arr in ts.data.items():
                    if arr.ndim >= 2:
                        new_data[var] = arr[lat_mask][:, lon_mask]
                    else:
                        new_data[var] = arr
                new_timesteps.append(ForecastTimestep(
                    valid_time=ts.valid_time,
                    lead_time=ts.lead_time,
                    data=new_data,
                    ensemble_member=ts.ensemble_member,
                    quality_flags=ts.quality_flags,
                ))
        else:
            # 2D grid
            lat_mask = (self.grid_lat >= south) & (self.grid_lat <= north)
            lon_mask = (self.grid_lon >= west) & (self.grid_lon <= east)
            combined_mask = lat_mask & lon_mask

            # Find bounding box of valid points
            rows = np.any(combined_mask, axis=1)
            cols = np.any(combined_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]] if rows.any() else (0, 0)
            cmin, cmax = np.where(cols)[0][[0, -1]] if cols.any() else (0, 0)

            new_lat = self.grid_lat[rmin:rmax+1, cmin:cmax+1]
            new_lon = self.grid_lon[rmin:rmax+1, cmin:cmax+1]

            new_timesteps = []
            for ts in self.timesteps:
                new_data = {}
                for var, arr in ts.data.items():
                    if arr.ndim >= 2:
                        new_data[var] = arr[rmin:rmax+1, cmin:cmax+1]
                    else:
                        new_data[var] = arr
                new_timesteps.append(ForecastTimestep(
                    valid_time=ts.valid_time,
                    lead_time=ts.lead_time,
                    data=new_data,
                    ensemble_member=ts.ensemble_member,
                    quality_flags=ts.quality_flags,
                ))

        new_metadata = ForecastMetadata(
            provider=self.metadata.provider,
            forecast_type=self.metadata.forecast_type,
            initialization_time=self.metadata.initialization_time,
            variables=self.metadata.variables,
            lead_times=self.metadata.lead_times,
            spatial_resolution_m=self.metadata.spatial_resolution_m,
            temporal_resolution=self.metadata.temporal_resolution,
            ensemble_members=self.metadata.ensemble_members,
            model_version=self.metadata.model_version,
            source_url=self.metadata.source_url,
            crs=self.metadata.crs,
            bounds=(west, south, east, north),
        )

        return ForecastDataset(
            metadata=new_metadata,
            timesteps=new_timesteps,
            grid_lat=new_lat,
            grid_lon=new_lon,
            units=self.units,
        )

    def to_xarray(self) -> Any:
        """
        Convert to xarray Dataset (if xarray available).

        Returns:
            xarray.Dataset with all variables and dimensions
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required for to_xarray()")

        # Build coordinate arrays
        times = [ts.valid_time for ts in self.timesteps]
        lead_times_hours = [ts.lead_hours for ts in self.timesteps]

        # Create data arrays
        data_vars = {}
        for var in self.variables:
            # Stack timesteps
            arrays = []
            for ts in self.timesteps:
                arr = ts.get_variable(var)
                if arr is not None:
                    arrays.append(arr)

            if arrays:
                stacked = np.stack(arrays, axis=0)
                data_vars[var] = (["time", "lat", "lon"], stacked)

        coords = {
            "time": times,
            "lead_time_hours": ("time", lead_times_hours),
            "lat": (["lat"], self.grid_lat if self.grid_lat.ndim == 1 else self.grid_lat[:, 0]),
            "lon": (["lon"], self.grid_lon if self.grid_lon.ndim == 1 else self.grid_lon[0, :]),
        }

        attrs = self.metadata.to_dict()

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)


@dataclass
class ForecastIngestionConfig:
    """
    Configuration for forecast data ingestion.

    Attributes:
        providers: List of providers to use (in priority order)
        variables: Variables to extract
        temporal_range: Time range to extract (start, end)
        spatial_bounds: Bounding box (west, south, east, north)
        target_resolution_m: Target resolution for resampling
        normalize_units: Whether to normalize units to standard
        cache_forecasts: Whether to cache downloaded forecasts
        max_lead_time_hours: Maximum lead time to extract
    """

    providers: List[ForecastProvider] = field(default_factory=lambda: [ForecastProvider.GFS])
    variables: List[ForecastVariable] = field(default_factory=lambda: [
        ForecastVariable.PRECIPITATION,
        ForecastVariable.WIND_SPEED,
        ForecastVariable.TEMPERATURE,
    ])
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    target_resolution_m: Optional[float] = None
    normalize_units: bool = True
    cache_forecasts: bool = True
    max_lead_time_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "providers": [p.value for p in self.providers],
            "variables": [v.value for v in self.variables],
            "temporal_range": [t.isoformat() for t in self.temporal_range] if self.temporal_range else None,
            "spatial_bounds": self.spatial_bounds,
            "target_resolution_m": self.target_resolution_m,
            "normalize_units": self.normalize_units,
            "cache_forecasts": self.cache_forecasts,
            "max_lead_time_hours": self.max_lead_time_hours,
        }


@dataclass
class ForecastIngestionResult:
    """
    Result of forecast ingestion.

    Attributes:
        success: Whether ingestion was successful
        dataset: Ingested forecast dataset (if successful)
        provider_used: Provider that was used
        errors: List of errors encountered
        warnings: List of warnings
        ingestion_time: Time taken for ingestion
    """

    success: bool
    dataset: Optional[ForecastDataset] = None
    provider_used: Optional[ForecastProvider] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ingestion_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "provider_used": self.provider_used.value if self.provider_used else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "ingestion_time_seconds": self.ingestion_time_seconds,
            "dataset_metadata": self.dataset.metadata.to_dict() if self.dataset else None,
        }


class ForecastIngester:
    """
    Handles ingestion of forecast data from various providers.

    Provides a unified interface for loading forecast data from
    different sources and normalizing to a common format.

    Example:
        ingester = ForecastIngester()
        config = ForecastIngestionConfig(
            providers=[ForecastProvider.GFS],
            variables=[ForecastVariable.PRECIPITATION, ForecastVariable.WIND_SPEED],
            spatial_bounds=(-82, 24, -79, 27),  # Florida
        )
        result = ingester.ingest(config)
        if result.success:
            dataset = result.dataset
    """

    # Variable name mappings per provider
    VARIABLE_MAPPINGS: Dict[ForecastProvider, Dict[ForecastVariable, str]] = {
        ForecastProvider.GFS: {
            ForecastVariable.TEMPERATURE: "TMP",
            ForecastVariable.TEMPERATURE_2M: "TMP_2m",
            ForecastVariable.PRECIPITATION: "APCP",
            ForecastVariable.PRECIPITATION_RATE: "PRATE",
            ForecastVariable.WIND_U: "UGRD",
            ForecastVariable.WIND_V: "VGRD",
            ForecastVariable.PRESSURE_MSL: "PRMSL",
            ForecastVariable.HUMIDITY: "RH",
            ForecastVariable.CLOUD_COVER: "TCDC",
        },
        ForecastProvider.ECMWF_HRES: {
            ForecastVariable.TEMPERATURE: "t",
            ForecastVariable.TEMPERATURE_2M: "2t",
            ForecastVariable.PRECIPITATION: "tp",
            ForecastVariable.WIND_U: "u",
            ForecastVariable.WIND_V: "v",
            ForecastVariable.PRESSURE_MSL: "msl",
            ForecastVariable.HUMIDITY: "r",
            ForecastVariable.CLOUD_COVER: "tcc",
        },
        ForecastProvider.ERA5: {
            ForecastVariable.TEMPERATURE: "t",
            ForecastVariable.TEMPERATURE_2M: "t2m",
            ForecastVariable.PRECIPITATION: "tp",
            ForecastVariable.WIND_U: "u10",
            ForecastVariable.WIND_V: "v10",
            ForecastVariable.PRESSURE_MSL: "msl",
            ForecastVariable.HUMIDITY: "r",
            ForecastVariable.CLOUD_COVER: "tcc",
            ForecastVariable.SOIL_MOISTURE: "swvl1",
        },
    }

    # Standard units per variable
    STANDARD_UNITS: Dict[ForecastVariable, str] = {
        ForecastVariable.TEMPERATURE: "K",
        ForecastVariable.TEMPERATURE_2M: "K",
        ForecastVariable.PRECIPITATION: "mm",
        ForecastVariable.PRECIPITATION_RATE: "mm/h",
        ForecastVariable.WIND_U: "m/s",
        ForecastVariable.WIND_V: "m/s",
        ForecastVariable.WIND_SPEED: "m/s",
        ForecastVariable.PRESSURE_MSL: "Pa",
        ForecastVariable.HUMIDITY: "%",
        ForecastVariable.CLOUD_COVER: "%",
    }

    # Provider characteristics
    PROVIDER_INFO: Dict[ForecastProvider, Dict[str, Any]] = {
        ForecastProvider.GFS: {
            "resolution_m": 13000,
            "temporal_resolution_hours": 3,
            "forecast_horizon_days": 16,
            "ensemble_members": None,
        },
        ForecastProvider.ECMWF_HRES: {
            "resolution_m": 9000,
            "temporal_resolution_hours": 1,
            "forecast_horizon_days": 10,
            "ensemble_members": None,
        },
        ForecastProvider.ECMWF_ENS: {
            "resolution_m": 18000,
            "temporal_resolution_hours": 3,
            "forecast_horizon_days": 15,
            "ensemble_members": 51,
        },
        ForecastProvider.ERA5: {
            "resolution_m": 30000,
            "temporal_resolution_hours": 1,
            "forecast_horizon_days": 0,  # Reanalysis
            "ensemble_members": None,
        },
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        api_keys: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize forecast ingester.

        Args:
            cache_dir: Directory for caching forecast data
            api_keys: API keys for providers requiring authentication
        """
        self.cache_dir = cache_dir or Path.home() / ".multiverse_dive" / "forecast_cache"
        self.api_keys = api_keys or {}
        self._loaders: Dict[ForecastProvider, Callable] = {}

        # Register default loaders
        self._register_default_loaders()

    def _register_default_loaders(self):
        """Register default provider loaders."""
        self._loaders[ForecastProvider.GFS] = self._load_gfs
        self._loaders[ForecastProvider.ECMWF_HRES] = self._load_ecmwf
        self._loaders[ForecastProvider.ECMWF_ENS] = self._load_ecmwf
        self._loaders[ForecastProvider.ERA5] = self._load_era5
        self._loaders[ForecastProvider.CUSTOM] = self._load_custom

    def register_loader(
        self,
        provider: ForecastProvider,
        loader: Callable[[ForecastIngestionConfig], ForecastDataset],
    ):
        """Register a custom loader for a provider."""
        self._loaders[provider] = loader

    def ingest(
        self,
        config: ForecastIngestionConfig,
    ) -> ForecastIngestionResult:
        """
        Ingest forecast data according to configuration.

        Tries providers in order until one succeeds.

        Args:
            config: Ingestion configuration

        Returns:
            ForecastIngestionResult with dataset or errors
        """
        import time
        start_time = time.time()

        errors = []
        warnings = []

        for provider in config.providers:
            logger.info(f"Attempting to load forecast from {provider.value}")

            if provider not in self._loaders:
                warnings.append(f"No loader registered for {provider.value}")
                continue

            try:
                loader = self._loaders[provider]
                dataset = loader(config)

                # Apply spatial subset if specified
                if config.spatial_bounds and dataset:
                    dataset = dataset.slice_spatial(*config.spatial_bounds)

                # Apply temporal subset if specified
                if config.temporal_range and dataset:
                    dataset = dataset.slice_time(*config.temporal_range)

                # Filter by max lead time
                if config.max_lead_time_hours and dataset:
                    max_lead = timedelta(hours=config.max_lead_time_hours)
                    dataset.timesteps = [
                        ts for ts in dataset.timesteps
                        if ts.lead_time <= max_lead
                    ]

                elapsed = time.time() - start_time
                logger.info(f"Successfully loaded forecast from {provider.value} in {elapsed:.2f}s")

                return ForecastIngestionResult(
                    success=True,
                    dataset=dataset,
                    provider_used=provider,
                    warnings=warnings,
                    ingestion_time_seconds=elapsed,
                )

            except Exception as e:
                error_msg = f"Failed to load from {provider.value}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue

        elapsed = time.time() - start_time
        return ForecastIngestionResult(
            success=False,
            errors=errors,
            warnings=warnings,
            ingestion_time_seconds=elapsed,
        )

    def _load_gfs(self, config: ForecastIngestionConfig) -> ForecastDataset:
        """Load GFS forecast data (stub implementation)."""
        logger.info("Loading GFS forecast data")

        # In real implementation, would use NOMADS or AWS S3
        # For now, create synthetic data for testing
        return self._create_synthetic_forecast(
            provider=ForecastProvider.GFS,
            config=config,
        )

    def _load_ecmwf(self, config: ForecastIngestionConfig) -> ForecastDataset:
        """Load ECMWF forecast data (stub implementation)."""
        logger.info("Loading ECMWF forecast data")

        # Check for API key
        if "ecmwf" not in self.api_keys:
            raise ValueError("ECMWF API key required")

        return self._create_synthetic_forecast(
            provider=ForecastProvider.ECMWF_HRES,
            config=config,
        )

    def _load_era5(self, config: ForecastIngestionConfig) -> ForecastDataset:
        """Load ERA5 reanalysis data (stub implementation)."""
        logger.info("Loading ERA5 reanalysis data")

        return self._create_synthetic_forecast(
            provider=ForecastProvider.ERA5,
            config=config,
        )

    def _load_custom(self, config: ForecastIngestionConfig) -> ForecastDataset:
        """Load custom forecast data from local files."""
        raise NotImplementedError("Custom loader requires explicit registration")

    def _create_synthetic_forecast(
        self,
        provider: ForecastProvider,
        config: ForecastIngestionConfig,
    ) -> ForecastDataset:
        """
        Create synthetic forecast data for testing.

        This is a placeholder that generates realistic-looking
        forecast data for development and testing purposes.
        """
        info = self.PROVIDER_INFO.get(provider, {})

        # Determine spatial grid
        bounds = config.spatial_bounds or (-82, 24, -79, 27)  # Default: Florida
        west, south, east, north = bounds

        resolution = config.target_resolution_m or info.get("resolution_m", 10000)
        n_lat = max(10, int((north - south) * 111000 / resolution))
        n_lon = max(10, int((east - west) * 111000 * np.cos(np.radians((north + south) / 2)) / resolution))

        grid_lat = np.linspace(south, north, n_lat)
        grid_lon = np.linspace(west, east, n_lon)

        # Determine temporal range
        now = datetime.now(timezone.utc)
        init_time = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)

        temporal_res_hours = info.get("temporal_resolution_hours", 3)
        horizon_days = info.get("forecast_horizon_days", 7)

        if config.max_lead_time_hours:
            max_lead = timedelta(hours=config.max_lead_time_hours)
        else:
            max_lead = timedelta(days=horizon_days)

        # Generate timesteps
        timesteps = []
        lead_times = []
        current_lead = timedelta(0)

        while current_lead <= max_lead:
            lead_times.append(current_lead)
            valid_time = init_time + current_lead

            # Generate data for each variable
            data = {}
            for var in config.variables:
                arr = self._generate_variable_data(var, n_lat, n_lon, current_lead.total_seconds() / 3600)
                if arr is not None:
                    data[var.value] = arr

            timesteps.append(ForecastTimestep(
                valid_time=valid_time,
                lead_time=current_lead,
                data=data,
                quality_flags={v.value: 1.0 for v in config.variables},
            ))

            current_lead += timedelta(hours=temporal_res_hours)

        # Build metadata
        metadata = ForecastMetadata(
            provider=provider,
            forecast_type=ForecastType.REANALYSIS if provider == ForecastProvider.ERA5 else ForecastType.DETERMINISTIC,
            initialization_time=init_time,
            variables=config.variables,
            lead_times=lead_times,
            spatial_resolution_m=resolution,
            temporal_resolution=timedelta(hours=temporal_res_hours),
            ensemble_members=info.get("ensemble_members"),
            model_version="synthetic-v1.0",
            bounds=bounds,
        )

        # Build units
        units = {v.value: self.STANDARD_UNITS.get(v, "unknown") for v in config.variables}

        return ForecastDataset(
            metadata=metadata,
            timesteps=timesteps,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            units=units,
        )

    def _generate_variable_data(
        self,
        var: ForecastVariable,
        n_lat: int,
        n_lon: int,
        lead_hours: float,
    ) -> Optional[np.ndarray]:
        """Generate synthetic data for a variable."""
        # Base pattern with some spatial variation
        x = np.linspace(0, 2 * np.pi, n_lon)
        y = np.linspace(0, 2 * np.pi, n_lat)
        xx, yy = np.meshgrid(x, y)

        # Add lead time variation
        phase = lead_hours * 0.1

        if var in [ForecastVariable.TEMPERATURE, ForecastVariable.TEMPERATURE_2M]:
            # Temperature: 280-310 K
            return 295 + 10 * np.sin(xx + phase) * np.cos(yy) + np.random.randn(n_lat, n_lon) * 2

        elif var in [ForecastVariable.PRECIPITATION, ForecastVariable.ACCUMULATED_PRECIP]:
            # Precipitation: 0-50 mm, with some zeros
            base = np.maximum(0, 10 * np.sin(xx + phase) * np.sin(yy) + 5)
            base[np.random.rand(n_lat, n_lon) > 0.3] = 0
            return base + np.abs(np.random.randn(n_lat, n_lon) * 2)

        elif var == ForecastVariable.PRECIPITATION_RATE:
            # Precipitation rate: 0-10 mm/h
            base = np.maximum(0, 2 * np.sin(xx + phase) * np.sin(yy) + 1)
            base[np.random.rand(n_lat, n_lon) > 0.3] = 0
            return base + np.abs(np.random.randn(n_lat, n_lon) * 0.5)

        elif var in [ForecastVariable.WIND_U, ForecastVariable.WIND_V]:
            # Wind components: -20 to 20 m/s
            return 5 * np.sin(xx + phase) + np.random.randn(n_lat, n_lon) * 2

        elif var == ForecastVariable.WIND_SPEED:
            # Wind speed: 0-30 m/s
            u = 5 * np.sin(xx + phase) + np.random.randn(n_lat, n_lon) * 2
            v = 5 * np.cos(yy + phase) + np.random.randn(n_lat, n_lon) * 2
            return np.sqrt(u**2 + v**2)

        elif var == ForecastVariable.WIND_GUST:
            # Wind gust: slightly higher than wind speed
            speed = 5 * np.sin(xx + phase) + np.random.randn(n_lat, n_lon) * 2
            return np.abs(speed) * 1.5 + np.abs(np.random.randn(n_lat, n_lon) * 3)

        elif var in [ForecastVariable.PRESSURE, ForecastVariable.PRESSURE_MSL]:
            # Pressure: 98000-103000 Pa
            return 101325 + 2000 * np.sin(xx + phase) + np.random.randn(n_lat, n_lon) * 100

        elif var in [ForecastVariable.HUMIDITY, ForecastVariable.RELATIVE_HUMIDITY]:
            # Humidity: 0-100%
            return np.clip(60 + 30 * np.sin(xx + phase) * np.cos(yy) + np.random.randn(n_lat, n_lon) * 10, 0, 100)

        elif var == ForecastVariable.CLOUD_COVER:
            # Cloud cover: 0-100%
            return np.clip(50 + 40 * np.sin(xx + phase) + np.random.randn(n_lat, n_lon) * 15, 0, 100)

        elif var == ForecastVariable.SOIL_MOISTURE:
            # Soil moisture: 0-1 (volumetric)
            return np.clip(0.3 + 0.2 * np.sin(xx) + np.random.randn(n_lat, n_lon) * 0.05, 0, 1)

        else:
            # Default: return noise
            return np.random.randn(n_lat, n_lon)


def ingest_forecast(
    providers: Optional[List[ForecastProvider]] = None,
    variables: Optional[List[ForecastVariable]] = None,
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None,
    temporal_range: Optional[Tuple[datetime, datetime]] = None,
    max_lead_hours: Optional[float] = None,
) -> ForecastIngestionResult:
    """
    Convenience function to ingest forecast data.

    Args:
        providers: List of providers to try (default: [GFS])
        variables: Variables to extract
        spatial_bounds: Bounding box (west, south, east, north)
        temporal_range: Time range to extract
        max_lead_hours: Maximum lead time in hours

    Returns:
        ForecastIngestionResult with dataset or errors
    """
    config = ForecastIngestionConfig(
        providers=providers or [ForecastProvider.GFS],
        variables=variables or [
            ForecastVariable.PRECIPITATION,
            ForecastVariable.WIND_SPEED,
            ForecastVariable.TEMPERATURE,
        ],
        spatial_bounds=spatial_bounds,
        temporal_range=temporal_range,
        max_lead_time_hours=max_lead_hours,
    )

    ingester = ForecastIngester()
    return ingester.ingest(config)
