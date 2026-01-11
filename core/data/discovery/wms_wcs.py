"""
OGC WMS/WCS discovery adapter.

Queries OGC Web Map Service (WMS) and Web Coverage Service (WCS)
endpoints for geospatial data discovery.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from xml.etree import ElementTree as ET

from core.data.discovery.base import (
    DiscoveryAdapter,
    DiscoveryResult,
    DiscoveryError
)


class WMSWCSAdapter(DiscoveryAdapter):
    """
    OGC WMS/WCS discovery adapter.

    Queries OGC services for available layers/coverages and their capabilities.
    Note: WMS/WCS typically don't support temporal queries as robustly as STAC.
    """

    def __init__(self):
        super().__init__("wms_wcs")
        self._session: Optional[aiohttp.ClientSession] = None

    async def discover(
        self,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[DiscoveryResult]:
        """
        Discover datasets via WMS/WCS GetCapabilities.

        Args:
            provider: Provider configuration
            spatial: GeoJSON geometry
            temporal: Temporal extent
            constraints: Optional constraints

        Returns:
            List of DiscoveryResult objects
        """
        constraints = constraints or {}
        access = provider.access
        endpoint = access["endpoint"]
        protocol = access["protocol"]

        # Query capabilities based on protocol
        try:
            if protocol == "wms":
                results = await self._discover_wms(
                    endpoint=endpoint,
                    provider=provider,
                    spatial=spatial,
                    temporal=temporal
                )
            elif protocol == "wcs":
                results = await self._discover_wcs(
                    endpoint=endpoint,
                    provider=provider,
                    spatial=spatial,
                    temporal=temporal
                )
            else:
                raise DiscoveryError(f"Unsupported protocol: {protocol}")

            return results

        except Exception as e:
            raise DiscoveryError(
                f"WMS/WCS discovery failed: {str(e)}",
                provider=provider.id
            )

    def supports_provider(self, provider: Any) -> bool:
        """Check if provider uses WMS or WCS protocol."""
        protocol = provider.access.get("protocol", "")
        return protocol in ["wms", "wcs"]

    async def _discover_wms(
        self,
        endpoint: str,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str]
    ) -> List[DiscoveryResult]:
        """
        Discover layers via WMS GetCapabilities.

        Args:
            endpoint: WMS endpoint URL
            provider: Provider configuration
            spatial: Spatial extent
            temporal: Temporal extent

        Returns:
            List of DiscoveryResult objects
        """
        # Build GetCapabilities request
        params = {
            "service": "WMS",
            "request": "GetCapabilities",
            "version": "1.3.0"
        }

        # Fetch capabilities
        capabilities_xml = await self._fetch_xml(endpoint, params)

        # Parse capabilities
        try:
            root = ET.fromstring(capabilities_xml)
            namespace = {"wms": "http://www.opengis.net/wms"}

            # Extract layers
            layers = root.findall(".//wms:Layer", namespace)

            results = []
            query_bbox = self._extract_bbox(spatial)

            for layer in layers:
                # Extract layer metadata
                name = layer.find("wms:Name", namespace)
                if name is None or not name.text:
                    continue

                title = layer.find("wms:Title", namespace)
                title_text = title.text if title is not None else name.text

                # Extract bounding box
                bbox_elem = layer.find("wms:BoundingBox", namespace)
                if bbox_elem is None:
                    bbox_elem = layer.find("wms:EX_GeographicBoundingBox", namespace)

                if bbox_elem is not None:
                    layer_bbox = self._parse_wms_bbox(bbox_elem, namespace)
                else:
                    # Default to global
                    layer_bbox = [-180, -90, 180, 90]

                # Calculate spatial coverage
                spatial_coverage = self._calculate_coverage_percent(layer_bbox, query_bbox)

                # Skip if no overlap
                if spatial_coverage == 0:
                    continue

                # Build WMS GetMap URL as source_uri
                source_uri = self._build_wms_getmap_url(
                    endpoint=endpoint,
                    layer_name=name.text,
                    bbox=query_bbox
                )

                # Create DiscoveryResult
                result = DiscoveryResult(
                    dataset_id=f"{provider.id}_{name.text}",
                    provider=provider.id,
                    data_type=provider.type,
                    source_uri=source_uri,
                    format="geotiff",  # WMS GetMap can return GeoTIFF
                    acquisition_time=datetime.now(),  # WMS doesn't provide acquisition time
                    spatial_coverage_percent=spatial_coverage,
                    resolution_m=provider.capabilities.get("resolution_m", 10.0),
                    quality_flag="good",
                    cost_tier=provider.cost.get("tier", "open"),
                    metadata={
                        "layer_name": name.text,
                        "title": title_text,
                        "protocol": "wms"
                    }
                )

                results.append(result)

            return results

        except ET.ParseError as e:
            raise DiscoveryError(f"Failed to parse WMS capabilities: {str(e)}")

    async def _discover_wcs(
        self,
        endpoint: str,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str]
    ) -> List[DiscoveryResult]:
        """
        Discover coverages via WCS GetCapabilities.

        Args:
            endpoint: WCS endpoint URL
            provider: Provider configuration
            spatial: Spatial extent
            temporal: Temporal extent

        Returns:
            List of DiscoveryResult objects
        """
        # Build GetCapabilities request
        params = {
            "service": "WCS",
            "request": "GetCapabilities",
            "version": "2.0.1"
        }

        # Fetch capabilities
        capabilities_xml = await self._fetch_xml(endpoint, params)

        # Parse capabilities (simplified - WCS XML is complex)
        try:
            root = ET.fromstring(capabilities_xml)

            # WCS namespaces can vary - try common ones
            namespaces = {
                "wcs": "http://www.opengis.net/wcs/2.0",
                "ows": "http://www.opengis.net/ows/2.0"
            }

            # Extract coverage summaries
            coverages = root.findall(".//wcs:CoverageSummary", namespaces)

            results = []
            query_bbox = self._extract_bbox(spatial)

            for coverage in coverages:
                # Extract coverage ID
                cov_id = coverage.find("wcs:CoverageId", namespaces)
                if cov_id is None or not cov_id.text:
                    continue

                # Extract bounding box (simplified)
                bbox_elem = coverage.find(".//ows:BoundingBox", namespaces)
                if bbox_elem is not None:
                    lower = bbox_elem.find("ows:LowerCorner", namespaces)
                    upper = bbox_elem.find("ows:UpperCorner", namespaces)

                    if lower is not None and upper is not None:
                        lower_coords = [float(x) for x in lower.text.split()]
                        upper_coords = [float(x) for x in upper.text.split()]
                        layer_bbox = [
                            lower_coords[0], lower_coords[1],
                            upper_coords[0], upper_coords[1]
                        ]
                    else:
                        layer_bbox = [-180, -90, 180, 90]
                else:
                    layer_bbox = [-180, -90, 180, 90]

                # Calculate spatial coverage
                spatial_coverage = self._calculate_coverage_percent(layer_bbox, query_bbox)

                if spatial_coverage == 0:
                    continue

                # Build WCS GetCoverage URL
                source_uri = self._build_wcs_getcoverage_url(
                    endpoint=endpoint,
                    coverage_id=cov_id.text,
                    bbox=query_bbox
                )

                # Create DiscoveryResult
                result = DiscoveryResult(
                    dataset_id=f"{provider.id}_{cov_id.text}",
                    provider=provider.id,
                    data_type=provider.type,
                    source_uri=source_uri,
                    format="geotiff",
                    acquisition_time=datetime.now(),
                    spatial_coverage_percent=spatial_coverage,
                    resolution_m=provider.capabilities.get("resolution_m", 10.0),
                    quality_flag="good",
                    cost_tier=provider.cost.get("tier", "open"),
                    metadata={
                        "coverage_id": cov_id.text,
                        "protocol": "wcs"
                    }
                )

                results.append(result)

            return results

        except ET.ParseError as e:
            raise DiscoveryError(f"Failed to parse WCS capabilities: {str(e)}")

    async def _fetch_xml(self, endpoint: str, params: Dict[str, str]) -> str:
        """Fetch XML response from OGC service."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

        try:
            async with self._session.get(
                endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                return await response.text()

        except aiohttp.ClientError as e:
            raise DiscoveryError(f"Failed to fetch capabilities: {str(e)}")

    def _parse_wms_bbox(
        self,
        bbox_elem: ET.Element,
        namespace: Dict[str, str]
    ) -> List[float]:
        """Parse bounding box from WMS capabilities."""
        # Try BoundingBox element first (has minx, miny, maxx, maxy attributes)
        minx = bbox_elem.get("minx")
        if minx is not None:
            return [
                float(bbox_elem.get("minx", -180)),
                float(bbox_elem.get("miny", -90)),
                float(bbox_elem.get("maxx", 180)),
                float(bbox_elem.get("maxy", 90))
            ]

        # Try EX_GeographicBoundingBox (has child elements)
        west = bbox_elem.find("wms:westBoundLongitude", namespace)
        south = bbox_elem.find("wms:southBoundLatitude", namespace)
        east = bbox_elem.find("wms:eastBoundLongitude", namespace)
        north = bbox_elem.find("wms:northBoundLatitude", namespace)

        if all(e is not None for e in [west, south, east, north]):
            return [
                float(west.text),
                float(south.text),
                float(east.text),
                float(north.text)
            ]

        # Default to global
        return [-180, -90, 180, 90]

    def _build_wms_getmap_url(
        self,
        endpoint: str,
        layer_name: str,
        bbox: List[float]
    ) -> str:
        """Build WMS GetMap request URL."""
        params = {
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": layer_name,
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "crs": "EPSG:4326",
            "width": "1024",
            "height": "1024",
            "format": "image/geotiff"
        }

        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{endpoint}?{param_str}"

    def _build_wcs_getcoverage_url(
        self,
        endpoint: str,
        coverage_id: str,
        bbox: List[float]
    ) -> str:
        """Build WCS GetCoverage request URL."""
        params = [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("coverageId", coverage_id),
            ("subset", f"x({bbox[0]},{bbox[2]})"),
            ("subset", f"y({bbox[1]},{bbox[3]})"),
            ("format", "image/tiff")
        ]

        param_str = "&".join(f"{k}={v}" for k, v in params)
        return f"{endpoint}?{param_str}"

    async def close(self):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
