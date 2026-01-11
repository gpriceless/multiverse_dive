"""
Pytest configuration and fixtures for multiverse_dive tests.

Markers:
    @pytest.mark.flood - Flood-related tests
    @pytest.mark.wildfire - Wildfire-related tests
    @pytest.mark.storm - Storm-related tests
    @pytest.mark.slow - Tests that take longer to run
    @pytest.mark.integration - Integration tests requiring external resources
    @pytest.mark.quality - Quality control and uncertainty tests
    @pytest.mark.agents - Agent orchestration tests

Usage:
    pytest -m flood              # Run only flood tests
    pytest -m "not slow"         # Skip slow tests
    pytest -m "flood and not slow"  # Fast flood tests only
    pytest -m quality            # Run quality control tests
    pytest -m agents             # Run agent tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "flood: Flood hazard tests")
    config.addinivalue_line("markers", "wildfire: Wildfire hazard tests")
    config.addinivalue_line("markers", "storm: Storm hazard tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "algorithm(name): Test specific algorithm")
    config.addinivalue_line("markers", "schema: JSON schema tests")
    config.addinivalue_line("markers", "intent: Intent resolution tests")
    config.addinivalue_line("markers", "provider: Data provider tests")
    config.addinivalue_line("markers", "quality: Quality control tests")
    config.addinivalue_line("markers", "agents: Agent orchestration tests")


def pytest_collection_modifyitems(config, items):
    """Auto-apply markers based on test file names and test names."""
    for item in items:
        # Mark based on file name
        if "flood" in item.fspath.basename:
            item.add_marker(pytest.mark.flood)
        if "wildfire" in item.fspath.basename:
            item.add_marker(pytest.mark.wildfire)
        if "storm" in item.fspath.basename:
            item.add_marker(pytest.mark.storm)
        if "schema" in item.fspath.basename:
            item.add_marker(pytest.mark.schema)
        if "intent" in item.fspath.basename:
            item.add_marker(pytest.mark.intent)
        if "provider" in item.fspath.basename:
            item.add_marker(pytest.mark.provider)

        # Mark based on test name
        test_name = item.name.lower()
        if "flood" in test_name and not item.get_closest_marker("flood"):
            item.add_marker(pytest.mark.flood)
        if "wildfire" in test_name or "fire" in test_name:
            if not item.get_closest_marker("wildfire"):
                item.add_marker(pytest.mark.wildfire)
        if "storm" in test_name or "wind" in test_name:
            if not item.get_closest_marker("storm"):
                item.add_marker(pytest.mark.storm)

        # Mark slow tests
        if "large" in test_name or "stress" in test_name or "performance" in test_name:
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def sample_dem():
    """Provide a sample DEM array for testing."""
    import numpy as np
    np.random.seed(42)
    # Create a simple terrain with a valley
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    dem = 100 + 20 * np.sin(X * 0.5) + 10 * np.cos(Y * 0.3)
    # Add a river channel
    dem[45:55, :] -= 15
    return dem.astype(np.float32)


@pytest.fixture
def sample_sar_image():
    """Provide a sample SAR backscatter image for testing."""
    import numpy as np
    np.random.seed(42)
    # Simulated SAR backscatter in dB
    sar = np.random.normal(-12, 3, (100, 100))
    # Add water (low backscatter)
    sar[40:60, 20:80] = np.random.normal(-20, 1, (20, 60))
    return sar.astype(np.float32)


@pytest.fixture
def sample_optical_bands():
    """Provide sample optical bands (Green, NIR) for NDWI testing."""
    import numpy as np
    np.random.seed(42)
    # Simulated reflectance values [0, 1]
    green = np.random.uniform(0.05, 0.15, (100, 100))
    nir = np.random.uniform(0.2, 0.4, (100, 100))
    # Add water pixels (high green, low NIR relative to land)
    green[40:60, 20:80] = np.random.uniform(0.08, 0.12, (20, 60))
    nir[40:60, 20:80] = np.random.uniform(0.02, 0.08, (20, 60))
    return {
        "green": green.astype(np.float32),
        "nir": nir.astype(np.float32)
    }


@pytest.fixture
def sample_event_spec():
    """Provide a sample event specification."""
    return {
        "event": {
            "id": "test_event_001",
            "intent": {
                "class": "flood.coastal",
                "source": "explicit"
            },
            "spatial": {
                "type": "Polygon",
                "coordinates": [[[-80.5, 25.5], [-80.0, 25.5], [-80.0, 26.0], [-80.5, 26.0], [-80.5, 25.5]]],
                "crs": "EPSG:4326"
            },
            "temporal": {
                "start": "2024-09-15T00:00:00Z",
                "end": "2024-09-20T23:59:59Z"
            }
        }
    }
