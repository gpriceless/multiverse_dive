# Code Fixes Required

This document tracks bugs, hallucinated APIs, and issues discovered during code review. Each fix includes enough context for an agent to implement the correction without additional research.

---

## Fix Priority Legend

- **P0 (Critical)**: Will cause runtime errors - must fix before any testing
- **P1 (Medium)**: Logic bugs or deprecated APIs - fix before production
- **P2 (Low)**: Style/best practice - fix when convenient

---

## P0: Critical Fixes

### FIX-001: broker.py - Calling .get() on Provider dataclass **[FIXED]**

**File:** `core/data/broker.py`
**Line:** 455
**Status:** Fixed (2026-01-10, Track 1 agent)
**Category:** Hallucinated API

**Problem:**
```python
provider_info = self.provider_registry.get_provider(candidate.provider)
if provider_info:
    return provider_info.get("preference_score", 0.5)
```

The `get_provider()` method returns a `Provider` dataclass, not a dictionary. Calling `.get()` on it will raise `AttributeError`.

**Fix:**
```python
provider_info = self.provider_registry.get_provider(candidate.provider)
if provider_info:
    return provider_info.metadata.get("preference_score", 0.5)
```

**Context:** The `Provider` dataclass is defined in `core/data/providers/registry.py` with a `metadata: dict` field that contains preference_score.

**Verification:** Run `PYTHONPATH=. .venv/bin/pytest tests/test_data_providers.py -v`

---

### FIX-002: broker.py - Hallucinated candidates attribute **[FIXED]**

**File:** `core/data/broker.py`
**Line:** 152
**Status:** Fixed (2026-01-10, Track 1 agent)
**Category:** Hallucinated attribute

**Problem:**
```python
for result in results:
    if isinstance(result, DiscoveryResult):
        all_results.extend(result.candidates if hasattr(result, 'candidates') else [result])
```

The `DiscoveryResult` dataclass (defined in `core/data/discovery/base.py`) does not have a `candidates` attribute. The `discover()` method returns `List[DiscoveryResult]`, not an object containing candidates.

**Fix:**
```python
for result in results:
    if isinstance(result, DiscoveryResult):
        all_results.append(result)
    elif isinstance(result, list):
        all_results.extend(result)
```

**Context:** Check `core/data/discovery/base.py` for `DiscoveryResult` dataclass definition to confirm available fields.

**Verification:** Run `PYTHONPATH=. .venv/bin/pytest tests/test_data_providers.py -v`

---

### FIX-003: wms_wcs.py - Duplicate dictionary key

**File:** `core/data/discovery/wms_wcs.py`
**Lines:** 379-380
**Status:** Open
**Category:** Logic bug

**Problem:**
```python
params = {
    "service": "WCS",
    "version": "2.0.1",
    "request": "GetCoverage",
    "coverageId": coverage_id,
    "subset": f"x({bbox[0]},{bbox[2]})",
    "subset": f"y({bbox[1]},{bbox[3]})",  # Overwrites previous "subset"!
    "format": "image/tiff"
}
```

Python dictionaries cannot have duplicate keys. The second `"subset"` overwrites the first, losing the x-dimension subset entirely.

**Fix:**
WCS 2.0 allows multiple subset parameters. Use a list of tuples or construct the URL manually:
```python
# Option 1: Build params without subset, add to URL manually
params = {
    "service": "WCS",
    "version": "2.0.1",
    "request": "GetCoverage",
    "coverageId": coverage_id,
    "format": "image/tiff"
}
# Then append: &subset=x(...)&subset=y(...) to URL

# Option 2: Use requests with a list of tuples
params = [
    ("service", "WCS"),
    ("version", "2.0.1"),
    ("request", "GetCoverage"),
    ("coverageId", coverage_id),
    ("subset", f"x({bbox[0]},{bbox[2]})"),
    ("subset", f"y({bbox[1]},{bbox[3]})"),
    ("format", "image/tiff")
]
```

**Verification:** Manual test or add unit test for WCS query construction

---

### FIX-004: hand_model.py - Hallucinated scipy API (grey_erosion)

**File:** `core/analysis/library/baseline/flood/hand_model.py`
**Line:** 305
**Status:** Open
**Category:** Hallucinated API

**Problem:**
```python
filled = np.maximum(dem, ndimage.grey_erosion(seed, size=(3, 3)))
```

`scipy.ndimage.grey_erosion()` does not exist. The correct function for morphological erosion on grayscale images is `scipy.ndimage.grey_dilation()` (for dilation) or `scipy.ndimage.minimum_filter()` / `scipy.ndimage.maximum_filter()`.

**Fix:**
For depression filling via morphological reconstruction, the typical approach uses iterative dilation:
```python
# Use grey_dilation for morphological reconstruction
from scipy.ndimage import grey_dilation

def _fill_depressions_simple(dem: np.ndarray) -> np.ndarray:
    """Fill depressions using morphological reconstruction."""
    seed = dem.copy()
    seed[1:-1, 1:-1] = np.inf

    # Iterative reconstruction
    footprint = np.ones((3, 3))
    while True:
        dilated = grey_dilation(seed, footprint=footprint)
        new_seed = np.minimum(dilated, dem)
        new_seed = np.maximum(new_seed, dem)  # Ensure we don't go below original
        if np.array_equal(new_seed, seed):
            break
        seed = new_seed
    return seed
```

**Alternative:** Mark this algorithm as experimental/stub and document that it needs proper implementation.

**Context:** See scipy documentation for `scipy.ndimage` morphological operations.

**Verification:** Run `PYTHONPATH=. .venv/bin/pytest tests/test_flood_algorithms.py -v`

---

### FIX-005: hand_model.py - Wrong scipy API parameters

**File:** `core/analysis/library/baseline/flood/hand_model.py`
**Lines:** 378-382
**Status:** Open
**Category:** Hallucinated API parameter

**Problem:**
```python
indices = ndimage.distance_transform_edt(
    ~drainage_network,
    return_distances=False,  # This parameter doesn't exist!
    return_indices=True
)
```

The `return_distances` parameter does not exist in `scipy.ndimage.distance_transform_edt()`. The function signature is:
```python
distance_transform_edt(input, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None)
```

Wait - it does exist but the default is `True`. However, when `return_indices=True` and `return_distances=True` (or not specified), it returns a tuple `(distances, indices)`.

**Fix:**
```python
distances, indices = ndimage.distance_transform_edt(
    ~drainage_network,
    return_indices=True
)
# Or if you only need indices:
_, indices = ndimage.distance_transform_edt(
    ~drainage_network,
    return_indices=True
)
```

**Verification:** Run `PYTHONPATH=. .venv/bin/pytest tests/test_flood_algorithms.py -v`

---

### FIX-006: provenance.schema.json - Broken $ref

**File:** `openspec/schemas/provenance.schema.json`
**Line:** 112
**Status:** Open
**Category:** Missing schema definition

**Problem:**
```json
"processing_level": {
    "$ref": "common.schema.json#/$defs/processing_level"
}
```

The `processing_level` definition does not exist in `common.schema.json`. This will cause JSON Schema validation to fail.

**Fix - Option A:** Add the missing definition to `common.schema.json`:
```json
"processing_level": {
    "type": "string",
    "description": "Data processing level (e.g., L1C, L2A, ARD)",
    "examples": ["L1C", "L2A", "L1", "L2", "ARD", "raw"]
}
```

**Fix - Option B:** Change the reference in `provenance.schema.json` to inline definition:
```json
"processing_level": {
    "type": "string",
    "description": "Data processing level"
}
```

**Preferred:** Option A - maintains consistency with other schemas that may reference processing_level.

**Verification:** Run `PYTHONPATH=. .venv/bin/pytest tests/test_schemas.py -v`

---

## P1: Medium Priority Fixes

### FIX-007: classifier.py - Classification bias toward deeper classes

**File:** `core/intent/classifier.py`
**Lines:** 206-208
**Status:** Open
**Category:** Logic bug

**Problem:**
The confidence scoring adds a depth bonus (`class_path.count(".") * 0.1`) that unfairly advantages deeper classes. For input "coastal flood after hurricane":
- `flood` matches "flood" → confidence ~0.175 (depth 0)
- `storm.tropical_cyclone` matches "hurricane" → confidence ~0.245 (depth 1)
- `flood.coastal` matches "coastal flood" → confidence ~0.245 (depth 1)

The algorithm may incorrectly classify flood events as storms.

**Fix:**
Consider weighting by keyword specificity rather than class depth, or reduce the depth bonus:
```python
# Reduce depth bonus from 0.1 to 0.02 per level
depth_bonus = class_path.count(".") * 0.02

# Or weight by number of matching keywords instead
keyword_weight = len(matched_keywords) * 0.15
```

**Verification:** Add test case: `classify("coastal flood after hurricane")` should return `flood.coastal.*` not `storm.*`

---

### FIX-008: resolver.py - Python 3.11+ only import

**File:** `core/intent/resolver.py`
**Line:** 8
**Status:** Open
**Category:** Compatibility

**Problem:**
```python
from datetime import UTC, datetime
```

`datetime.UTC` was added in Python 3.11. This will fail on Python 3.10 and earlier.

**Fix:**
```python
from datetime import datetime, timezone

# Later in code, replace:
#   datetime.now(UTC)
# with:
#   datetime.now(timezone.utc)
```

**Verification:** Test import on Python 3.10 if available, or just apply fix preventatively.

---

### FIX-009: broker.py - Deprecated datetime.utcnow()

**File:** `core/data/broker.py`
**Line:** 125
**Status:** Open
**Category:** Deprecated API

**Problem:**
```python
query_timestamp = datetime.utcnow()
```

`datetime.utcnow()` is deprecated as of Python 3.12.

**Fix:**
```python
from datetime import datetime, timezone

query_timestamp = datetime.now(timezone.utc)
```

**Verification:** No warnings when running with Python 3.12+

---

### FIX-010: hand_model.py - Stub D8 flow accumulation

**File:** `core/analysis/library/baseline/flood/hand_model.py`
**Lines:** 310-342
**Status:** Open
**Category:** Incomplete implementation

**Problem:**
The D8 flow accumulation implementation just counts upslope neighbors instead of properly routing flow and accumulating contributing area. The comment admits "NOT a proper flow accumulation algorithm".

**Fix Options:**

1. **Mark as experimental:** Add clear warning in docstring and metadata that this is a simplified placeholder
2. **Implement properly:** Use a proper D8 algorithm with flow direction and accumulation
3. **Use external library:** Integrate with `pysheds`, `richdem`, or `whitebox` for proper flow routing

**Minimum fix:** Update the algorithm metadata to set `validated_regions: []` and add a warning flag.

---

### FIX-011: provider_api.py - Import inside class

**File:** `core/data/discovery/provider_api.py`
**Line:** 250
**Status:** Open
**Category:** Code style

**Problem:**
```python
class ProviderStrategy(ABC):
    from abc import ABC, abstractmethod  # Import inside class!
```

**Fix:**
Remove the import line from inside the class. `ABC` and `abstractmethod` should already be imported at the top of the file.

---

## P2: Low Priority Fixes

### FIX-012: resolver.py - Library configures logging

**File:** `core/intent/resolver.py`
**Lines:** 15-16
**Status:** Open
**Category:** Best practice

**Problem:**
```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

Library code should not call `basicConfig()` as it affects global logging configuration.

**Fix:**
```python
logger = logging.getLogger(__name__)
# Remove basicConfig() call - let application configure logging
```

---

### FIX-013: registry.py - Hardcoded path traversal

**File:** `core/intent/registry.py`
**Lines:** 108-109
**Status:** Open
**Category:** Fragility

**Problem:**
```python
project_root = Path(__file__).parent.parent.parent
definitions_dir = project_root / "openspec" / "definitions" / "event_classes"
```

Uses brittle `parent.parent.parent` path traversal.

**Fix:**
Use package resources or environment variable:
```python
import importlib.resources

# Option 1: Environment variable
definitions_dir = Path(os.environ.get(
    "OPENSPEC_DEFINITIONS_DIR",
    Path(__file__).parent.parent.parent / "openspec" / "definitions" / "event_classes"
))

# Option 2: Package resources (Python 3.9+)
# with importlib.resources.files("openspec.definitions.event_classes") as p:
#     definitions_dir = p
```

---

### FIX-014: Multiple files - Unnecessary hasattr checks

**Files:**
- `core/data/discovery/stac.py:256`
- `core/data/discovery/wms_wcs.py:173, 281`
- `core/data/discovery/provider_api.py:214`

**Status:** Open
**Category:** Code cleanup

**Problem:**
```python
cost_tier = provider.cost.get("tier", "open") if hasattr(provider, "cost") else "open"
```

The `Provider` dataclass always has a `cost` attribute (with default factory).

**Fix:**
```python
cost_tier = provider.cost.get("tier", "open")
```

---

### FIX-015: providers/registry.py - Empty stub method

**File:** `core/data/providers/registry.py`
**Line:** 68
**Status:** Open
**Category:** Incomplete

**Problem:**
```python
def _load_default_providers(self):
    """Load default provider configurations."""
    pass
```

**Fix:**
Either implement default loading or remove the method and its call if not needed.

---

### FIX-016: Schemas - Inconsistent confidence_score usage

**Files:** Multiple schema files
**Status:** Open
**Category:** Consistency

**Problem:**
Some schemas define confidence inline while others use `$ref` to `common.schema.json#/$defs/confidence_score`.

**Affected:**
- `intent.schema.json` lines 39-41, 53-54, 77-80 - inline
- `event.schema.json` lines 34-38 - inline
- `quality.schema.json` line 33 - uses $ref
- `provenance.schema.json` lines 152, 160, 181 - uses $ref

**Fix:**
Update all inline confidence definitions to use:
```json
"confidence": {
    "$ref": "common.schema.json#/$defs/confidence_score"
}
```

---

## Verification Commands

After applying fixes, run the full test suite:

```bash
# Set PYTHONPATH and run all tests
PYTHONPATH=. .venv/bin/pytest tests/ -v

# Run specific test files
PYTHONPATH=. .venv/bin/pytest tests/test_schemas.py -v
PYTHONPATH=. .venv/bin/pytest tests/test_validator.py -v
PYTHONPATH=. .venv/bin/pytest tests/test_intent.py -v
PYTHONPATH=. .venv/bin/pytest tests/test_data_providers.py -v
PYTHONPATH=. .venv/bin/pytest tests/test_flood_algorithms.py -v
```

---

## Changelog

| Date | Fix ID | Status | Notes |
|------|--------|--------|-------|
| 2026-01-10 | NEW-005 | Fixed | Track 5: Division by zero in _calculate_fusion_confidence when configuration.sensors is empty |
| 2026-01-10 | NEW-004 | Fixed | Track 1: None handling in soft_weights context (AttributeError when soft_weights was None) |
| 2026-01-10 | NEW-003 | Fixed | Track 1: Resolution score clamping bug in constraints.py (negative values produced invalid scores > 1.0) |
| 2026-01-10 | NEW-002 | Fixed | Track 1: Cloud cover score clamping bug in constraints.py (negative/over 100% values produced invalid scores) |
| 2026-01-10 | NEW-001 | Fixed | Track 4: Degraded mode threshold bug in strategy.py (MEDIUM confidence incorrectly treated as degraded) |
| 2026-01-09 | All | Documented | Initial code review completed |
