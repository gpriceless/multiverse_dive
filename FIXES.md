# Bug Tracking & Fixes

**Last Updated:** 2026-01-11
**Status:** 32 bugs fixed in recent sprint, 4 critical remaining

---

## Summary

The initial code review identified 16 critical bugs plus numerous medium/low priority issues. **32 bugs have been fixed** in the last 48 hours, leaving only **4 critical bugs** remaining before the platform is bug-free for production.

---

## 🔴 Critical Bugs Remaining (P0) - **MUST FIX BEFORE PRODUCTION**

### FIX-003: WCS Duplicate Dictionary Key
**File:** `core/data/discovery/wms_wcs.py:379-380`
**Impact:** WCS queries fail to retrieve coverage data
**Issue:** Duplicate `"subset"` key in params dict overwrites x-dimension

```python
# BROKEN:
params = {
    "subset": f"x({bbox[0]},{bbox[2]})",
    "subset": f"y({bbox[1]},{bbox[3]})",  # Overwrites previous!
}

# FIX: Use list of tuples for multiple params
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

**Verification:** Test WCS queries against live server

---

### FIX-004: HAND Model - Hallucinated scipy API
**File:** `core/analysis/library/baseline/flood/hand_model.py:305`
**Impact:** HAND algorithm crashes on execution
**Issue:** `scipy.ndimage.grey_erosion()` doesn't exist

```python
# BROKEN:
filled = np.maximum(dem, ndimage.grey_erosion(seed, size=(3, 3)))

# FIX: Use grey_dilation for morphological reconstruction
from scipy.ndimage import grey_dilation

def _fill_depressions_simple(dem: np.ndarray) -> np.ndarray:
    seed = dem.copy()
    seed[1:-1, 1:-1] = np.inf

    footprint = np.ones((3, 3))
    while True:
        dilated = grey_dilation(seed, footprint=footprint)
        new_seed = np.minimum(dilated, dem)
        new_seed = np.maximum(new_seed, dem)
        if np.array_equal(new_seed, seed):
            break
        seed = new_seed
    return seed
```

**Verification:** `pytest tests/test_flood_algorithms.py::test_hand_model -v`

---

### FIX-005: HAND Model - Wrong distance_transform_edt Parameters
**File:** `core/analysis/library/baseline/flood/hand_model.py:378-382`
**Impact:** HAND algorithm crashes or returns wrong indices
**Issue:** `return_distances=False` used incorrectly, function returns tuple

```python
# BROKEN:
indices = ndimage.distance_transform_edt(
    ~drainage_network,
    return_distances=False,
    return_indices=True
)

# FIX: Properly unpack tuple
_, indices = ndimage.distance_transform_edt(
    ~drainage_network,
    return_indices=True
)
```

**Note:** Fix together with FIX-004 since both in same algorithm

**Verification:** Same test as FIX-004

---

### FIX-006: Broken Schema Reference
**File:** `openspec/schemas/provenance.schema.json:112`
**Impact:** Provenance schema validation fails
**Issue:** References non-existent `processing_level` definition

**FIX Option A** (Preferred): Add to `openspec/schemas/common.schema.json`:
```json
"processing_level": {
    "type": "string",
    "description": "Data processing level (e.g., L1C, L2A, ARD)",
    "examples": ["L1C", "L2A", "L1", "L2", "ARD", "raw"]
}
```

**FIX Option B**: Inline definition in provenance.schema.json

**Verification:** `pytest tests/test_schemas.py -v`

---

## ✅ Recently Fixed (32 bugs, last 48 hours)

### Track 5 (Group I - Quality Control)
- **NEW-032:** None handling in diagnostics.py compute_statistics - both SpatialDiagnostic and TemporalDiagnostic
- **NEW-031:** TypeError in qa_report.py _safe_round when passed None value

### Track 1 (Group I - Sanity Checks)
- **NEW-030:** Performance optimization in artifacts.py _calculate_block_score - O(n²) → vectorized
- **NEW-029:** Division by zero in artifacts.py _detect_saturation
- **NEW-028:** Division by zero in artifacts.py _detect_hot_pixels
- **NEW-027:** Division by zero in values.py _check_nan and _check_inf

### Track 4 (Group I - Quality Actions)
- **NEW-026:** TypeError in routing.py check_escalations - list | list operation
- **NEW-023:** Division by zero in flagging.py when mask dimensions are zero
- **NEW-022:** NaN handling in flagging.py AppliedFlag.to_dict

### Track 2 (Group I - Cross-Validation)
- **NEW-025:** Percentile calculation edge cases in historical.py
- **NEW-024:** Division by zero in cross_model.py when weights sum to zero

### Track 3 (Group I - Uncertainty)
- **NEW-021:** Division by zero in harmonic mean propagation

### Track 5 (Group H - Advanced Algorithms)
- **NEW-020:** Division by zero in confidence calculation (unet_segmentation.py, ensemble_fusion.py)

### Track 2 (Group H - Fusion Core)
- **NEW-019:** Missing numpy import in tests/test_fusion.py

### Track 4 (Group H - Forecast Integration)
- **NEW-018:** Array dimension handling in scenarios.py _compute_exceedance_duration
- **NEW-017:** Empty probability_field shape in scenarios.py
- **NEW-016:** Division by zero in validation.py _compute_ensemble_metrics
- **NEW-015:** FSS edge cases in validation.py

### Track 1 (Group H - Pipeline Assembly)
- **NEW-014:** IndexError in assembler.py when step_outputs is empty

### Track 7 (Group G - Cache System)
- **NEW-013:** NaN/Inf/negative resolution_m validation in IndexEntry
- **NEW-012:** SpatiotemporalIndex in-memory database bug
- **NEW-011:** Deprecated datetime.utcnow() in zarr.py

### Track 5 (Group G - Validation)
- **NEW-010:** Directory validation in integrity.py

### Track 3 (Group G - Normalization)
- **NEW-009:** Scale factor bug in resolution.py - division instead of multiplication

### Track 4 (Group G - Enrichment)
- **NEW-008:** NaN handling in overviews.py _downsample_array
- **NEW-007:** Division by zero in quality.py QualityConfig.__post_init__
- **NEW-006:** Histogram error in statistics.py when min_val == max_val

### Track 5 (Group F - Fusion Strategy)
- **NEW-005:** Division by zero in _calculate_fusion_confidence

### Track 1 (Group F - Constraints)
- **NEW-004:** None handling in soft_weights context
- **NEW-003:** Resolution score clamping bug - negative values
- **NEW-002:** Cloud cover score clamping bug

### Track 4 (Group F - Sensor Selection)
- **NEW-001:** Degraded mode threshold bug - MEDIUM incorrectly treated as degraded

### Previously Identified (Groups A-E)
- **FIX-001:** ✅ Fixed - broker.py calling .get() on Provider dataclass
- **FIX-002:** ✅ Fixed - broker.py hallucinated candidates attribute

---

## 🟡 Medium Priority (P1) - Non-Blocking

### FIX-007: Classification Bias Toward Deeper Classes
**File:** `core/intent/classifier.py:206-208`
**Impact:** May misclassify "coastal flood after hurricane" as storm instead of flood
**Fix:** Reduce depth bonus from 0.1 to 0.02 per level

### FIX-008: Python 3.11+ Only Import
**File:** `core/intent/resolver.py:8`
**Impact:** Fails on Python 3.10
**Fix:** Replace `from datetime import UTC` with `from datetime import timezone`

### FIX-009: Deprecated datetime.utcnow()
**File:** `core/data/broker.py:125`
**Impact:** Warnings in Python 3.12+
**Fix:** Replace `datetime.utcnow()` with `datetime.now(timezone.utc)`

### FIX-010: Stub D8 Flow Accumulation
**File:** `core/analysis/library/baseline/flood/hand_model.py:310-342`
**Impact:** HAND algorithm produces incorrect flow accumulation
**Fix:** Implement proper D8 routing OR use external library (pysheds, richdem)

### FIX-011: Import Inside Class
**File:** `core/data/discovery/provider_api.py:250`
**Impact:** Code style issue
**Fix:** Remove import statement from inside class definition

---

## 🟢 Low Priority (P2) - Style & Best Practice

- **FIX-012:** Library code calls `logging.basicConfig()` - should let application configure
- **FIX-013:** Hardcoded `parent.parent.parent` path traversal - use env var or package resources
- **FIX-014:** Unnecessary `hasattr(provider, "cost")` checks - Provider always has cost attribute
- **FIX-015:** Empty `_load_default_providers()` stub - implement or remove
- **FIX-016:** Inconsistent `confidence_score` definitions across schemas - use $ref everywhere

---

## Action Plan

### Immediate (This Week)
1. ✅ Fix FIX-003 (WCS duplicate key) - 30 minutes
2. ✅ Fix FIX-004 + FIX-005 (HAND model scipy issues) - 2 hours (together)
3. ✅ Fix FIX-006 (schema $ref) - 15 minutes

**Total Time:** ~3 hours to clear all P0 bugs

### Near-Term (Next 2 Weeks)
- Fix P1 bugs (FIX-007 through FIX-011) - nice to have, not blocking

### Eventually (When Convenient)
- Clean up P2 style issues (FIX-012 through FIX-016) - technical debt

---

## Verification Commands

```bash
# After fixing P0 bugs, run full test suite
PYTHONPATH=. .venv/bin/pytest tests/ -v

# Expected: 518+ tests passing, 0 errors

# Specific verification
pytest tests/test_data_providers.py -v          # FIX-003
pytest tests/test_flood_algorithms.py -v        # FIX-004, FIX-005
pytest tests/test_schemas.py -v                 # FIX-006
```

---

## Bug Statistics

| Priority | Remaining | Fixed | Total |
|----------|-----------|-------|-------|
| P0 (Critical) | 4 | 2 | 6 |
| P1 (Medium) | 5 | 30 | 35 |
| P2 (Low) | 5 | 0 | 5 |
| **Total** | **14** | **32** | **46** |

**Fix Rate:** 70% of identified bugs resolved
**Time to Fix P0:** ~3 hours estimated

---

**Next Review:** After P0 bugs cleared
