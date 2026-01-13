# Claude Context

Geospatial event intelligence platform that converts (area, time window, event type) into decision products.

## Quick Reference

| Item | Location |
|------|----------|
| Bug fixes (check first!) | `FIXES.md` |
| Roadmap & tasks | `ROADMAP.md` |
| Agent memory | `.claude/agents/PROJECT_MEMORY.md` |
| Active specs | `OPENSPEC.md` |
| Completed specs | `docs/OPENSPEC_ARCHIVE.md` |

## Current Status

- **Core Platform:** Complete (170K+ lines, 518+ tests)
- **P0 Bugs:** 4 remaining - **FIX THESE FIRST**

### P0 Bugs (from FIXES.md)

| ID | Issue | File |
|----|-------|------|
| FIX-003 | WCS duplicate dict key | `core/data/discovery/wms_wcs.py:379` |
| FIX-004 | Hallucinated scipy API (grey_erosion) | `core/analysis/library/baseline/flood/hand_model.py:305` |
| FIX-005 | Wrong distance_transform_edt usage | `core/analysis/library/baseline/flood/hand_model.py:378` |
| FIX-006 | Broken schema $ref | `openspec/schemas/provenance.schema.json:112` |

### Work Streams (after bugs fixed)

1. **Image Validation** - Band validation before processing
2. **Distributed Processing** - Dask parallelization for large rasters

## Before Starting Work

1. Check `FIXES.md` for P0 bugs
2. Read `.claude/agents/PROJECT_MEMORY.md` for context
3. Run tests: `./run_tests.py` or `./run_tests.py <category>`

## Test Commands

```bash
./run_tests.py                    # All tests
./run_tests.py flood              # Flood tests
./run_tests.py wildfire           # Wildfire tests
./run_tests.py schemas            # Schema tests
./run_tests.py --algorithm sar    # Specific algorithm
./run_tests.py --list             # Show categories
```

## Git Workflow

```bash
source ~/.keychain/*-sh           # Load SSH keychain
git add <files>
git commit -m "Short description"
git push origin main
```

Email: `gpriceless@users.noreply.github.com`
