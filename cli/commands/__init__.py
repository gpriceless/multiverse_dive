"""
Multiverse Dive CLI Commands

This package contains all CLI subcommands for the mdive tool.

Commands:
    discover - Find available data for an area and time window
    ingest   - Download and normalize data to analysis-ready format
    analyze  - Run analysis algorithms on prepared data
    validate - Run quality control checks on results
    export   - Generate final products in various formats
    run      - Execute full pipeline from specification to products
    status   - Check workflow state and progress
    resume   - Resume an interrupted workflow
"""

from cli.commands import (
    discover,
    ingest,
    analyze,
    validate,
    export,
    run,
    status,
    resume,
)

__all__ = [
    "discover",
    "ingest",
    "analyze",
    "validate",
    "export",
    "run",
    "status",
    "resume",
]
