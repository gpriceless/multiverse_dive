"""
Status Command - Check workflow state and progress.

Usage:
    mdive status --workdir ./products/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import click

logger = logging.getLogger("mdive.status")


@click.command("status")
@click.option(
    "--workdir",
    "-w",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Working directory containing workflow state.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format (default: text).",
)
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    default=False,
    help="Show detailed stage information.",
)
@click.pass_obj
def status(
    ctx,
    workdir: Path,
    output_format: str,
    detailed: bool,
):
    """
    Check workflow state and progress.

    Displays the current status of a workflow including completed stages,
    current stage, and any errors encountered.

    \b
    Examples:
        # Check workflow status
        mdive status --workdir ./products/

        # Detailed status as JSON
        mdive status --workdir ./products/ --format json --detailed
    """
    # Load workflow state
    state_file = workdir / ".workflow_state.json"

    if not state_file.exists():
        click.echo(f"No workflow state found in {workdir}")
        click.echo("This directory may not be a workflow output directory.")
        raise SystemExit(1)

    with open(state_file) as f:
        state = json.load(f)

    # Load ingest state if available
    ingest_state_file = workdir / ".ingest_state.json"
    ingest_state = None
    if ingest_state_file.exists():
        with open(ingest_state_file) as f:
            ingest_state = json.load(f)

    if output_format == "json":
        output_json_status(state, ingest_state, detailed)
    else:
        output_text_status(state, ingest_state, workdir, detailed)


def output_text_status(
    state: Dict[str, Any],
    ingest_state: Optional[Dict[str, Any]],
    workdir: Path,
    detailed: bool,
):
    """Output status as formatted text."""
    click.echo(f"\n{'=' * 50}")
    click.echo(f"  Workflow Status")
    click.echo(f"{'=' * 50}")

    # Basic info
    click.echo(f"\n  Directory: {workdir}")
    click.echo(f"  Started: {format_datetime(state.get('started_at'))}")

    # Configuration
    config = state.get("config", {})
    if config:
        click.echo(f"\n  Configuration:")
        click.echo(f"    Event type: {config.get('event_type', 'unknown')}")
        click.echo(f"    Profile: {config.get('profile', 'unknown')}")
        if config.get("algorithm"):
            click.echo(f"    Algorithm: {config.get('algorithm')}")

    # Stage status
    stages = ["discover", "ingest", "analyze", "validate", "export"]
    completed = state.get("completed_stages", [])
    current = state.get("current_stage")
    stage_results = state.get("stage_results", {})

    click.echo(f"\n  Stages:")
    for stage in stages:
        if stage in completed:
            result = stage_results.get(stage, {})
            elapsed = calculate_elapsed(
                result.get("started_at"),
                result.get("completed_at"),
            )
            click.echo(f"    [X] {stage.capitalize()}: completed ({elapsed})")
        elif stage == current:
            result = stage_results.get(stage, {})
            status_str = result.get("status", "in_progress")
            if status_str == "failed":
                error = result.get("error", "unknown error")
                click.echo(f"    [!] {stage.capitalize()}: FAILED - {error}")
            else:
                started = format_datetime(result.get("started_at"))
                click.echo(f"    [>] {stage.capitalize()}: in progress (started {started})")
        else:
            click.echo(f"    [ ] {stage.capitalize()}: pending")

    # Ingest progress if available
    if ingest_state:
        completed_items = len(ingest_state.get("completed", []))
        failed_items = len(ingest_state.get("failed", []))
        in_progress = ingest_state.get("in_progress")

        click.echo(f"\n  Ingest Progress:")
        click.echo(f"    Completed: {completed_items}")
        click.echo(f"    Failed: {failed_items}")
        if in_progress:
            click.echo(f"    In progress: {in_progress}")

    # Overall status
    click.echo(f"\n  Overall Status: ", nl=False)
    if all(s in completed for s in stages):
        click.echo("COMPLETED")
    elif current and stage_results.get(current, {}).get("status") == "failed":
        click.echo("FAILED")
    elif current:
        click.echo("IN PROGRESS")
    else:
        click.echo("PENDING")

    # Detailed stage information
    if detailed:
        click.echo(f"\n{'=' * 50}")
        click.echo(f"  Stage Details")
        click.echo(f"{'=' * 50}")

        for stage in stages:
            result = stage_results.get(stage, {})
            if result:
                click.echo(f"\n  {stage.capitalize()}:")
                click.echo(f"    Status: {result.get('status', 'unknown')}")
                if result.get("started_at"):
                    click.echo(f"    Started: {format_datetime(result.get('started_at'))}")
                if result.get("completed_at"):
                    click.echo(f"    Completed: {format_datetime(result.get('completed_at'))}")
                if result.get("error"):
                    click.echo(f"    Error: {result.get('error')}")
                if result.get("result"):
                    click.echo(f"    Result: {json.dumps(result.get('result'), indent=6)}")

    # Next steps
    click.echo(f"\n  Next Steps:")
    if all(s in completed for s in stages):
        click.echo("    Workflow complete. Products available in 'products/' directory.")
    elif current and stage_results.get(current, {}).get("status") == "failed":
        click.echo(f"    Fix the error and run: mdive resume --workdir {workdir}")
    elif current:
        click.echo("    Workflow is running. Wait for completion or interrupt to resume later.")
    else:
        next_stage = next((s for s in stages if s not in completed), stages[0])
        click.echo(f"    Resume workflow: mdive resume --workdir {workdir}")

    click.echo()


def output_json_status(
    state: Dict[str, Any],
    ingest_state: Optional[Dict[str, Any]],
    detailed: bool,
):
    """Output status as JSON."""
    stages = ["discover", "ingest", "analyze", "validate", "export"]
    completed = state.get("completed_stages", [])
    current = state.get("current_stage")
    stage_results = state.get("stage_results", {})

    # Calculate overall status
    if all(s in completed for s in stages):
        overall_status = "completed"
    elif current and stage_results.get(current, {}).get("status") == "failed":
        overall_status = "failed"
    elif current:
        overall_status = "in_progress"
    else:
        overall_status = "pending"

    output = {
        "overall_status": overall_status,
        "started_at": state.get("started_at"),
        "config": state.get("config", {}),
        "stages": {},
    }

    for stage in stages:
        if stage in completed:
            output["stages"][stage] = "completed"
        elif stage == current:
            status_str = stage_results.get(stage, {}).get("status", "in_progress")
            output["stages"][stage] = status_str
        else:
            output["stages"][stage] = "pending"

    if detailed:
        output["stage_details"] = stage_results
        if ingest_state:
            output["ingest_details"] = ingest_state

    click.echo(json.dumps(output, indent=2))


def format_datetime(dt_str: Optional[str]) -> str:
    """Format datetime string for display."""
    if not dt_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return dt_str


def calculate_elapsed(start_str: Optional[str], end_str: Optional[str]) -> str:
    """Calculate elapsed time between two datetime strings."""
    if not start_str or not end_str:
        return "unknown"
    try:
        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        elapsed = end - start
        seconds = elapsed.total_seconds()
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"
    except (ValueError, TypeError):
        return "unknown"
