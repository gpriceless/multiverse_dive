"""
Multiverse Dive CLI - Main Entry Point

Command-line interface for the geospatial event intelligence platform.
Built with Click for robust argument parsing and help generation.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mdive")


class MdiveContext:
    """Context object for passing global options to subcommands."""

    def __init__(
        self,
        verbose: bool = False,
        quiet: bool = False,
        config_path: Optional[Path] = None,
    ):
        self.verbose = verbose
        self.quiet = quiet
        self.config_path = config_path
        self._config = None

        # Configure logging based on verbosity
        if quiet:
            logger.setLevel(logging.WARNING)
        elif verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    @property
    def config(self):
        """Lazy load configuration from file."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self):
        """Load configuration from file or defaults."""
        import yaml

        config = {
            "default_profile": "workstation",
            "cache_dir": Path.home() / ".mdive" / "cache",
            "data_dir": Path.home() / ".mdive" / "data",
            "output_dir": Path.cwd() / "output",
            "providers": {
                "stac_catalogs": [
                    "https://earth-search.aws.element84.com/v1",
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                ],
            },
            "profiles": {
                "laptop": {
                    "memory_mb": 2048,
                    "max_workers": 2,
                    "tile_size": 256,
                },
                "workstation": {
                    "memory_mb": 8192,
                    "max_workers": 4,
                    "tile_size": 512,
                },
                "cloud": {
                    "memory_mb": 32768,
                    "max_workers": 16,
                    "tile_size": 1024,
                },
                "edge": {
                    "memory_mb": 1024,
                    "max_workers": 1,
                    "tile_size": 128,
                },
            },
        }

        # Load user config if it exists
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        self._merge_config(config, user_config)
                if self.verbose:
                    logger.debug(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        # Check for default config locations
        default_paths = [
            Path.cwd() / ".mdive.yaml",
            Path.cwd() / "mdive.yaml",
            Path.home() / ".mdive" / "config.yaml",
        ]

        for path in default_paths:
            if path.exists() and not self.config_path:
                try:
                    with open(path) as f:
                        user_config = yaml.safe_load(f)
                        if user_config:
                            self._merge_config(config, user_config)
                    if self.verbose:
                        logger.debug(f"Loaded config from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")

        return config

    def _merge_config(self, base: dict, override: dict):
        """Deep merge override into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value


# Custom Click group with enhanced help formatting
class MdiveGroup(click.Group):
    """Custom Click group with improved help formatting."""

    def format_help(self, ctx, formatter):
        """Format help with custom banner and examples."""
        # Add banner
        formatter.write_paragraph()
        formatter.write_text(
            "Multiverse Dive - Geospatial Event Intelligence Platform"
        )
        formatter.write_paragraph()
        formatter.write_text(
            "Transform (area, time window, event type) into decision products."
        )
        formatter.write_paragraph()

        # Standard help formatting
        super().format_help(ctx, formatter)

        # Add examples section
        formatter.write_paragraph()
        formatter.write_text("Examples:")
        formatter.indent()

        examples = [
            "# Discover available data for a flood event",
            "mdive discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood",
            "",
            "# Download and prepare data",
            "mdive ingest --area miami.geojson --source sentinel1 --output ./data/",
            "",
            "# Run analysis with specific algorithm",
            "mdive analyze --input ./data/ --algorithm sar_threshold --output ./results/",
            "",
            "# Full pipeline with laptop profile",
            "mdive run --area miami.geojson --event flood --profile laptop --output ./products/",
            "",
            "# Check workflow status",
            "mdive status --workdir ./products/",
            "",
            "# Resume interrupted workflow",
            "mdive resume --workdir ./products/",
        ]

        for line in examples:
            formatter.write_text(line)

        formatter.dedent()


pass_context = click.make_pass_decorator(MdiveContext, ensure=True)


@click.group(cls=MdiveGroup)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output (debug logging).",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Quiet mode (only warnings and errors).",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file.",
)
@click.version_option(
    version="0.1.0",
    prog_name="mdive",
    message="%(prog)s version %(version)s - Multiverse Dive CLI",
)
@click.pass_context
def app(ctx, verbose: bool, quiet: bool, config_path: Optional[Path]):
    """
    Multiverse Dive CLI - Geospatial Event Intelligence

    A situation-agnostic platform for processing satellite imagery
    and generating decision products for flood, wildfire, and storm events.
    """
    # Handle mutually exclusive verbose/quiet
    if verbose and quiet:
        raise click.UsageError("Cannot use both --verbose and --quiet")

    # Create context object
    ctx.ensure_object(dict)
    ctx.obj = MdiveContext(
        verbose=verbose,
        quiet=quiet,
        config_path=config_path,
    )


# Import and register subcommands
def register_commands():
    """Register all subcommands."""
    from cli.commands import discover, ingest, analyze, validate, export, run, status, resume

    app.add_command(discover.discover)
    app.add_command(ingest.ingest)
    app.add_command(analyze.analyze)
    app.add_command(validate.validate)
    app.add_command(export.export)
    app.add_command(run.run)
    app.add_command(status.status)
    app.add_command(resume.resume)


# Version command is handled by @click.version_option above


@app.command("info")
@pass_context
def info(ctx):
    """Display system information and configuration."""
    import platform
    import json

    click.echo("\n=== Multiverse Dive System Info ===\n")

    # Python info
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.system()} {platform.release()}")

    # Package versions
    click.echo("\n--- Package Versions ---")
    packages = ["numpy", "rasterio", "geopandas", "xarray", "click"]
    for pkg in packages:
        try:
            import importlib.metadata

            version = importlib.metadata.version(pkg)
            click.echo(f"  {pkg}: {version}")
        except Exception:
            click.echo(f"  {pkg}: not installed")

    # Configuration
    click.echo("\n--- Configuration ---")
    config = ctx.config
    click.echo(f"  Default profile: {config.get('default_profile', 'workstation')}")
    click.echo(f"  Cache directory: {config.get('cache_dir', 'N/A')}")
    click.echo(f"  Data directory: {config.get('data_dir', 'N/A')}")

    # Profiles
    click.echo("\n--- Execution Profiles ---")
    profiles = config.get("profiles", {})
    for name, settings in profiles.items():
        click.echo(
            f"  {name}: {settings.get('memory_mb', 0)}MB, "
            f"{settings.get('max_workers', 1)} workers, "
            f"{settings.get('tile_size', 256)}px tiles"
        )

    click.echo()


# Register commands when module is imported
try:
    register_commands()
except ImportError:
    # Commands not yet available - will be registered when they exist
    pass


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except Exception as e:
        logger.error(f"Error: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
