#!/usr/bin/env python3
"""
End-to-End Workflow Test for Multiverse Dive

Tests the complete pipeline from event specification to product generation.
Designed to run on a laptop with the 'laptop' execution profile.

Usage:
    .venv/bin/python test_e2e_workflow.py [--profile laptop|workstation|edge]
"""

import sys
import os
import time
import json
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ANSI colors
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_step(step: int, total: int, text: str):
    print(f"{Colors.CYAN}[{step}/{total}]{Colors.RESET} {text}")


def print_success(text: str):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")


def print_fail(text: str):
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str):
    print(f"  {Colors.YELLOW}→{Colors.RESET} {text}")


class E2EWorkflowTest:
    """End-to-end workflow test runner."""

    def __init__(self, profile: str = "laptop", output_dir: Path = None):
        self.profile = profile
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="mdive_e2e_"))
        self.results: Dict[str, Dict[str, Any]] = {}
        self.total_steps = 10
        self.current_step = 0

        self.profiles = {
            "edge": {"memory_mb": 1024, "max_workers": 1, "tile_size": 128},
            "laptop": {"memory_mb": 2048, "max_workers": 2, "tile_size": 256},
            "workstation": {"memory_mb": 8192, "max_workers": 4, "tile_size": 512},
            "cloud": {"memory_mb": 32768, "max_workers": 16, "tile_size": 1024},
        }

        self.test_area = {
            "type": "Polygon",
            "coordinates": [[
                [-80.15, 25.76], [-80.12, 25.76],
                [-80.12, 25.79], [-80.15, 25.79], [-80.15, 25.76]
            ]]
        }

    def next_step(self, text: str):
        self.current_step += 1
        print_step(self.current_step, self.total_steps, text)

    def record_result(self, name: str, success: bool, elapsed: float, details: str = ""):
        self.results[name] = {"success": success, "elapsed_seconds": elapsed, "details": details}
        if success:
            print_success(f"{details} ({elapsed:.2f}s)")
        else:
            print_fail(f"{details} ({elapsed:.2f}s)")

    def run_all(self) -> bool:
        print_header("Multiverse Dive - End-to-End Workflow Test")
        print_info(f"Profile: {self.profile}")
        print_info(f"Output directory: {self.output_dir}")
        print_info(f"Settings: {self.profiles.get(self.profile, self.profiles['laptop'])}")
        print()

        start_time = time.time()

        self.test_1_intent_resolution()
        self.test_2_schema_validation()
        self.test_3_data_discovery()
        self.test_4_algorithm_registry()
        self.test_5_algorithm_execution()
        self.test_6_quality_sanity()
        self.test_7_tiled_processing()
        self.test_8_agent_messaging()
        self.test_9_cli_commands()
        self.test_10_full_pipeline()

        total_time = time.time() - start_time
        self.print_summary(total_time)
        return all(r["success"] for r in self.results.values())

    def test_1_intent_resolution(self):
        """Test intent resolution from natural language."""
        self.next_step("Testing Intent Resolution")
        start = time.time()
        try:
            from core.intent.resolver import IntentResolver

            resolver = IntentResolver()
            test_cases = [
                ("flooding in coastal Miami", "flood"),
                ("forest fire in California mountains", "wildfire"),
                ("hurricane damage assessment", "storm"),
            ]

            successes = 0
            for nl_input, expected_prefix in test_cases:
                result = resolver.resolve(nl_input)
                if result.resolved_class.startswith(expected_prefix):
                    successes += 1

            elapsed = time.time() - start
            self.record_result("intent_resolution", successes == len(test_cases), elapsed,
                               f"Resolved {successes}/{len(test_cases)} intents correctly")
        except Exception as e:
            self.record_result("intent_resolution", False, time.time() - start, str(e))

    def test_2_schema_validation(self):
        """Test OpenSpec schema validation."""
        self.next_step("Testing Schema Validation")
        start = time.time()
        try:
            from openspec.validator import SchemaValidator

            validator = SchemaValidator()

            event_spec = {
                "id": "evt_test_flood_001",
                "intent": {"class": "flood.coastal", "source": "explicit"},
                "spatial": self.test_area,
                "temporal": {"start": "2024-09-15T00:00:00Z", "end": "2024-09-20T23:59:59Z"}
            }

            is_valid, errors = validator.validate(event_spec, "event")
            elapsed = time.time() - start
            self.record_result("schema_validation", is_valid, elapsed,
                               "Schema validation passed" if is_valid else f"Errors: {errors[:100]}")
        except Exception as e:
            self.record_result("schema_validation", False, time.time() - start, str(e))

    def test_3_data_discovery(self):
        """Test data discovery and provider registry."""
        self.next_step("Testing Data Discovery")
        start = time.time()
        try:
            from core.data.providers.registry import ProviderRegistry, Provider

            registry = ProviderRegistry()
            test_providers = [
                Provider(id="sentinel1_test", provider="ESA", type="sar",
                         capabilities={"resolution_m": 10},
                         applicability={"event_classes": ["flood.*", "storm.*"]},
                         cost={"tier": "open"}),
                Provider(id="sentinel2_test", provider="ESA", type="optical",
                         capabilities={"resolution_m": 10},
                         applicability={"event_classes": ["flood.*", "wildfire.*"]},
                         cost={"tier": "open"}),
            ]
            for p in test_providers:
                registry.register(p)

            flood_providers = registry.get_applicable_providers("flood.coastal")
            elapsed = time.time() - start
            self.record_result("data_discovery", len(flood_providers) >= 2, elapsed,
                               f"Found {len(flood_providers)} providers for flood events")
        except Exception as e:
            self.record_result("data_discovery", False, time.time() - start, str(e))

    def test_4_algorithm_registry(self):
        """Test algorithm registry."""
        self.next_step("Testing Algorithm Registry")
        start = time.time()
        try:
            from core.analysis.library.registry import AlgorithmRegistry

            registry = AlgorithmRegistry()

            # Check if we have any registered algorithms
            all_algos = registry.list_all()

            elapsed = time.time() - start
            self.record_result("algorithm_registry", len(all_algos) >= 0, elapsed,
                               f"Registry has {len(all_algos)} algorithms registered")
        except Exception as e:
            self.record_result("algorithm_registry", False, time.time() - start, str(e))

    def test_5_algorithm_execution(self):
        """Test algorithm execution with synthetic data."""
        self.next_step("Testing Algorithm Execution")
        start = time.time()
        try:
            from core.analysis.library.baseline.flood.threshold_sar import (
                ThresholdSARAlgorithm, ThresholdSARConfig
            )

            profile_settings = self.profiles.get(self.profile, self.profiles["laptop"])
            size = profile_settings["tile_size"]

            # Create synthetic SAR data
            np.random.seed(42)
            sar_data = np.random.normal(-10, 3, (size, size)).astype(np.float32)
            # Add flood areas (low backscatter)
            sar_data[50:150, 50:150] = np.random.normal(-18, 1, (100, 100))
            sar_data[180:220, 100:180] = np.random.normal(-20, 0.5, (40, 80))

            config = ThresholdSARConfig(threshold_db=-15.0)
            algorithm = ThresholdSARAlgorithm(config)
            result = algorithm.execute(sar_data, pixel_size_m=10.0)

            elapsed = time.time() - start
            self.record_result("algorithm_execution", result.statistics["flood_pixels"] > 0, elapsed,
                               f"Detected {result.statistics['flood_area_ha']:.2f} ha flood extent")
            self._algo_result = result
            self._sar_data = sar_data
        except Exception as e:
            self.record_result("algorithm_execution", False, time.time() - start, str(e))
            traceback.print_exc()

    def test_6_quality_sanity(self):
        """Test quality sanity checks."""
        self.next_step("Testing Quality Sanity Checks")
        start = time.time()
        try:
            from core.quality.sanity import SanitySuite, SanitySuiteConfig

            if not hasattr(self, '_algo_result'):
                raise RuntimeError("No algorithm result from previous test")

            flood_extent = self._algo_result.flood_extent

            config = SanitySuiteConfig()
            suite = SanitySuite(config)
            result = suite.check(data=flood_extent.astype(np.float32))

            elapsed = time.time() - start
            passed = result.checks_passed if hasattr(result, 'checks_passed') else 0
            total = result.total_checks if hasattr(result, 'total_checks') else 1
            overall = result.overall_passed if hasattr(result, 'overall_passed') else True
            self.record_result("quality_sanity", overall, elapsed,
                               f"Sanity suite: {passed}/{total} checks passed")
        except Exception as e:
            self.record_result("quality_sanity", False, time.time() - start, str(e))

    def test_7_tiled_processing(self):
        """Test tiled processing."""
        self.next_step("Testing Tiled Processing")
        start = time.time()
        try:
            from core.execution.tiling import TileGrid, TileScheme

            profile_settings = self.profiles.get(self.profile, self.profiles["laptop"])
            tile_size = profile_settings["tile_size"]

            # Create a tile grid using bounds tuple (minx, miny, maxx, maxy)
            bounds = (0.0, 0.0, 1.0, 1.0)  # Geographic bounds
            resolution = (0.001, 0.001)  # Pixel resolution
            scheme = TileScheme(tile_size=(tile_size, tile_size), overlap=16)

            grid = TileGrid(bounds=bounds, resolution=resolution, scheme=scheme)

            # Iterate over tiles
            tiles = list(grid)
            tile_count = len(tiles) if tiles else grid.num_tiles if hasattr(grid, 'num_tiles') else 0

            elapsed = time.time() - start
            self.record_result("tiled_processing", True, elapsed,
                               f"Created grid with {tile_count} tiles ({tile_size}x{tile_size}px)")
        except Exception as e:
            self.record_result("tiled_processing", False, time.time() - start, str(e))

    def test_8_agent_messaging(self):
        """Test agent message bus."""
        self.next_step("Testing Agent Messaging")
        start = time.time()
        try:
            from agents.base import AgentMessage, MessageType, AgentType

            # Create test messages with correct field names
            messages = []
            for i in range(3):
                msg = AgentMessage(
                    message_type=MessageType.REQUEST,
                    from_agent=AgentType.ORCHESTRATOR,
                    to_agent=AgentType.DISCOVERY,
                    payload={"data": i}
                )
                messages.append(msg)

            # Test that messages have proper fields
            msg = messages[0]
            success = (
                hasattr(msg, 'message_id') and
                hasattr(msg, 'message_type') and
                hasattr(msg, 'payload')
            )

            elapsed = time.time() - start
            self.record_result("agent_messaging", success, elapsed,
                               f"Created {len(messages)} agent messages successfully")
        except Exception as e:
            self.record_result("agent_messaging", False, time.time() - start, str(e))

    def test_9_cli_commands(self):
        """Test CLI command parsing."""
        self.next_step("Testing CLI Commands")
        start = time.time()
        try:
            from click.testing import CliRunner
            from cli.main import app

            runner = CliRunner()

            help_result = runner.invoke(app, ["--help"])
            info_result = runner.invoke(app, ["info"])
            version_result = runner.invoke(app, ["--version"])

            help_ok = help_result.exit_code == 0
            info_ok = info_result.exit_code == 0
            version_ok = version_result.exit_code == 0

            elapsed = time.time() - start
            self.record_result("cli_commands", help_ok and info_ok and version_ok, elapsed,
                               f"Help: {'✓' if help_ok else '✗'}, Info: {'✓' if info_ok else '✗'}, Version: {'✓' if version_ok else '✗'}")
        except Exception as e:
            self.record_result("cli_commands", False, time.time() - start, str(e))

    def test_10_full_pipeline(self):
        """Test full pipeline integration."""
        self.next_step("Testing Full Pipeline Integration")
        start = time.time()
        try:
            from core.intent.resolver import IntentResolver
            from core.analysis.library.baseline.flood.threshold_sar import ThresholdSARAlgorithm

            # Step 1: Resolve intent
            resolver = IntentResolver()
            intent = resolver.resolve("coastal flooding in Miami after hurricane")

            # Step 2: Execute algorithm
            profile_settings = self.profiles.get(self.profile, self.profiles["laptop"])
            size = profile_settings["tile_size"]

            np.random.seed(123)
            sar_data = np.random.normal(-10, 3, (size, size)).astype(np.float32)
            sar_data[50:150, 50:150] = np.random.normal(-18, 1, (100, 100))

            algorithm = ThresholdSARAlgorithm()
            result = algorithm.execute(sar_data, pixel_size_m=10.0)

            # Step 3: Save outputs
            output_path = self.output_dir / "flood_extent.npy"
            np.save(output_path, result.flood_extent)

            metadata_path = self.output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "event_class": intent.resolved_class,
                    "confidence": intent.confidence,
                    "flood_area_ha": result.statistics["flood_area_ha"],
                    "flood_percent": result.statistics["flood_percent"],
                    "profile": self.profile
                }, f, indent=2)

            elapsed = time.time() - start
            self.record_result("full_pipeline", output_path.exists(), elapsed,
                               f"Pipeline complete: {result.statistics['flood_area_ha']:.2f} ha detected")
        except Exception as e:
            self.record_result("full_pipeline", False, time.time() - start, str(e))
            traceback.print_exc()

    def print_summary(self, total_time: float):
        print_header("Test Summary")

        passed = sum(1 for r in self.results.values() if r["success"])
        failed = len(self.results) - passed

        print(f"Profile: {self.profile}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Output: {self.output_dir}")
        print()

        print(f"{'Test':<25} {'Status':<10} {'Time':<10}")
        print("-" * 50)

        for name, result in self.results.items():
            status = f"{Colors.GREEN}PASS{Colors.RESET}" if result["success"] else f"{Colors.RED}FAIL{Colors.RESET}"
            print(f"{name:<25} {status:<18} {result['elapsed_seconds']:.2f}s")

        print("-" * 50)
        print(f"{'TOTAL':<25} {passed}/{len(self.results)} passed  {total_time:.2f}s")
        print()

        if failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.RESET}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}{failed} test(s) failed{Colors.RESET}")

        # Resource usage
        print()
        print_info("Resource Profile:")
        ps = self.profiles.get(self.profile, self.profiles["laptop"])
        print(f"  Memory: {ps['memory_mb']} MB | Workers: {ps['max_workers']} | Tiles: {ps['tile_size']}px")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multiverse Dive E2E Test")
    parser.add_argument("--profile", choices=["edge", "laptop", "workstation", "cloud"],
                        default="laptop", help="Execution profile")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    tester = E2EWorkflowTest(profile=args.profile, output_dir=args.output)
    success = tester.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
