"""
Data broker - orchestrates discovery and selection of datasets.

Main entry point for data discovery. Coordinates discovery adapters,
evaluates constraints, ranks candidates, and returns selection decisions.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.data.discovery.base import DiscoveryAdapter, DiscoveryResult
from core.data.providers.registry import ProviderRegistry


class BrokerQuery:
    """Structured query for data discovery."""

    def __init__(
        self,
        event_id: str,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        intent_class: str,
        data_types: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        ranking_weights: Optional[Dict[str, float]] = None,
        cache_policy: str = "prefer_cache"
    ):
        self.event_id = event_id
        self.spatial = spatial
        self.temporal = temporal
        self.intent_class = intent_class
        self.data_types = data_types or []
        self.constraints = constraints or {}
        self.ranking_weights = ranking_weights or self._default_weights()
        self.cache_policy = cache_policy

    @staticmethod
    def _default_weights() -> Dict[str, float]:
        """Default ranking weights."""
        return {
            "spatial_coverage": 0.25,
            "temporal_proximity": 0.20,
            "resolution": 0.15,
            "cloud_cover": 0.15,
            "data_quality": 0.10,
            "access_cost": 0.10,
            "provider_preference": 0.05
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "spatial": self.spatial,
            "temporal": self.temporal,
            "intent_class": self.intent_class,
            "data_types": self.data_types,
            "constraints": self.constraints,
            "ranking_weights": self.ranking_weights,
            "cache_policy": self.cache_policy
        }


class BrokerResponse:
    """Structured response from data broker."""

    def __init__(
        self,
        event_id: str,
        query_timestamp: datetime,
        selected_datasets: List[Dict[str, Any]],
        selection_summary: Dict[str, Any],
        trade_off_record: Optional[List[Dict[str, Any]]] = None,
        degraded_mode: Optional[Dict[str, Any]] = None
    ):
        self.event_id = event_id
        self.query_timestamp = query_timestamp
        self.selected_datasets = selected_datasets
        self.selection_summary = selection_summary
        self.trade_off_record = trade_off_record or []
        self.degraded_mode = degraded_mode or {"active": False}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "query_timestamp": self.query_timestamp.isoformat(),
            "selected_datasets": self.selected_datasets,
            "selection_summary": self.selection_summary,
            "trade_off_record": self.trade_off_record,
            "degraded_mode": self.degraded_mode
        }


class DataBroker:
    """
    Main data broker orchestrator.

    Coordinates discovery across multiple providers, evaluates candidates
    against constraints, ranks them, and returns optimal dataset selections.
    """

    def __init__(self, provider_registry: Optional[ProviderRegistry] = None):
        self.provider_registry = provider_registry or ProviderRegistry()
        self.discovery_adapters: List[DiscoveryAdapter] = []
        self._executor = ThreadPoolExecutor(max_workers=10)

    def register_adapter(self, adapter: DiscoveryAdapter) -> None:
        """Register a discovery adapter."""
        self.discovery_adapters.append(adapter)

    async def discover(self, query: BrokerQuery) -> BrokerResponse:
        """
        Main discovery orchestration method.

        Args:
            query: Broker query with spatial/temporal/intent parameters

        Returns:
            BrokerResponse with selected datasets and metadata
        """
        query_timestamp = datetime.utcnow()

        # Get applicable providers for this event class
        applicable_providers = self.provider_registry.get_applicable_providers(
            event_class=query.intent_class,
            data_types=query.data_types
        )

        # Run discovery across all adapters in parallel
        discovery_tasks = []
        for adapter in self.discovery_adapters:
            for provider in applicable_providers:
                if adapter.supports_provider(provider):
                    task = adapter.discover(
                        provider=provider,
                        spatial=query.spatial,
                        temporal=query.temporal,
                        constraints=query.constraints.get(provider.data_type, {})
                    )
                    discovery_tasks.append(task)

        # Wait for all discovery tasks to complete
        all_results: List[DiscoveryResult] = []
        if discovery_tasks:
            results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    # Result is already a list of DiscoveryResults
                    all_results.extend(result)
                elif isinstance(result, DiscoveryResult):
                    # Result is a single DiscoveryResult
                    all_results.append(result)
                elif isinstance(result, Exception):
                    # Log error but continue with other results
                    print(f"Discovery error: {result}")

        # Evaluate and rank candidates
        evaluated_candidates = self._evaluate_candidates(all_results, query)
        ranked_candidates = self._rank_candidates(evaluated_candidates, query.ranking_weights)

        # Select optimal datasets
        selected_datasets, trade_offs, degraded_mode = self._select_datasets(
            ranked_candidates,
            query
        )

        # Build selection summary
        selection_summary = self._build_summary(
            all_results,
            selected_datasets,
            query
        )

        return BrokerResponse(
            event_id=query.event_id,
            query_timestamp=query_timestamp,
            selected_datasets=selected_datasets,
            selection_summary=selection_summary,
            trade_off_record=trade_offs,
            degraded_mode=degraded_mode
        )

    def _evaluate_candidates(
        self,
        candidates: List[DiscoveryResult],
        query: BrokerQuery
    ) -> List[Dict[str, Any]]:
        """
        Evaluate candidates against hard and soft constraints.

        Returns list of candidates with evaluation scores.
        """
        evaluated = []

        for candidate in candidates:
            # Check hard constraints
            if not self._meets_hard_constraints(candidate, query):
                continue

            # Calculate soft constraint scores
            scores = {
                "spatial_coverage": self._score_spatial_coverage(candidate, query),
                "temporal_proximity": self._score_temporal_proximity(candidate, query),
                "resolution": self._score_resolution(candidate, query),
                "cloud_cover": self._score_cloud_cover(candidate, query),
                "data_quality": self._score_data_quality(candidate),
                "access_cost": self._score_access_cost(candidate),
                "provider_preference": self._score_provider_preference(candidate)
            }

            evaluated.append({
                "candidate": candidate,
                "scores": scores
            })

        return evaluated

    def _rank_candidates(
        self,
        evaluated: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Rank candidates using weighted scoring."""
        for item in evaluated:
            # Calculate weighted total score
            total_score = sum(
                item["scores"][key] * weights.get(key, 0.0)
                for key in item["scores"]
            )
            item["total_score"] = total_score

        # Sort by total score (descending)
        return sorted(evaluated, key=lambda x: x["total_score"], reverse=True)

    def _select_datasets(
        self,
        ranked: List[Dict[str, Any]],
        query: BrokerQuery
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Select optimal datasets from ranked candidates.

        Returns:
            - selected_datasets: List of selected dataset metadata
            - trade_offs: List of trade-off decisions made
            - degraded_mode: Degraded mode status
        """
        selected = []
        trade_offs = []
        degraded_mode = {"active": False, "level": 0, "flags": []}

        # Group by data type
        by_data_type: Dict[str, List[Dict[str, Any]]] = {}
        for item in ranked:
            data_type = item["candidate"].data_type
            if data_type not in by_data_type:
                by_data_type[data_type] = []
            by_data_type[data_type].append(item)

        # Select best candidate for each required data type
        for data_type in query.data_types:
            candidates = by_data_type.get(data_type, [])

            if not candidates:
                # Missing required data type
                degraded_mode["active"] = True
                degraded_mode["level"] = max(degraded_mode["level"], 1)
                degraded_mode["flags"].append(f"missing_data_type:{data_type}")
                continue

            # Select top candidate
            best = candidates[0]
            selected.append(self._format_selected_dataset(best, query))

            # Record trade-offs if alternatives exist
            if len(candidates) > 1:
                trade_offs.append({
                    "decision": f"Select {data_type} dataset",
                    "selected": best["candidate"].dataset_id,
                    "alternatives": [c["candidate"].dataset_id for c in candidates[1:4]],
                    "rationale": f"Highest score: {best['total_score']:.3f}"
                })

        return selected, trade_offs, degraded_mode

    def _format_selected_dataset(
        self,
        evaluated: Dict[str, Any],
        query: BrokerQuery
    ) -> Dict[str, Any]:
        """Format selected dataset for response."""
        candidate = evaluated["candidate"]

        return {
            "dataset_id": candidate.dataset_id,
            "data_type": candidate.data_type,
            "provider": candidate.provider,
            "source_uri": candidate.source_uri,
            "source_format": candidate.format,
            "acquisition_time": candidate.acquisition_time,
            "spatial_coverage_percent": candidate.spatial_coverage_percent,
            "cloud_cover_percent": getattr(candidate, 'cloud_cover_percent', None),
            "resolution_m": candidate.resolution_m,
            "role": self._determine_role(candidate, query),
            "selection_score": evaluated["total_score"],
            "selection_rationale": self._generate_rationale(evaluated)
        }

    def _determine_role(self, candidate: DiscoveryResult, query: BrokerQuery) -> str:
        """Determine dataset role in analysis pipeline."""
        # Simple heuristic - can be enhanced
        temporal_extent = query.temporal
        acq_time = candidate.acquisition_time

        # Parse temporal extent
        from datetime import datetime
        start = datetime.fromisoformat(temporal_extent["start"].replace('Z', '+00:00'))

        if acq_time < start:
            return "pre_event"
        else:
            return "post_event"

    def _generate_rationale(self, evaluated: Dict[str, Any]) -> str:
        """Generate human-readable selection rationale."""
        scores = evaluated["scores"]
        top_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        rationale_parts = []
        for factor, score in top_factors:
            if score > 0.7:
                rationale_parts.append(f"{factor.replace('_', ' ')}: {score:.2f}")

        return "; ".join(rationale_parts) if rationale_parts else "Best overall match"

    def _build_summary(
        self,
        all_results: List[DiscoveryResult],
        selected: List[Dict[str, Any]],
        query: BrokerQuery
    ) -> Dict[str, Any]:
        """Build selection summary statistics."""
        data_types_covered = list(set(ds["data_type"] for ds in selected))
        data_types_missing = [dt for dt in query.data_types if dt not in data_types_covered]

        return {
            "total_candidates_evaluated": len(all_results),
            "datasets_selected": len(selected),
            "data_types_covered": data_types_covered,
            "data_types_missing": data_types_missing,
            "cache_hits": 0,  # TODO: implement cache
            "estimated_acquisition_size_gb": sum(
                ds.get("size_gb", 0) for ds in selected
            )
        }

    # Constraint checking methods

    def _meets_hard_constraints(
        self,
        candidate: DiscoveryResult,
        query: BrokerQuery
    ) -> bool:
        """Check if candidate meets all hard constraints."""
        constraints = query.constraints.get(candidate.data_type, {})

        # Spatial coverage minimum
        min_coverage = constraints.get("min_spatial_coverage", 0.5)
        if candidate.spatial_coverage_percent < min_coverage * 100:
            return False

        # Cloud cover maximum (for optical)
        if candidate.data_type == "optical":
            max_cloud = constraints.get("max_cloud_cover", 1.0)
            if hasattr(candidate, 'cloud_cover_percent'):
                if candidate.cloud_cover_percent > max_cloud * 100:
                    return False

        # Resolution maximum
        max_resolution = constraints.get("max_resolution_m", float('inf'))
        if candidate.resolution_m > max_resolution:
            return False

        return True

    # Scoring methods (0.0 to 1.0)

    def _score_spatial_coverage(self, candidate: DiscoveryResult, query: BrokerQuery) -> float:
        """Score based on spatial coverage of AOI."""
        return min(candidate.spatial_coverage_percent / 100.0, 1.0)

    def _score_temporal_proximity(self, candidate: DiscoveryResult, query: BrokerQuery) -> float:
        """Score based on temporal proximity to event."""
        from datetime import datetime

        temporal_extent = query.temporal
        reference_time = temporal_extent.get("reference_time") or temporal_extent["start"]
        ref_dt = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))

        # Calculate time difference in days
        time_diff_days = abs((candidate.acquisition_time - ref_dt).total_seconds() / 86400)

        # Score decreases with time difference (exponential decay)
        import math
        score = math.exp(-time_diff_days / 7.0)  # 7-day half-life

        return score

    def _score_resolution(self, candidate: DiscoveryResult, query: BrokerQuery) -> float:
        """Score based on spatial resolution (higher resolution = better)."""
        # Normalize resolution score (10m = 1.0, 100m = 0.5, 1000m = 0.1)
        import math
        score = math.exp(-candidate.resolution_m / 100.0)
        return min(score, 1.0)

    def _score_cloud_cover(self, candidate: DiscoveryResult, query: BrokerQuery) -> float:
        """Score based on cloud cover (lower = better)."""
        if not hasattr(candidate, 'cloud_cover_percent'):
            return 1.0  # N/A for non-optical

        # Invert cloud cover percentage
        return 1.0 - (candidate.cloud_cover_percent / 100.0)

    def _score_data_quality(self, candidate: DiscoveryResult) -> float:
        """Score based on data quality indicators."""
        # Placeholder - can be enhanced with actual quality metrics
        quality_flag = getattr(candidate, 'quality_flag', 'good')

        quality_scores = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.3
        }

        return quality_scores.get(quality_flag, 0.5)

    def _score_access_cost(self, candidate: DiscoveryResult) -> float:
        """Score based on access cost (lower = better)."""
        cost_tier = getattr(candidate, 'cost_tier', 'open')

        cost_scores = {
            'open': 1.0,
            'open_restricted': 0.7,
            'commercial': 0.3
        }

        return cost_scores.get(cost_tier, 0.5)

    def _score_provider_preference(self, candidate: DiscoveryResult) -> float:
        """Score based on provider preference."""
        # Get provider preference from registry
        provider_info = self.provider_registry.get_provider(candidate.provider)
        if provider_info:
            return provider_info.metadata.get("preference_score", 0.5)
        return 0.5
