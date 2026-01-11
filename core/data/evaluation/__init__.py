"""
Data evaluation module.

Provides constraint evaluation and multi-criteria ranking for data source selection.
"""

from core.data.evaluation.constraints import (
    HardConstraint,
    SoftConstraint,
    EvaluationResult,
    ConstraintEvaluator,
    evaluate_candidates,
    get_passing_candidates
)

from core.data.evaluation.ranking import (
    RankingCriteria,
    RankedCandidate,
    TradeOffRecord,
    MultiCriteriaRanker
)

__all__ = [
    'HardConstraint',
    'SoftConstraint',
    'EvaluationResult',
    'ConstraintEvaluator',
    'evaluate_candidates',
    'get_passing_candidates',
    'RankingCriteria',
    'RankedCandidate',
    'TradeOffRecord',
    'MultiCriteriaRanker'
]
