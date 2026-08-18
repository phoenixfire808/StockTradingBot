"""ML signal pipeline — feature engineering, gradient-boosted model, walk-forward validation.

Public API:
    build_feature_frame   — assemble technical + sentiment feature frame
    GradientBoostedSignal — fit/predict_proba gradient-boosted classifier
    walk_forward_split    — time-respecting train/test split generator
    evaluate_oos          — out-of-sample classification metrics
"""

from bot.ml.features import build_feature_frame, FEATURE_COLUMNS
from bot.ml.model import GradientBoostedSignal
from bot.ml.validation import walk_forward_split, evaluate_oos

__all__ = [
    "build_feature_frame",
    "FEATURE_COLUMNS",
    "GradientBoostedSignal",
    "walk_forward_split",
    "evaluate_oos",
]
