"""Pose normalisation and feature engineering for FallGuard AI."""

from ai.preprocessing.pose_normalizer import NormalizedPose, PoseNormalizer
from ai.preprocessing.pose_preprocessor import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    KINEMATIC_FEATURE_COUNT,
    KINEMATIC_FEATURE_NAMES,
    KINEMATIC_FEATURE_OFFSET,
    PRESENT_FEATURE_INDEX,
    PosePreprocessor,
    build_feature_vector,
    compute_kinematic_features,
    zero_feature_vector,
)

__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FEATURE_SPEC_VERSION",
    "KINEMATIC_FEATURE_COUNT",
    "KINEMATIC_FEATURE_NAMES",
    "KINEMATIC_FEATURE_OFFSET",
    "PRESENT_FEATURE_INDEX",
    "NormalizedPose",
    "PoseNormalizer",
    "PosePreprocessor",
    "build_feature_vector",
    "compute_kinematic_features",
    "zero_feature_vector",
]
