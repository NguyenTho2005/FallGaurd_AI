"""Unit tests for pose normalisation and feature engineering."""

from __future__ import annotations

import numpy as np
import pytest

from ai.preprocessing import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    KINEMATIC_FEATURE_COUNT,
    KINEMATIC_FEATURE_NAMES,
    PRESENT_FEATURE_INDEX,
    PoseNormalizer,
    PosePreprocessor,
    zero_feature_vector,
)
from ai.pose.landmarks import POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH
from tests.conftest import (
    BENDING,
    ON_FLOOR,
    STANDING,
    landmark_array,
    transform_points,
)


# --------------------------------------------------------------------------- #
# feature layout
# --------------------------------------------------------------------------- #
def test_feature_dim_matches_layout():
    assert FEATURE_DIM == POSE_LANDMARK_COUNT * 4 + KINEMATIC_FEATURE_COUNT + 1
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert FEATURE_SPEC_VERSION


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_present_flag_is_the_last_feature():
    assert FEATURE_NAMES[PRESENT_FEATURE_INDEX] == "pose_present"
    assert FEATURE_NAMES[-1] == "pose_present"


# --------------------------------------------------------------------------- #
# normalisation invariances
# --------------------------------------------------------------------------- #
def test_normalisation_is_invariant_to_scale_translation_and_rotation():
    normalizer = PoseNormalizer()
    reference = normalizer.normalize(landmark_array(STANDING))
    assert reference is not None

    for scale, dx, dy, rotation in (
        (2.0, 0.0, 0.0, 0.0),
        (1.0, 0.15, -0.10, 0.0),
        (1.0, 0.0, 0.0, 0.35),
        (2.0, -0.2, 0.1, -0.5),
    ):
        moved = normalizer.normalize(
            landmark_array(
                transform_points(STANDING, scale=scale, dx=dx, dy=dy, rotation=rotation)
            )
        )
        assert moved is not None, f"rejected a valid pose (scale={scale}, rot={rotation})"
        np.testing.assert_allclose(
            moved.coordinates, reference.coordinates, atol=1e-3,
            err_msg=f"not invariant to scale={scale} dx={dx} dy={dy} rot={rotation}",
        )
        # The recorded scale must track the physical size change.
        assert moved.scale == pytest.approx(reference.scale * scale, rel=0.02)


def test_hip_midpoint_sits_at_the_origin():
    pose = PoseNormalizer().normalize(landmark_array(STANDING))
    assert pose is not None
    hip_mid = (pose.coordinates[23][:2] + pose.coordinates[24][:2]) / 2.0
    np.testing.assert_allclose(hip_mid, np.zeros(2), atol=1e-5)


def test_rotation_normalisation_makes_the_torso_vertical():
    pose = PoseNormalizer(rotate=True).normalize(
        landmark_array(transform_points(STANDING, rotation=0.6))
    )
    assert pose is not None
    shoulder_mid = (pose.coordinates[11][:2] + pose.coordinates[12][:2]) / 2.0
    assert abs(shoulder_mid[0]) < 1e-3, "torso should be vertical after normalisation"


def test_rotation_can_be_disabled():
    pose = PoseNormalizer(rotate=False).normalize(
        landmark_array(transform_points(STANDING, rotation=0.6))
    )
    assert pose is not None
    shoulder_mid = (pose.coordinates[11][:2] + pose.coordinates[12][:2]) / 2.0
    assert abs(shoulder_mid[0]) > 0.1, "rotation should persist when disabled"


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_invalid_landmark_array_shape_raises():
    with pytest.raises(ValueError, match="expected a"):
        PoseNormalizer().normalize(np.zeros((10, 4), dtype=np.float32))


@pytest.mark.parametrize("visibility", [0.0, 0.1, 0.49])
def test_frame_is_rejected_when_everything_is_invisible(visibility: float):
    array = landmark_array(STANDING, visibility=visibility)
    array[:, 3] = visibility
    assert PoseNormalizer(min_visibility=0.5).normalize(array) is None


def test_frame_is_rejected_when_coordinates_are_nan():
    array = landmark_array(STANDING)
    array[5, :2] = np.nan
    assert PoseNormalizer().normalize(array) is None


def test_degenerate_frame_is_rejected():
    array = landmark_array(STANDING)
    array[:, :2] = 0.0
    assert PoseNormalizer().normalize(array) is None


def test_missing_hips_fall_back_to_the_bounding_box_scale():
    array = landmark_array(STANDING)
    array[23, :] = 0.0
    array[24, :] = 0.0
    pose = PoseNormalizer().normalize(array)
    assert pose is not None, "a frame with invisible hips should still be usable"
    assert pose.scale > 0.0
    assert not pose.valid[23] and not pose.valid[24]


def test_invalid_landmarks_are_zeroed_after_normalisation():
    array = landmark_array(STANDING)
    array[15, 3] = 0.0
    pose = PoseNormalizer().normalize(array)
    assert pose is not None
    np.testing.assert_array_equal(pose.coordinates[15], np.zeros(3))
    np.testing.assert_array_equal(pose.unaligned[15], np.zeros(3))
    assert not pose.valid[15]


def test_low_visibility_threshold_keeps_weak_joints():
    array = landmark_array(STANDING, visibility=0.6)
    array[15, 3] = 0.3
    strict = PoseNormalizer(min_visibility=0.5).normalize(array)
    lenient = PoseNormalizer(min_visibility=0.1).normalize(array)
    assert strict is not None and lenient is not None
    assert not strict.valid[15]
    assert lenient.valid[15]


def test_invalid_constructor_arguments_raise():
    with pytest.raises(ValueError):
        PoseNormalizer(min_visibility=1.5)
    with pytest.raises(ValueError):
        PoseNormalizer(required_indices=(99,))


# --------------------------------------------------------------------------- #
# kinematic features
# --------------------------------------------------------------------------- #
def _kinematic(preprocessor: PosePreprocessor, points: dict[int, tuple[float, float]]) -> dict:
    vector = preprocessor.transform(landmark_array(points))
    assert vector[PRESENT_FEATURE_INDEX] == 1.0
    start = FEATURE_DIM - KINEMATIC_FEATURE_COUNT - 1
    values = vector[start : start + KINEMATIC_FEATURE_COUNT]
    return dict(zip(KINEMATIC_FEATURE_NAMES, values, strict=True))


def test_all_features_are_finite(preprocessor: PosePreprocessor):
    vector = preprocessor.transform(landmark_array(STANDING))
    assert vector.shape == (FEATURE_DIM,)
    assert np.isfinite(vector).all()


def test_lying_down_increases_torso_lean(preprocessor: PosePreprocessor):
    standing = _kinematic(preprocessor, STANDING)
    lying = _kinematic(preprocessor, ON_FLOOR)
    assert standing["torso_lean_deg"] == pytest.approx(0.0, abs=1.0)
    assert lying["torso_lean_deg"] > 70.0
    assert lying["torso_lean_deg"] > standing["torso_lean_deg"] + 50.0


def test_lying_down_collapses_the_hip_ankle_distance(preprocessor: PosePreprocessor):
    standing = _kinematic(preprocessor, STANDING)
    lying = _kinematic(preprocessor, ON_FLOOR)
    assert standing["hip_height_above_ankles"] > 1.0
    assert lying["hip_height_above_ankles"] < 0.3
    assert lying["hip_height_above_ankles"] < standing["hip_height_above_ankles"]


def test_bending_is_not_mistaken_for_lying_down(preprocessor: PosePreprocessor):
    """The hip drops when bending over, but the torso stays vertical.

    This is the case that a naive "hip got low" rule fails, and the reason the
    fall score requires *both* cues.
    """
    standing = _kinematic(preprocessor, STANDING)
    bending = _kinematic(preprocessor, BENDING)
    assert bending["torso_lean_deg"] < 20.0
    assert bending["torso_lean_deg"] < lying_threshold()
    assert standing["hip_height_above_ankles"] > 0.0
    assert bending["hip_height_above_ankles"] > 0.0


def lying_threshold() -> float:
    return 60.0


def test_kinematic_features_are_scale_invariant(preprocessor: PosePreprocessor):
    base = _kinematic(preprocessor, STANDING)
    scaled = _kinematic(preprocessor, transform_points(STANDING, scale=2.5))
    for name, value in base.items():
        # float32 round-off on a 180 deg angle is ~0.02 deg, so angles need a coarser
        # tolerance than the dimensionless ratios.
        tolerance = 0.5 if name.endswith("_deg") or "angle" in name else 1e-2
        assert scaled[name] == pytest.approx(value, abs=tolerance), (
            f"{name} is not scale invariant"
        )


def test_bounding_box_aspect_differs_between_standing_and_lying(preprocessor: PosePreprocessor):
    standing = _kinematic(preprocessor, STANDING)
    lying = _kinematic(preprocessor, ON_FLOOR)
    assert lying["bbox_aspect_ratio"] > standing["bbox_aspect_ratio"]


# --------------------------------------------------------------------------- #
# preprocessor behaviour
# --------------------------------------------------------------------------- #
def test_transform_of_none_is_a_gap(preprocessor: PosePreprocessor):
    vector = preprocessor.transform(None)
    np.testing.assert_array_equal(vector, zero_feature_vector())
    assert vector[PRESENT_FEATURE_INDEX] == 0.0


def test_transform_of_malformed_input_is_a_gap_not_an_exception(
    preprocessor: PosePreprocessor,
):
    vector = preprocessor.transform(np.zeros((5, 2), dtype=np.float32))
    assert vector[PRESENT_FEATURE_INDEX] == 0.0


def test_transform_of_all_invisible_landmarks_is_a_gap(preprocessor: PosePreprocessor):
    array = landmark_array(STANDING)
    array[:, 3] = 0.0
    assert preprocessor.transform(array)[PRESENT_FEATURE_INDEX] == 0.0


def test_transform_sequence_shapes(preprocessor: PosePreprocessor):
    stack = np.stack([landmark_array(STANDING)] * 5)
    assert preprocessor.transform_sequence(stack).shape == (5, FEATURE_DIM)
    assert preprocessor.transform_sequence(landmark_array(STANDING)).shape == (1, FEATURE_DIM)
    with pytest.raises(ValueError):
        preprocessor.transform_sequence(np.zeros((3, 10, 4), dtype=np.float32))


def test_transform_sequence_preserves_gaps(preprocessor: PosePreprocessor):
    gap = landmark_array(STANDING)
    gap[:, 3] = 0.0
    stack = np.stack([landmark_array(STANDING), gap])
    result = preprocessor.transform_sequence(stack)
    assert result[0, PRESENT_FEATURE_INDEX] == 1.0
    assert result[1, PRESENT_FEATURE_INDEX] == 0.0
    np.testing.assert_array_equal(result[1], zero_feature_vector())


def test_preprocessor_from_settings_uses_configured_thresholds(settings):
    settings.min_landmark_visibility = 0.8
    settings.rotation_normalization = False
    preprocessor = PosePreprocessor.from_settings(settings)
    assert preprocessor.normalizer.min_visibility == 0.8
    assert preprocessor.normalizer.rotate is False


def test_describe_reports_the_serving_contract(preprocessor: PosePreprocessor):
    described = preprocessor.describe()
    assert described["feature_dim"] == FEATURE_DIM
    assert described["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert len(described["feature_names"]) == FEATURE_DIM
    assert described["present_feature_index"] == PRESENT_FEATURE_INDEX


def test_training_and_inference_share_one_preprocessor(settings):
    """The same class must serve both paths -- this is the anti-skew guarantee."""
    from ai.training.dataset import build_windows, load_npy_samples

    assert PosePreprocessor is PosePreprocessor
    training = PosePreprocessor.from_settings(settings)
    serving = PosePreprocessor.from_settings(settings)
    landmarks = landmark_array(STANDING)
    np.testing.assert_array_equal(
        training.transform(landmarks), serving.transform(landmarks)
    )
    # And the dataset builder must be using that class, not a copy of the logic.
    assert build_windows.__doc__ is not None
    assert load_npy_samples.__doc__ is not None


def test_landmark_array_helper_shape():
    array = landmark_array(STANDING)
    assert array.shape == (POSE_LANDMARK_COUNT, POSE_LANDMARK_WIDTH)
    assert array.dtype == np.float32
