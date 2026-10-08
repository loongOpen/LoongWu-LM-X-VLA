"""Regression coverage for the rewritten pose, trajectory and normalization layers."""

import numpy as np

from lm_x.data_utils import destandardize, scale_from_unit, scale_to_unit, standardize
from lm_x.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from lm_x.state_action.pose import EndEffectorPose, JointPose
from lm_x.state_action.processor import StateActionProcessor
from lm_x.types import ActionFormat, ModalityConfig


def test_pose_formats_and_relative_roundtrip():
    reference = EndEffectorPose(
        translation=[0.4, -0.2, 0.8],
        rotation=[15.0, -20.0, 35.0],
        rotation_type="euler",
        rotation_order="xyz",
    )
    target = EndEffectorPose(
        translation=[0.7, 0.1, 1.2],
        rotation=[0.1, -0.2, 0.05],
        rotation_type="rotvec",
    )

    relative = target - reference
    recovered = EndEffectorActionChunk([relative]).rebase(reference)[0]
    np.testing.assert_allclose(recovered.homogeneous, target.homogeneous, atol=1e-7)

    for action_format in (ActionFormat.XYZ_ROT6D, ActionFormat.XYZ_ROTVEC):
        encoded = EndEffectorActionChunk([target]).encode(action_format)
        decoded = EndEffectorActionChunk.from_array(encoded, action_format)[0]
        np.testing.assert_allclose(decoded.homogeneous, target.homogeneous, atol=1e-7)


def test_joint_chunks_support_relative_delta_and_interpolation():
    poses = [JointPose([0.0, 1.0]), JointPose([2.0, 3.0]), JointPose([4.0, 5.0])]
    chunk = JointActionChunk(poses, times=[0.0, 2.0, 4.0])

    relative = chunk.relative_to(JointPose([1.0, 1.0]))
    np.testing.assert_allclose(relative.to_array(), [[-1.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(relative.rebase(JointPose([1.0, 1.0])).to_array(), chunk.to_array())
    np.testing.assert_allclose(chunk.increments().to_array(), [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
    np.testing.assert_allclose(
        chunk.interpolate(times=np.array([1.0, 3.0])).to_array(),
        [[1.0, 2.0], [3.0, 4.0]],
    )


def test_numeric_transforms_preserve_degenerate_features():
    params = {
        "min": np.array([-2.0, 5.0]),
        "max": np.array([2.0, 5.0]),
        "mean": np.array([1.0, 3.0]),
        "std": np.array([2.0, 0.0]),
    }
    values = np.array([[0.0, 7.0]], dtype=np.float32)
    scaled = scale_to_unit(values, params)
    np.testing.assert_allclose(scaled, [[0.0, 0.0]])
    np.testing.assert_allclose(scale_from_unit(scaled, params), [[0.0, 5.0]])
    scores = standardize(values, params)
    np.testing.assert_allclose(scores, [[-0.5, 7.0]])
    np.testing.assert_allclose(destandardize(scores, params), values)


def test_processor_reconstructs_relative_joint_actions():
    tag = "robot"
    configs = {
        tag: {
            "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
            "action": ModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["arm"],
                action_configs=[
                    {
                        "rep": "RELATIVE",
                        "type": "NON_EEF",
                        "format": "DEFAULT",
                    }
                ],
            ),
        }
    }
    absolute_stats = {
        "min": [-2.0, -2.0],
        "max": [2.0, 2.0],
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
        "q01": [-1.0, -1.0],
        "q99": [1.0, 1.0],
    }
    relative_stats = {
        "min": [-1.0, -1.0],
        "max": [1.0, 1.0],
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
        "q01": [-0.5, -0.5],
        "q99": [0.5, 0.5],
    }
    processor = StateActionProcessor(
        configs,
        statistics={
            tag: {
                "state": {"arm": absolute_stats},
                "action": {"arm": absolute_stats},
                "relative_action": {"arm": relative_stats},
            }
        },
        use_relative_action=True,
    )
    decoded = processor.unapply_action(
        {"arm": np.zeros((1, 2, 2), dtype=np.float32)},
        tag,
        state={"arm": np.array([[[0.25, -0.5]]], dtype=np.float32)},
    )
    np.testing.assert_allclose(
        decoded["arm"],
        np.array([[[0.25, -0.5], [0.25, -0.5]]]),
    )
