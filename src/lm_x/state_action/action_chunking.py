"""Trajectory containers used to convert normalized action chunks into robot poses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from lm_x.state_action.pose import EndEffectorPose, JointPose, Pose
from lm_x.types import ActionFormat

logger = logging.getLogger(__name__)
PoseType = TypeVar("PoseType", bound=Pose)


@dataclass(frozen=True)
class _InterpolationGrid:
    source: NDArray[np.float64]
    keep: NDArray[np.int64]

    @classmethod
    def from_samples(cls, values: NDArray[np.float64]) -> _InterpolationGrid:
        source = np.asarray(values, dtype=np.float64)
        if source.ndim != 1:
            raise ValueError(f"timestamps must be one-dimensional, got {source.shape}")
        kept = [0]
        for index in range(1, len(source)):
            if source[index] > source[kept[-1]]:
                kept.append(index)
            else:
                logger.warning(
                    "Dropping timestamp %s at index %d; previous retained value is %s",
                    source[index],
                    index,
                    source[kept[-1]],
                )
        if len(kept) < 2:
            raise ValueError("Need at least 2 poses with increasing timestamps for interpolation")
        indices = np.asarray(kept, dtype=np.int64)
        return cls(source=source[indices], keep=indices)

    def query(
        self,
        *,
        num_points: int | None,
        times: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if num_points is None and times is None:
            raise ValueError("Must provide either num_points or times")
        if times is None:
            if isinstance(num_points, bool) or not isinstance(num_points, int) or num_points < 1:
                raise ValueError("num_points must be a positive integer")
            result = np.linspace(self.source[0], self.source[-1], num_points)
        else:
            result = np.asarray(times, dtype=np.float64)
            if result.ndim != 1:
                raise ValueError(f"interpolation times must be one-dimensional, got {result.shape}")
        if np.any(result < self.source[0]) or np.any(result > self.source[-1]):
            raise ValueError(
                f"Interpolation times must be within [{self.source[0]}, {self.source[-1]}]"
            )
        return result


class ActionChunk(Generic[PoseType]):
    """A typed pose sequence with an aligned time axis."""

    pose_class: type[Pose]

    def __init__(
        self,
        poses: Sequence[PoseType],
        times: Sequence[float] | NDArray[np.float64] | None = None,
    ):
        pose_list = list(poses)
        if not pose_list:
            raise ValueError("ActionChunk must contain at least one pose")
        if not all(isinstance(pose, self.pose_class) for pose in pose_list):
            raise TypeError(f"All poses must be {self.pose_class.__name__} objects")
        time_axis = (
            np.arange(len(pose_list), dtype=np.float64)
            if times is None
            else np.asarray(times, dtype=np.float64)
        )
        if time_axis.ndim != 1 or len(time_axis) != len(pose_list):
            raise ValueError("Number of times must match number of poses")
        self._poses = pose_list
        self._times = time_axis.copy()

    @property
    def poses(self) -> list[PoseType]:
        return list(self._poses)

    @property
    def times(self) -> NDArray[np.float64]:
        return self._times.copy()

    @property
    def num_poses(self) -> int:
        return len(self._poses)

    def relative_to(self, reference: PoseType | None = None) -> ActionChunk[PoseType]:
        anchor = self._poses[0] if reference is None else reference
        return self.__class__([pose - anchor for pose in self._poses], self._times)

    def increments(self, initial: PoseType | None = None) -> ActionChunk[PoseType]:
        previous = self._poses[0] if initial is None else initial
        differences: list[PoseType] = []
        for pose in self._poses:
            differences.append(pose - previous)
            previous = pose
        return self.__class__(differences, self._times)

    def relative_chunking(
        self,
        reference_frame: PoseType | None = None,
    ) -> ActionChunk[PoseType]:
        return self.relative_to(reference_frame)

    def delta_chunking(
        self,
        reference_frame: PoseType | None = None,
    ) -> ActionChunk[PoseType]:
        return self.increments(reference_frame)

    def rebase(self, reference: PoseType) -> ActionChunk[PoseType]:
        raise NotImplementedError

    def to_absolute_chunking(self, reference_frame: PoseType) -> ActionChunk[PoseType]:
        return self.rebase(reference_frame)

    def interpolate(
        self,
        num_points: int | None = None,
        times: NDArray[np.float64] | None = None,
    ) -> ActionChunk[PoseType]:
        raise NotImplementedError

    def encode(self, action_format: ActionFormat) -> NDArray[np.float64]:
        raise NotImplementedError

    def to(self, action_format: ActionFormat) -> NDArray[np.float64]:
        return self.encode(action_format)

    def __len__(self) -> int:
        return len(self._poses)

    def __getitem__(self, index: int) -> PoseType:
        return self._poses[index]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_poses={len(self)}, "
            f"time_range=[{self._times[0]:.2f}, {self._times[-1]:.2f}])"
        )


class JointActionChunk(ActionChunk[JointPose]):
    pose_class = JointPose

    def interpolate(
        self,
        num_points: int | None = None,
        times: NDArray[np.float64] | None = None,
    ) -> JointActionChunk:
        if len(self) < 2:
            raise ValueError("Need at least 2 poses for interpolation")
        grid = _InterpolationGrid.from_samples(self._times)
        query = grid.query(num_points=num_points, times=times)
        samples = np.stack([pose.joints for pose in self._poses])[grid.keep]
        values = interp1d(grid.source, samples, kind="linear", axis=0)(query)
        names = self._poses[0].joint_names
        return JointActionChunk([JointPose(value, names) for value in values], query)

    def to_array(self) -> NDArray[np.float64]:
        return np.stack([pose.joints for pose in self._poses])

    def rebase(self, reference: JointPose) -> JointActionChunk:
        expected = reference.joints.shape
        if self._poses[0].joints.shape != expected:
            raise ValueError(
                "Cannot apply relative trajectory with joint dimensions "
                f"{self._poses[0].joints.shape} and {expected}"
            )
        poses = [
            JointPose(reference.joints + pose.joints, reference.joint_names) for pose in self._poses
        ]
        return JointActionChunk(poses, self._times)

    def encode(self, action_format: ActionFormat) -> NDArray[np.float64]:
        if action_format is not ActionFormat.DEFAULT:
            raise ValueError(
                f"ActionFormat {action_format} is not supported for JointActionChunk; "
                f"expected {ActionFormat.DEFAULT}"
            )
        return self.to_array()


class EndEffectorActionChunk(ActionChunk[EndEffectorPose]):
    pose_class = EndEffectorPose

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        action_format: ActionFormat,
    ) -> EndEffectorActionChunk:
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError(f"end-effector action data must have shape (T, D), got {array.shape}")
        return cls([EndEffectorPose.from_action_format(row, action_format) for row in array])

    def interpolate(
        self,
        num_points: int | None = None,
        times: NDArray[np.float64] | None = None,
    ) -> EndEffectorActionChunk:
        if len(self) < 2:
            raise ValueError("Need at least 2 poses for interpolation")
        grid = _InterpolationGrid.from_samples(self._times)
        query = grid.query(num_points=num_points, times=times)
        matrices = self.to_homogeneous_matrices()[grid.keep]
        positions = interp1d(
            grid.source,
            matrices[:, :3, 3],
            kind="linear",
            axis=0,
        )(query)
        orientations = Slerp(
            grid.source,
            Rotation.from_matrix(matrices[:, :3, :3]),
        )(query)
        poses = [
            EndEffectorPose(position, rotation.as_matrix(), "matrix")
            for position, rotation in zip(positions, orientations)
        ]
        return EndEffectorActionChunk(poses, query)

    def to_homogeneous_matrices(self) -> NDArray[np.float64]:
        return np.stack([pose.homogeneous for pose in self._poses])

    def to_translation_rot6d(self) -> NDArray[np.float64]:
        return np.stack([pose.xyz_rot6d for pose in self._poses])

    def to_translation_rotvec(self) -> NDArray[np.float64]:
        return np.stack([pose.xyz_rotvec for pose in self._poses])

    def rebase(self, reference: EndEffectorPose) -> EndEffectorActionChunk:
        origin = reference.homogeneous
        return EndEffectorActionChunk(
            [EndEffectorPose(homogeneous=origin @ pose.homogeneous) for pose in self._poses],
            self._times,
        )

    def encode(self, action_format: ActionFormat) -> NDArray[np.float64]:
        encoders = {
            ActionFormat.DEFAULT: self.to_homogeneous_matrices,
            ActionFormat.XYZ_ROT6D: self.to_translation_rot6d,
            ActionFormat.XYZ_ROTVEC: self.to_translation_rotvec,
        }
        try:
            return encoders[action_format]()
        except KeyError as exc:
            raise ValueError(f"Unsupported action format: {action_format}") from exc
