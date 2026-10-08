"""Small immutable-by-convention pose objects used while decoding actions."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from lm_x.types import ActionFormat

PoseT = TypeVar("PoseT", bound="Pose")


class RotationType(Enum):
    QUAT = "quat"
    EULER = "euler"
    ROTVEC = "rotvec"
    MATRIX = "matrix"
    ROT6D = "rot6d"


class EulerOrder(Enum):
    XYZ = "xyz"
    ZYX = "zyx"
    XZY = "xzy"
    YXZ = "yxz"
    YZX = "yzx"
    ZXY = "zxy"


class QuatOrder(Enum):
    WXYZ = "wxyz"
    XYZW = "xyzw"


class _RigidTransform:
    @staticmethod
    def matrix(value: np.ndarray) -> NDArray[np.float64]:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (4, 4):
            raise ValueError(f"homogeneous transform must have shape (4, 4), got {result.shape}")
        return result

    @classmethod
    def inverse(cls, value: np.ndarray) -> NDArray[np.float64]:
        transform = cls.matrix(value)
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = transform[:3, :3].T
        inverse[:3, 3] = -(inverse[:3, :3] @ transform[:3, 3])
        return inverse

    @classmethod
    def displacement(cls, origin: np.ndarray, target: np.ndarray) -> NDArray[np.float64]:
        return cls.inverse(origin) @ cls.matrix(target)


class _RotationCodec:
    @staticmethod
    def _kind(value: str) -> RotationType:
        try:
            return RotationType(value.lower())
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"Unknown rotation type: {value}") from exc

    @staticmethod
    def _quat_order(value: str | None) -> QuatOrder:
        try:
            return QuatOrder.WXYZ if value is None else QuatOrder(value.lower())
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"Unknown quaternion order: {value}") from exc

    @staticmethod
    def _euler_order(value: str | None) -> EulerOrder:
        try:
            return EulerOrder.XYZ if value is None else EulerOrder(value.lower())
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"Unknown Euler order: {value}") from exc

    @staticmethod
    def matrix_from_6d(value: np.ndarray) -> NDArray[np.float64]:
        rows = np.asarray(value, dtype=np.float64).reshape(2, 3)
        first_norm = np.linalg.norm(rows[0])
        if np.isclose(first_norm, 0):
            raise ValueError("rot6d first row must be non-zero")
        first = rows[0] / first_norm
        second = rows[1] - np.dot(first, rows[1]) * first
        second_norm = np.linalg.norm(second)
        if np.isclose(second_norm, 0):
            raise ValueError("rot6d rows must be linearly independent")
        second /= second_norm
        return np.vstack((first, second, np.cross(first, second)))

    @staticmethod
    def matrix_to_6d(value: np.ndarray) -> NDArray[np.float64]:
        return np.asarray(value, dtype=np.float64)[:2].reshape(6).copy()

    @classmethod
    def decode(
        cls,
        value: np.ndarray | list,
        kind: str,
        order: str | None,
        *,
        degrees: bool,
    ) -> Rotation:
        array = np.asarray(value, dtype=np.float64)
        rotation_type = cls._kind(kind)
        if rotation_type is RotationType.QUAT:
            quaternion = array
            if cls._quat_order(order) is QuatOrder.WXYZ:
                quaternion = array[[1, 2, 3, 0]]
            return Rotation.from_quat(quaternion)
        if rotation_type is RotationType.EULER:
            return Rotation.from_euler(cls._euler_order(order).value, array, degrees=degrees)
        if rotation_type is RotationType.ROTVEC:
            return Rotation.from_rotvec(array)
        if rotation_type is RotationType.MATRIX:
            return Rotation.from_matrix(array)
        return Rotation.from_matrix(cls.matrix_from_6d(array))

    @classmethod
    def encode(
        cls,
        rotation: Rotation,
        kind: str,
        order: str | None,
        *,
        degrees: bool,
    ) -> NDArray[np.float64]:
        rotation_type = cls._kind(kind)
        if rotation_type is RotationType.QUAT:
            xyzw = rotation.as_quat()
            return xyzw[[3, 0, 1, 2]] if cls._quat_order(order) is QuatOrder.WXYZ else xyzw
        if rotation_type is RotationType.EULER:
            return rotation.as_euler(cls._euler_order(order).value, degrees=degrees)
        if rotation_type is RotationType.ROTVEC:
            return rotation.as_rotvec()
        matrix = rotation.as_matrix()
        return matrix if rotation_type is RotationType.MATRIX else cls.matrix_to_6d(matrix)


class Pose:
    pose_type: str

    def __sub__(self: PoseT, other: PoseT) -> PoseT:
        if type(self) is not type(other):
            raise TypeError(
                "Cannot compute a displacement between "
                f"{type(self).__name__} and {type(other).__name__}"
            )
        return self._relative_to(other)

    def _relative_to(self: PoseT, other: PoseT) -> PoseT:
        raise NotImplementedError

    def copy(self: PoseT) -> PoseT:
        raise NotImplementedError


class JointPose(Pose):
    pose_type = "joint"

    def __init__(self, joints: list | np.ndarray, joint_names: list[str] | None = None):
        self.joints = np.asarray(joints, dtype=np.float64).copy()
        names = (
            [f"joint_{index}" for index in range(len(self.joints))]
            if joint_names is None
            else list(joint_names)
        )
        if len(names) != len(self.joints):
            raise ValueError(
                f"Number of joint names ({len(names)}) must match "
                f"number of joints ({len(self.joints)})"
            )
        self.joint_names = names

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.joints))

    def _relative_to(self, other: JointPose) -> JointPose:
        if self.joints.shape != other.joints.shape:
            raise ValueError(
                "Cannot compute relative joint pose with dimensions "
                f"{self.joints.shape} and {other.joints.shape}"
            )
        return JointPose(self.joints - other.joints, self.joint_names)

    def copy(self) -> JointPose:
        return JointPose(self.joints, self.joint_names)

    def __repr__(self) -> str:
        if len(self.joints) <= 6:
            values = np.array2string(self.joints, precision=4, suppress_small=True)
        else:
            values = (
                f"[{self.joints[0]:.4f}, ..., {self.joints[-1]:.4f}] ({len(self.joints)} joints)"
            )
        return f"JointPose(joints={values})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, JointPose)
            and self.joint_names == other.joint_names
            and np.allclose(self.joints, other.joints)
        )

    def __getitem__(self, index):
        return self.joints[index]

    def __len__(self) -> int:
        return len(self.joints)


class EndEffectorPose(Pose):
    pose_type = "end_effector"

    def __init__(
        self,
        translation: list | np.ndarray | None = None,
        rotation: list | np.ndarray | None = None,
        rotation_type: str | None = None,
        rotation_order: str | None = None,
        homogeneous: np.ndarray | None = None,
        degrees: bool = True,
    ):
        self._matrix_cache: NDArray[np.float64] | None = None
        if homogeneous is not None:
            transform = _RigidTransform.matrix(homogeneous)
            self._position = transform[:3, 3].copy()
            self._orientation = Rotation.from_matrix(transform[:3, :3])
            return

        self._position = (
            np.zeros(3, dtype=np.float64)
            if translation is None
            else np.asarray(translation, dtype=np.float64).copy()
        )
        if self._position.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {self._position.shape}")
        if rotation is None:
            self._orientation = Rotation.identity()
        else:
            if rotation_type is None:
                raise ValueError("rotation_type must be specified when rotation is provided")
            self._orientation = _RotationCodec.decode(
                rotation,
                rotation_type,
                rotation_order,
                degrees=degrees,
            )

    @property
    def translation(self) -> NDArray[np.float64]:
        return self._position.copy()

    @property
    def quat_wxyz(self) -> NDArray[np.float64]:
        return self.to_rotation("quat", "wxyz")

    @property
    def quat_xyzw(self) -> NDArray[np.float64]:
        return self.to_rotation("quat", "xyzw")

    @property
    def euler_xyz(self) -> NDArray[np.float64]:
        return self.to_rotation("euler", "xyz")

    @property
    def rotvec(self) -> NDArray[np.float64]:
        return self.to_rotation("rotvec")

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        return self.to_rotation("matrix")

    @property
    def rot6d(self) -> NDArray[np.float64]:
        return self.to_rotation("rot6d")

    @property
    def xyz_rot6d(self) -> NDArray[np.float64]:
        return np.concatenate((self._position, self.rot6d))

    @property
    def xyz_rotvec(self) -> NDArray[np.float64]:
        return np.concatenate((self._position, self.rotvec))

    @property
    def homogeneous(self) -> NDArray[np.float64]:
        if self._matrix_cache is None:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = self._orientation.as_matrix()
            transform[:3, 3] = self._position
            self._matrix_cache = transform
        return self._matrix_cache.copy()

    def to_rotation(
        self,
        rotation_type: str,
        rotation_order: str | None = None,
        degrees: bool = True,
    ) -> NDArray[np.float64]:
        return _RotationCodec.encode(
            self._orientation,
            rotation_type,
            rotation_order,
            degrees=degrees,
        )

    def to_homogeneous(self) -> NDArray[np.float64]:
        return self.homogeneous

    def set_rotation(
        self,
        rotation: list | np.ndarray,
        rotation_type: str,
        rotation_order: str | None = None,
        degrees: bool = True,
    ) -> None:
        self._orientation = _RotationCodec.decode(
            rotation,
            rotation_type,
            rotation_order,
            degrees=degrees,
        )
        self._matrix_cache = None

    def _relative_to(self, other: EndEffectorPose) -> EndEffectorPose:
        return EndEffectorPose(
            homogeneous=_RigidTransform.displacement(other.homogeneous, self.homogeneous)
        )

    @classmethod
    def from_action_format(
        cls,
        data: np.ndarray,
        action_format: ActionFormat,
    ) -> EndEffectorPose:
        value = np.asarray(data, dtype=np.float64)
        decoders = {
            ActionFormat.XYZ_ROT6D: lambda: cls(value[:3], value[3:], "rot6d"),
            ActionFormat.XYZ_ROTVEC: lambda: cls(value[:3], value[3:], "rotvec"),
            ActionFormat.DEFAULT: lambda: cls(homogeneous=value.reshape(4, 4)),
        }
        try:
            return decoders[action_format]()
        except KeyError as exc:
            raise ValueError(f"Unsupported ActionFormat: {action_format}") from exc

    def copy(self) -> EndEffectorPose:
        return EndEffectorPose(
            translation=self._position,
            rotation=self._orientation.as_quat(),
            rotation_type="quat",
            rotation_order="xyzw",
        )

    def __repr__(self) -> str:
        return (
            f"EndEffectorPose(translation={self.translation}, rotation_quat_wxyz={self.quat_wxyz})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, EndEffectorPose)
            and np.allclose(self._position, other._position)
            and np.allclose(self._orientation.as_quat(), other._orientation.as_quat())
        )
