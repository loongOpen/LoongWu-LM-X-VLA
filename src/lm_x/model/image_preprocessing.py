"""Deterministic image preprocessing used by the inference processor."""

from collections.abc import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms


def apply_with_replay(
    transform: A.Compose,
    images: Sequence,
    masks: Sequence[np.ndarray] | None = None,
    replay: dict | None = None,
) -> tuple[list[torch.Tensor], None]:
    """Apply one deterministic transform independently to a temporal image sequence."""

    del replay
    if masks is not None and len(masks) != len(images):
        raise ValueError(
            f"Number of masks ({len(masks)}) must match number of images ({len(images)})"
        )

    transformed: list[torch.Tensor] = []
    for index, image in enumerate(images):
        image_array = np.asarray(image)
        kwargs = {"image": image_array}
        if masks is not None:
            kwargs["mask"] = np.asarray(masks[index])
        result = transform(**kwargs)
        image_array = result["image"]
        if image_array.dtype == np.float32:
            image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
        elif image_array.dtype != np.uint8:
            raise ValueError(f"Unexpected transformed image dtype: {image_array.dtype}")
        transformed.append(torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1))
    return transformed, None


class FractionalCenterCrop(A.DualTransform):
    """Center-crop a fixed fraction while preserving the input aspect ratio."""

    def __init__(
        self,
        crop_fraction: float = 0.9,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes, crop_coords, params["shape"]
        )

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        del data
        height, width = params["shape"][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


class LetterBoxPad(A.DualTransform):
    """Pad a non-square image to a centered square using black pixels."""

    def __init__(self, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        del params
        height, width = img.shape[:2]
        if height == width:
            return img
        size = max(height, width)
        pad_height = size - height
        pad_width = size - width
        top = pad_height // 2
        left = pad_width // 2
        return cv2.copyMakeBorder(
            img,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


class LetterBoxTransform:
    """Torchvision transform that pads the final two tensor axes to a square."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        *leading, channels, height, width = image.shape
        if height == width:
            return image
        size = max(height, width)
        pad_height = size - height
        pad_width = size - width
        top = pad_height // 2
        left = pad_width // 2
        reshaped = image.reshape(-1, channels, height, width)
        padded = transforms.functional.pad(
            reshaped,
            padding=[left, top, pad_width - left, pad_height - top],
            fill=0,
        )
        return padded.reshape(*leading, channels, size, size)


def build_albumentations_image_transform(
    image_target_size: Sequence[int] | None,
    image_crop_size: Sequence[int] | None,
    shortest_image_edge: int | None,
    crop_fraction: float | None,
    *,
    letter_box_transform: bool = False,
) -> A.Compose:
    """Build the deterministic resize, center-crop and resize pipeline."""

    if crop_fraction is None:
        if image_crop_size is None or image_target_size is None:
            raise ValueError(
                "image_crop_size and image_target_size are required when crop_fraction is None"
            )
        crop_fraction = image_crop_size[0] / image_target_size[0]
    if shortest_image_edge is None:
        if image_target_size is None:
            raise ValueError("image_target_size is required when shortest_image_edge is None")
        shortest_image_edge = image_target_size[0]

    pipeline = []
    if letter_box_transform:
        pipeline.append(LetterBoxPad())
    pipeline.extend(
        [
            A.SmallestMaxSize(max_size=shortest_image_edge, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=crop_fraction),
            A.SmallestMaxSize(max_size=shortest_image_edge, interpolation=cv2.INTER_AREA),
        ]
    )
    return A.Compose(pipeline)


def build_torchvision_image_transform(
    image_target_size: Sequence[int],
    image_crop_size: Sequence[int],
    *,
    letter_box_transform: bool = False,
) -> transforms.Compose:
    """Build the deterministic torchvision resize and center-crop pipeline."""

    pipeline = [transforms.ToImage()]
    if letter_box_transform:
        pipeline.append(LetterBoxTransform())
    pipeline.extend(
        [
            transforms.Resize(size=image_target_size),
            transforms.CenterCrop(size=image_crop_size),
            transforms.Resize(size=image_target_size),
        ]
    )
    return transforms.Compose(pipeline)


__all__ = [
    "apply_with_replay",
    "build_albumentations_image_transform",
    "build_torchvision_image_transform",
]
