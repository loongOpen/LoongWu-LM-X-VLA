"""Checkpoint completeness checks that run without model dependencies."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from ._support import PACKAGE_ROOT, git_lfs_pointer, make_checkpoint, write_json


def _load_checkpoint_module() -> ModuleType:
    """Load only checkpoint.py so package __init__ cannot pull in torch."""

    module_path = PACKAGE_ROOT / "checkpoint.py"
    if not module_path.is_file():
        raise AssertionError(
            "The inference runtime must provide lm_x/checkpoint.py with "
            "validate_checkpoint_manifest()."
        )
    spec = importlib.util.spec_from_file_location("_vla_checkpoint_contract", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load checkpoint validator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CheckpointManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module = _load_checkpoint_module()
        cls.validate = staticmethod(module.validate_checkpoint_manifest)
        cls.validate_loading_info = staticmethod(module.validate_model_loading_info)
        cls.error_type = module.CheckpointValidationError

    def test_accepts_materialized_monolithic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(Path(tmp) / "checkpoint")

            resolved = self.validate(checkpoint)

            self.assertEqual(resolved, checkpoint.resolve())

    def test_accepts_processor_metadata_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(
                Path(tmp) / "checkpoint",
                processor_in_subdir=True,
            )

            resolved = self.validate(checkpoint)

            self.assertEqual(resolved, checkpoint.resolve())

    def test_preserves_remote_model_identifier_for_backend_resolution(self) -> None:
        model_id = "nvidia/Cosmos-Reason2-2B"

        self.assertEqual(self.validate(model_id), model_id)

    def test_rejects_nonexistent_absolute_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = (Path(tmp) / "missing-checkpoint").resolve()

            with self.assertRaisesRegex(self.error_type, r"(?i)does not exist"):
                self.validate(missing)

    def test_rejects_config_only_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(
                Path(tmp) / "checkpoint",
                include_weights=False,
            )

            with self.assertRaisesRegex(self.error_type, r"(?i)weights"):
                self.validate(checkpoint)

    def test_rejects_weight_file_that_is_only_a_git_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(
                Path(tmp) / "checkpoint",
                lfs_weight=True,
            )

            with self.assertRaisesRegex(self.error_type, r"(?i)git lfs|pointer"):
                self.validate(checkpoint)

    def test_rejects_empty_weight_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(Path(tmp) / "checkpoint")
            (checkpoint / "model.safetensors").write_bytes(b"")

            with self.assertRaisesRegex(self.error_type, r"(?i)empty|zero"):
                self.validate(checkpoint)

    def test_rejects_processor_metadata_that_is_only_a_git_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(Path(tmp) / "checkpoint")
            (checkpoint / "statistics.json").write_bytes(git_lfs_pointer(size=128))

            with self.assertRaisesRegex(self.error_type, r"(?i)git lfs|pointer"):
                self.validate(checkpoint)

    def test_rejects_index_that_references_a_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(Path(tmp) / "checkpoint", sharded=True)
            missing_name = "model-00002-of-00002.safetensors"
            (checkpoint / missing_name).unlink()

            with self.assertRaisesRegex(self.error_type, missing_name):
                self.validate(checkpoint)

    def test_rejects_git_lfs_weight_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = make_checkpoint(
                Path(tmp) / "checkpoint",
                include_weights=False,
            )
            (checkpoint / "model.safetensors.index.json").write_bytes(git_lfs_pointer(size=2048))

            with self.assertRaisesRegex(self.error_type, r"(?i)git lfs|pointer"):
                self.validate(checkpoint)

    def test_loading_info_allows_independently_loaded_backbone_tensors(self) -> None:
        self.validate_loading_info(
            {
                "missing_keys": ["backbone.model.visual.patch_embed.proj.weight"],
                "unexpected_keys": [
                    "backbone.model.model.language_model.layers.16.mlp.down_proj.weight"
                ],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            model_reference="checkpoint",
            select_layer=16,
        )

    def test_loading_info_rejects_missing_action_or_optional_branch_tensors(self) -> None:
        for missing_key in (
            "action_head.action_decoder.layer1.W",
            "action_head.noise_decoder.layer1.W",
            "value_model.value_head.output.weight",
        ):
            with self.subTest(missing_key=missing_key):
                with self.assertRaisesRegex(self.error_type, r"(?i)missing keys"):
                    self.validate_loading_info(
                        {
                            "missing_keys": [missing_key],
                            "unexpected_keys": [],
                            "mismatched_keys": [],
                            "error_msgs": [],
                        },
                        model_reference="checkpoint",
                        select_layer=16,
                    )

    def test_loading_info_rejects_unexpected_or_mismatched_tensors(self) -> None:
        with self.assertRaisesRegex(self.error_type, r"(?i)unexpected keys|mismatched keys"):
            self.validate_loading_info(
                {
                    "missing_keys": [],
                    "unexpected_keys": ["value_model.unconfigured.weight"],
                    "mismatched_keys": ["action_head.state_encoder.layer1.W"],
                    "error_msgs": [],
                },
                model_reference="checkpoint",
                select_layer=16,
            )

    def test_rejects_malformed_or_nonobject_json_metadata(self) -> None:
        for filename in (
            "config.json",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
        ):
            for payload in (b"{broken", b"[]", b"null"):
                with self.subTest(filename=filename, payload=payload):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = make_checkpoint(Path(tmp))
                        (root / filename).write_bytes(payload)
                        with self.assertRaisesRegex(self.error_type, "JSON metadata"):
                            self.validate(root)

    def test_rejects_cross_platform_unsafe_shard_paths(self) -> None:
        for shard in ("../outside.bin", "/outside.bin", "C:\\outside.bin", "a\\..\\outside.bin"):
            with self.subTest(shard=shard):
                with tempfile.TemporaryDirectory() as tmp:
                    root = make_checkpoint(Path(tmp))
                    write_json(
                        root / "model.safetensors.index.json", {"weight_map": {"weight": shard}}
                    )
                    with self.assertRaisesRegex(self.error_type, "unsafe shard"):
                        self.validate(root)


if __name__ == "__main__":
    unittest.main()
