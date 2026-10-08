"""Dependency-aware mock test for the complete policy load/inference chain."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ._support import SRC_ROOT, make_checkpoint

RUNTIME_IMPORTS = (
    "albumentations",
    "cv2",
    "diffusers",
    "huggingface_hub",
    "numpy",
    "PIL",
    "safetensors",
    "scipy",
    "torch",
    "torchvision",
    "transformers",
    "tree",
)
RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in RUNTIME_IMPORTS)


@unittest.skipUnless(
    RUNTIME_AVAILABLE,
    "Mock policy chain requires the project's runtime dependencies.",
)
class LMXPolicyMockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SRC_ROOT))

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            sys.path.remove(str(SRC_ROOT))
        except ValueError:
            pass

    def test_concrete_loaders_and_get_action_support_both_processor_layouts(self) -> None:
        import numpy as np
        import torch

        from lm_x import policy as policy_module
        from lm_x.types import ModalityConfig

        tag = next(iter(policy_module.EmbodimentTag))
        modality_configs = {
            tag.value: {
                "video": ModalityConfig(delta_indices=[0], modality_keys=["camera"]),
                "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
                "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["joints"]),
                "language": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["instruction"],
                ),
            }
        }

        class FakeModel:
            def __init__(self) -> None:
                self.eval_calls = 0
                self.requires_grad_calls: list[bool] = []
                self.to_calls: list[dict[str, object]] = []
                self.action_calls: list[dict[str, object]] = []

            def eval(self):
                self.eval_calls += 1
                return self

            def requires_grad_(self, requires_grad):
                self.requires_grad_calls.append(requires_grad)
                return self

            def to(self, **kwargs):
                self.to_calls.append(kwargs)
                return self

            def get_action(self, **kwargs):
                self.action_calls.append(kwargs)
                return {"action_pred": torch.zeros((1, 2, 1), dtype=torch.float32)}

        class FakeProcessor:
            def __init__(self) -> None:
                self.modality_configs = modality_configs
                self.state_action_processor = SimpleNamespace(
                    norm_params={
                        tag.value: {name: {"joints": {"dim": 1}} for name in ("state", "action")}
                    }
                )
                self.eval_calls = 0
                self.process_calls: list[object] = []
                self.collate_calls: list[object] = []
                self.decode_calls: list[tuple[object, object, object]] = []

            def eval(self):
                self.eval_calls += 1
                return self

            def get_modality_configs(self):
                return self.modality_configs

            def __call__(self, messages):
                self.process_calls.append(messages)
                return {"encoded_state": torch.ones((1, 1), dtype=torch.float64)}

            @property
            def collator(self):
                def collate(processed):
                    self.collate_calls.append(processed)
                    return {"inputs": {"state": torch.ones((1, 1), dtype=torch.float64)}}

                return collate

            def decode_action(self, action, embodiment_tag, states):
                self.decode_calls.append((action, embodiment_tag, states))
                return {"joints": np.full((1, 2, 1), 0.25, dtype=np.float32)}

        for processor_in_subdir in (False, True):
            with self.subTest(processor_in_subdir=processor_in_subdir):
                with tempfile.TemporaryDirectory() as tmp:
                    checkpoint = make_checkpoint(
                        Path(tmp) / "checkpoint",
                        processor_in_subdir=processor_in_subdir,
                    )
                    fake_model = FakeModel()
                    fake_processor = FakeProcessor()
                    fake_config = SimpleNamespace(
                        model_name="remote/default-backbone",
                        select_layer=16,
                    )
                    backbone_path = Path(tmp) / "local-backbone"

                    with (
                        mock.patch.object(
                            policy_module.LMXConfig,
                            "from_pretrained",
                            return_value=fake_config,
                        ) as config_loader,
                        mock.patch.object(
                            policy_module.LMXModel,
                            "from_pretrained",
                            return_value=(
                                fake_model,
                                {
                                    "missing_keys": [],
                                    "unexpected_keys": [],
                                    "mismatched_keys": [],
                                    "error_msgs": [],
                                },
                            ),
                        ) as model_loader,
                        mock.patch.object(
                            policy_module.LMXProcessor,
                            "from_pretrained",
                            return_value=fake_processor,
                        ) as processor_loader,
                    ):
                        runtime = policy_module.LMXPolicy(
                            embodiment_tag=tag,
                            model_path=checkpoint,
                            backbone_path=backbone_path,
                            device="cpu",
                            local_files_only=True,
                            dtype=torch.float32,
                        )

                        observation = {
                            "video": {
                                "camera": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
                            },
                            "state": {
                                "joints": np.zeros((1, 1, 1), dtype=np.float32),
                            },
                            "language": {"instruction": [["move safely"]]},
                        }
                        action, info = runtime.get_action(
                            observation,
                            options={
                                "embodiment_tag": tag.value,
                                "temperature": 0.2,
                            },
                        )

                    config_loader.assert_called_once()
                    model_loader.assert_called_once()
                    processor_loader.assert_called_once()
                    self.assertEqual(
                        Path(config_loader.call_args.args[0]).resolve(),
                        checkpoint.resolve(),
                    )
                    self.assertEqual(
                        Path(model_loader.call_args.args[0]).resolve(),
                        checkpoint.resolve(),
                    )
                    self.assertIs(model_loader.call_args.kwargs["config"], fake_config)
                    self.assertTrue(model_loader.call_args.kwargs["output_loading_info"])
                    self.assertEqual(fake_config.model_name, str(backbone_path))
                    self.assertTrue(
                        model_loader.call_args.kwargs["transformers_loading_kwargs"][
                            "local_files_only"
                        ]
                    )

                    expected_processor_path = (
                        checkpoint / "processor" if processor_in_subdir else checkpoint
                    )
                    self.assertEqual(
                        Path(processor_loader.call_args.args[0]).resolve(),
                        expected_processor_path.resolve(),
                    )
                    self.assertEqual(
                        processor_loader.call_args.kwargs["model_name"],
                        str(backbone_path),
                    )
                    self.assertEqual(fake_model.eval_calls, 1)
                    self.assertEqual(fake_model.requires_grad_calls, [False])
                    self.assertEqual(
                        fake_model.to_calls,
                        [{"device": "cpu", "dtype": torch.float32}],
                    )
                    self.assertEqual(fake_processor.eval_calls, 1)
                    self.assertEqual(len(fake_processor.process_calls), 1)
                    self.assertEqual(len(fake_processor.collate_calls), 1)
                    self.assertEqual(len(fake_processor.decode_calls), 1)
                    self.assertEqual(len(fake_model.action_calls), 1)
                    self.assertEqual(
                        fake_model.action_calls[0]["options"],
                        {"temperature": 0.2},
                    )
                    self.assertTrue(
                        torch.is_floating_point(fake_model.action_calls[0]["inputs"]["state"]),
                        "Collated floating-point inputs must remain floating point.",
                    )
                    np.testing.assert_array_equal(
                        action["joints"],
                        np.full((1, 2, 1), 0.25, dtype=np.float32),
                    )
                    self.assertEqual(action["joints"].dtype, np.float32)
                    self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
