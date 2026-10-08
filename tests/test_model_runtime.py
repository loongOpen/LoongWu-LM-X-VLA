"""CPU regression tests for real preprocessing and the action integration loop."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from lm_x.embodiment_tags import EmbodimentTag
from lm_x.model import processing
from lm_x.model.modeling import LMXActionHead, LMXModel
from lm_x.types import LMXStepData, MessageType, ModalityConfig


def test_model_collator_initializes_lazily_and_accepts_policy_collator():
    model = LMXModel.__new__(LMXModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(model_name="unused/test", backbone_model_type="qwen")
    model._collator = None
    model._transformers_loading_kwargs = {"local_files_only": True}
    with mock.patch.object(processing, "LMXDataCollator", return_value=object()) as factory:
        first = model.collator
        assert model.collator is first
        factory.assert_called_once()
        supplied = object()
        model.collator = supplied
        assert model.collator is supplied
        assert factory.call_count == 1


def test_real_processor_normalizes_language_and_reuses_tokenizer():
    tag = next(iter(EmbodimentTag))
    configs = {
        tag.value: {
            name: ModalityConfig(delta_indices=[0], modality_keys=[key])
            for name, key in [
                ("video", "front"),
                ("state", "arm"),
                ("action", "arm"),
                ("language", "task"),
            ]
        }
    }
    stats = {
        tag.value: {
            name: {"arm": {"min": [-1, -1], "max": [1, 1], "mean": [0, 0], "std": [1, 1]}}
            for name in ("state", "action")
        }
    }
    backbone_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="right"),
        apply_chat_template=mock.Mock(return_value="processed prompt"),
    )
    with mock.patch.object(
        processing, "build_processor", return_value=backbone_processor
    ) as loader:
        processor = processing.LMXProcessor(
            modality_configs=configs,
            statistics=stats,
            embodiment_id_mapping={tag.value: 0},
            model_name="unused/test",
            image_target_size=[32, 32],
            image_crop_size=[32, 32],
            max_state_dim=2,
            max_action_dim=2,
            max_action_horizon=1,
        )
    assert loader.call_count == 1
    assert processor.collator.processor is processor.processor
    step = LMXStepData(
        images={"front": np.zeros((1, 32, 32, 3), np.uint8)},
        states={"arm": np.zeros((1, 2), np.float32)},
        text="Pick UP the Cup!",
        embodiment=tag,
    )
    result = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    assert result["state"].shape == (1, 2)
    assert result["embodiment_id"] == 0
    content = backbone_processor.apply_chat_template.call_args.args[0][0]["content"]
    assert content[-1] == {"type": "text", "text": "pick up the cup"}
    assert len(result["vlm_content"]["images"]) == 1


@pytest.mark.parametrize("uncertainty", [False, True])
def test_action_integration_matches_reference_and_caches_positions(uncertainty):
    position_embedding = torch.nn.Embedding(3, 2)
    with torch.no_grad():
        position_embedding.weight.fill_(0.2)
    head = SimpleNamespace(
        config=SimpleNamespace(action_horizon=3, add_pos_embed=True, use_alternate_vl_dit=False),
        action_dim=2,
        action_horizon=3,
        num_inference_timesteps=4,
        num_timestep_buckets=100,
        use_uncertainty=uncertainty,
        position_embedding=position_embedding,
        action_encoder=lambda action, t, emb: action + t[:, None, None],
        model=lambda hidden_states, **kwargs: hidden_states,
        action_decoder=lambda hidden, emb: hidden / 2,
        noise_decoder=lambda hidden, emb: torch.zeros_like(hidden),
        _bound_log_var=lambda value: value,
    )
    torch.manual_seed(42)
    noise = torch.randn(2, 3, 2)
    expected = noise.clone()
    for timestep in (0, 25, 50, 75):
        expected = expected + 0.25 * (expected + timestep + 0.2) / 2
    torch.manual_seed(42)
    with mock.patch.object(
        position_embedding, "forward", wraps=position_embedding.forward
    ) as lookup:
        result = LMXActionHead.get_action_with_features(
            head,
            backbone_features=torch.zeros(2, 1, 2),
            state_features=torch.zeros(2, 1, 2),
            value_output=None,
            embodiment_id=torch.zeros(2, dtype=torch.long),
            backbone_output=BatchFeature(),
        )
    torch.testing.assert_close(result["action_pred"], expected)
    assert lookup.call_count == 1
    if uncertainty:
        torch.testing.assert_close(result["action_uncertainty"], torch.full_like(noise, 1.25))
    else:
        assert "action_uncertainty" not in result
