"""Behavioral contracts for observations, actions and JSON-compatible specifications."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from lm_x.observation import build_io_spec, make_sample_observation
from lm_x.validation import InferenceValidationError, validate_actions, validate_observation


@pytest.fixture
def contract():
    configs = {
        "video": {"modality_keys": ["front", "wrist"], "delta_indices": [-1, 0]},
        "state": {"modality_keys": ["arm"], "delta_indices": [0]},
        "language": {"modality_keys": ["task"], "delta_indices": [0]},
        "action": {"modality_keys": ["arm"], "delta_indices": [0, 1, 2]},
    }
    norm = {name: {"arm": {"dim": np.array(7)}} for name in ("state", "action")}
    spec = build_io_spec(configs, norm, embodiment_tag="new_embodiment", image_size=(8, 12))
    return configs, norm, spec


def test_serialized_spec_builds_batched_observation(contract):
    configs, norm, spec = contract
    observation = make_sample_observation(json.loads(json.dumps(spec)), batch_size=2)
    assert validate_observation(observation, configs, norm) == 2
    assert observation["video"]["front"].shape == (2, 2, 8, 12, 3)
    assert observation["state"]["arm"].shape == (2, 1, 7)
    assert observation["language"]["task"] == [["move to the target"], ["move to the target"]]


@pytest.mark.parametrize(
    "modality,key,value",
    [
        ("video", "front", None),
        ("video", "front", np.array(1, dtype=np.uint8)),
        ("video", "front", np.zeros((0, 2, 8, 12, 3), dtype=np.uint8)),
        ("video", "front", np.zeros((1, 1, 8, 12, 3), dtype=np.uint8)),
        ("video", "front", np.zeros((1, 2, 0, 12, 3), dtype=np.uint8)),
        ("video", "front", np.zeros((1, 2, 8, 12, 4), dtype=np.uint8)),
        ("video", "front", np.zeros((1, 2, 8, 12, 3), dtype=np.float32)),
        ("state", "arm", np.zeros((2, 1, 7), dtype=np.float32)),
        ("state", "arm", np.zeros((1, 1, 6), dtype=np.float32)),
        ("state", "arm", np.full((1, 1, 7), np.nan, dtype=np.float32)),
        ("state", "arm", np.full((1, 1, 7), np.inf, dtype=np.float32)),
        ("language", "task", "pick up"),
        ("language", "task", []),
        ("language", "task", [["one", "two"]]),
        ("language", "task", [[1]]),
    ],
)
def test_invalid_inputs_fail_before_model(contract, modality, key, value):
    configs, norm, spec = contract
    observation = make_sample_observation(spec)
    observation[modality][key] = value
    with pytest.raises(InferenceValidationError):
        validate_observation(observation, configs, norm)


def test_rejects_unknown_or_missing_streams(contract):
    configs, norm, spec = contract
    for key in ("front", "unknown"):
        observation = make_sample_observation(spec)
        if key == "front":
            observation["video"].pop(key)
        else:
            observation["video"][key] = observation["video"]["front"]
        with pytest.raises(InferenceValidationError, match="keys do not match"):
            validate_observation(observation, configs, norm)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((1, 3, 6), dtype=np.float32),
        np.zeros((2, 3, 7), dtype=np.float32),
        np.zeros((1, 2, 7), dtype=np.float32),
        np.zeros((1, 3, 7), dtype=np.float64),
        np.full((1, 3, 7), np.inf, dtype=np.float32),
    ],
)
def test_actions_validate_dimensions_batch_dtype_and_finiteness(contract, value):
    configs, norm, _ = contract
    with pytest.raises(InferenceValidationError):
        validate_actions({"arm": value}, configs["action"], norm, batch_size=1)


def test_valid_actions_return_shapes(contract):
    configs, norm, _ = contract
    assert validate_actions({"arm": np.zeros((2, 3, 7), np.float32)}, configs["action"], norm) == {
        "arm": (2, 3, 7)
    }


@pytest.mark.parametrize("value", [None, 0, -1, 2.5, float("inf"), True])
def test_invalid_statistics_fail_explicitly(contract, value):
    configs, norm, _ = contract
    broken = copy.deepcopy(norm)
    broken["state"]["arm"]["dim"] = value
    with pytest.raises(InferenceValidationError):
        build_io_spec(configs, broken, embodiment_tag="new_embodiment")


def test_python_optimized_mode_preserves_validation(contract):
    configs, norm, spec = contract
    code = f"""
import json
from lm_x.observation import make_sample_observation
from lm_x.validation import validate_observation, InferenceValidationError
spec = json.loads({json.dumps(spec)!r})
obs = make_sample_observation(spec)
obs["state"]["arm"] = None
try:
    validate_observation(obs, {configs!r}, {{"state": {{"arm": {{"dim": 7}}}}}})
except InferenceValidationError:
    pass
else:
    raise RuntimeError("Validation disappeared under -O")
"""
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
