"""Shape and checkpoint-key contracts for the rewritten flow transformer."""

import torch

from lm_x.model.modules.dit import FlowTransformer, RoutedVisionLanguageFlow, TokenRefiner


def _kwargs():
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "output_dim": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "final_dropout": False,
        "positional_embeddings": None,
        "cross_attention_dim": 8,
        "interleave_self_attention": True,
    }


def test_flow_transformer_preserves_checkpoint_parameter_paths():
    model = FlowTransformer(**_kwargs())
    keys = set(model.state_dict())
    expected = {
        "timestep_encoder.timestep_embedder.linear_1.weight",
        "timestep_encoder.timestep_embedder.linear_2.weight",
        "transformer_blocks.0.norm1.linear.weight",
        "transformer_blocks.0.attn1.to_q.weight",
        "transformer_blocks.1.attn1.to_q.weight",
        "proj_out_1.weight",
        "proj_out_2.weight",
    }
    assert expected <= keys
    assert not any("DiffusionStepEmbedding" in key for key in keys)

    output, history = model(
        hidden_states=torch.zeros(2, 3, 8),
        encoder_hidden_states=torch.zeros(2, 5, 8),
        timestep=torch.tensor([0, 1]),
        return_all_hidden_states=True,
    )
    assert output.shape == (2, 3, 8)
    assert len(history) == 3


def test_routed_flow_and_token_refiner_shapes():
    routed = RoutedVisionLanguageFlow(**_kwargs(), attend_text_every_n_blocks=1)
    output = routed(
        hidden_states=torch.zeros(1, 3, 8),
        encoder_hidden_states=torch.zeros(1, 4, 8),
        timestep=torch.tensor([0]),
        image_mask=torch.tensor([[True, True, False, False]]),
        backbone_attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    assert output.shape == (1, 3, 8)

    refiner = TokenRefiner(
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
    )
    refined, history = refiner(torch.zeros(1, 3, 8), return_all_hidden_states=True)
    assert refined.shape == (1, 3, 8)
    assert len(history) == 3
