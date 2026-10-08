from dataclasses import MISSING, dataclass, field

from transformers import PretrainedConfig


@dataclass
class LMXConfig(PretrainedConfig):
    """Configuration needed to reconstruct the 龙悟LM-X inference graph."""

    # Model identification
    model_type: str = "longwu_lm_x"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    select_layer: int = 12
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )
    # Optional inference branches are disabled unless checkpoint config explicitly enables
    # them. This prevents absent branch weights from being silently randomized at load time.
    use_value: bool = False
    use_uncertainty: bool = False
    value_bin_num: int = 128
    value_low: float = -1.0  # Minimum value for value prediction bins
    value_high: float = 0.0  # Maximum value for value prediction bins

    # Flow matching parameters
    num_inference_timesteps: int = 4
    num_timestep_buckets: int = 1000

    # Multi-embodiment parameters
    max_num_embodiments: int = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())
