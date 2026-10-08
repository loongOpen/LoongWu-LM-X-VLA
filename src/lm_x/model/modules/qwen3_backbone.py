"""Cosmos / Qwen3-VL vision-language backbone wrapper for LM-X inference."""

from __future__ import annotations

import logging
import os
from types import MethodType
from typing import Any

import torch
from huggingface_hub.errors import GatedRepoError
from transformers.feature_extraction_utils import BatchFeature

logger = logging.getLogger(__name__)

try:
    from transformers import Qwen3VLForConditionalGeneration as _VLModel

    _VL_STACK_READY = True
except ImportError:
    _VL_STACK_READY = False
    _VLModel = None  # type: ignore[misc, assignment]

_PERF = os.environ.get("LMX_QWEN_PERF", "1") == "1"
_FAST_ROPE = os.environ.get("LMX_VECTORIZED_ROPE", "1") == "1"

_ACCESS_HINT = (
    "Cannot download the VLM backbone '{name}', which is a gated Hugging Face repo. "
    "This LM-X checkpoint loads that backbone as a separate resource, so inference "
    "requires access to both model repositories. Request access at "
    "https://huggingface.co/{name} and authenticate with `hf auth login` "
    "(or set HF_TOKEN) before loading the LM-X model."
)
_ACCESS_FRAGMENTS = ("gated repo", "is restricted", "access to model", "401 client error")


def _caused_by_gated_repo(error: BaseException) -> bool:
    node: BaseException | None = error
    for _ in range(10):
        if node is None:
            return False
        text = str(node).lower()
        if isinstance(node, GatedRepoError) or any(frag in text for frag in _ACCESS_FRAGMENTS):
            return True
        node = node.__cause__ or node.__context__
    return False


def _first_material_device(module: torch.nn.Module) -> torch.device:
    for bucket in (module.parameters(), module.buffers()):
        for tensor in bucket:
            if tensor.device.type != "meta":
                return tensor.device
    return torch.device("cpu")


def _rebuild_vision_inv_freq(
    rotary: torch.nn.Module, half_head: int, device: torch.device
) -> torch.Tensor:
    with torch.device(device):
        clone = type(rotary)(half_head)
    return clone.inv_freq.detach().to(device=device, dtype=torch.float32)


def _rebuild_text_inv_freq(
    rotary: torch.nn.Module, config: Any, device: torch.device
) -> tuple[torch.Tensor, float]:
    with torch.device(device):
        clone = type(rotary)(config=config, device=device)
    inv = clone.inv_freq.detach().to(device=device, dtype=torch.float32)
    scale = float(getattr(clone, "attention_scaling", 1.0))
    return inv, scale


def _write_inv_freq(
    rotary: torch.nn.Module, attr: str, value: torch.Tensor, *, persistent: bool
) -> bool:
    existing = getattr(rotary, attr, None)
    if (
        isinstance(existing, torch.Tensor)
        and existing.device.type != "meta"
        and existing.device == value.device
        and existing.shape == value.shape
        and existing.dtype == value.dtype
        and torch.equal(existing, value)
    ):
        return False
    if attr in rotary._buffers:
        rotary.register_buffer(attr, value, persistent=persistent)
    else:
        setattr(rotary, attr, value)
    return True


def _image_only_mrope(
    inner_model,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    video_grid_thw=None,
    attention_mask: torch.Tensor | None = None,
):
    """Fast mRoPE for uniform left-padded image batches (no video)."""
    if video_grid_thw is not None:
        raise NotImplementedError("vectorized RoPE does not support video inputs")

    config = inner_model.config
    merge = config.vision_config.spatial_merge_size
    image_id = config.image_token_id
    device, dtype = input_ids.device, input_ids.dtype
    batch, seq_len = input_ids.shape
    mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

    if image_grid_thw.shape[0] % batch:
        raise ValueError("image count must be divisible by batch size")
    images_per = image_grid_thw.shape[0] // batch
    if images_per == 0:
        raise ValueError("vectorized RoPE requires at least one image per sample")

    grid = image_grid_thw.long()
    llm_t, llm_h, llm_w = grid[:, 0], grid[:, 1] // merge, grid[:, 2] // merge
    grid_peak = torch.maximum(torch.maximum(llm_t - 1, llm_h - 1), llm_w - 1).view(batch, images_per)

    mirror = getattr(inner_model, "_vla_grid_thw_cpu", None)
    if mirror is None:
        raise RuntimeError("vectorized RoPE requires a CPU image-grid mirror")
    t0 = int(mirror[0, 0])
    h0 = int(mirror[0, 1] // merge)
    w0 = int(mirror[0, 2] // merge)

    t_axis = torch.arange(t0, device=device).view(-1, 1).expand(-1, h0 * w0).reshape(-1)
    h_axis = torch.arange(h0, device=device).view(1, -1, 1).expand(t0, -1, w0).reshape(-1)
    w_axis = torch.arange(w0, device=device).view(1, 1, -1).expand(t0, h0, -1).reshape(-1)
    grid_template = torch.stack((t_axis, h_axis, w_axis))

    is_img = input_ids.eq(image_id)
    prev = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=device), is_img[:, :-1]), dim=1)
    block_idx = (is_img & ~prev).cumsum(dim=1) - 1
    starts = torch.zeros(batch, images_per, dtype=torch.long, device=device)
    lengths = torch.zeros_like(starts)
    for i in range(images_per):
        block = is_img & block_idx.eq(i)
        first = block.int().argmax(dim=1)
        starts[:, i] = torch.where(block.any(dim=1), first, torch.full_like(first, seq_len))
        lengths[:, i] = block.sum(dim=1)

    left_pad = mask.eq(0).sum(dim=1).long()
    real_len = mask.sum(dim=1).long()
    text_starts = torch.zeros_like(starts)
    text_starts[:, 0] = left_pad
    if images_per > 1:
        text_starts[:, 1:] = starts[:, :-1] + lengths[:, :-1]
    text_lens = starts - text_starts

    step = text_lens + grid_peak + 1
    img_pos0 = torch.cat(
        (torch.zeros(batch, 1, dtype=torch.long, device=device), step.cumsum(dim=1)[:, :-1]),
        dim=1,
    )
    tail_pos0 = step.cumsum(dim=1)[:, -1]
    relative = torch.arange(seq_len, device=device).unsqueeze(0)
    position_ids = torch.ones(3, batch, seq_len, dtype=dtype, device=device)

    for i in range(images_per):
        t0i, img_i, length_i, p0 = text_starts[:, i], starts[:, i], lengths[:, i], img_pos0[:, i]
        text_mask = (relative >= t0i.unsqueeze(1)) & (relative < img_i.unsqueeze(1))
        text_vals = relative - t0i.unsqueeze(1) + p0.unsqueeze(1)
        position_ids = torch.where(
            text_mask.unsqueeze(0), text_vals.unsqueeze(0).expand(3, -1, -1), position_ids
        )
        img_mask = (relative >= img_i.unsqueeze(1)) & (relative < (img_i + length_i).unsqueeze(1))
        gidx = (relative - img_i.unsqueeze(1)).clamp(0, grid_template.shape[1] - 1)
        img_vals = grid_template[:, gidx.long()] + (text_lens[:, i] + p0).view(1, batch, 1)
        position_ids = torch.where(img_mask.unsqueeze(0), img_vals, position_ids)

    tail_begin = starts[:, -1] + lengths[:, -1]
    real_end = left_pad + real_len
    tail_mask = (relative >= tail_begin.unsqueeze(1)) & (relative < real_end.unsqueeze(1))
    tail_vals = relative - tail_begin.unsqueeze(1) + tail_pos0.unsqueeze(1)
    position_ids = torch.where(
        tail_mask.unsqueeze(0), tail_vals.unsqueeze(0).expand(3, -1, -1), position_ids
    )

    last_end = (starts - left_pad.unsqueeze(1))[:, -1] + lengths[:, -1]
    tail_len = real_len - last_end
    tail_max = tail_pos0 + tail_len - 1
    image_max = img_pos0[:, -1] + text_lens[:, -1] + grid_peak[:, -1]
    peak = torch.where(tail_len > 0, tail_max, image_max)
    deltas = (peak + 1 - seq_len).to(dtype).unsqueeze(1)
    return position_ids, deltas


def _flash_attn_with_host_max_seqlen(
    self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs
):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

    length = hidden_states.shape[0]
    query, key, value = (
        self.qkv(hidden_states).reshape(length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query, key = qwen3_vl.apply_rotary_pos_emb_vision(query, key, cos, sin)
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    attn_fn = qwen3_vl.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    max_seqlen = self._vla_max_seqlen
    out, _ = attn_fn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
        **kwargs,
    )
    return self.proj(out.reshape(length, -1).contiguous())


class VisionLanguageBackbone(torch.nn.Module):
    """Frozen Qwen3-VL tower that emits backbone hidden states for the action head."""

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        select_layer: int = -1,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        transformers_loading_kwargs: dict | None = None,
    ):
        if not _VL_STACK_READY:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is unavailable. "
                "Install transformers>=4.57.0 to enable the VL backbone."
            )
        super().__init__()
        load_kwargs = dict(transformers_loading_kwargs or {})
        attn_kwargs: dict[str, Any] = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                attn_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn missing; falling back to sdpa. "
                    "Install flash-attn for better throughput."
                )
                attn_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            attn_kwargs["torch_dtype"] = torch.bfloat16

        self.model_name = model_name
        try:
            self.model = _VLModel.from_pretrained(
                model_name, **attn_kwargs, **load_kwargs
            ).eval()
        except Exception as exc:
            if _caused_by_gated_repo(exc):
                raise RuntimeError(_ACCESS_HINT.format(name=model_name)) from exc
            raise

        if select_layer > 0:
            layers = self.model.language_model.layers
            while len(layers) > select_layer:
                layers.pop(-1)
            logger.info(
                "Trimmed language tower to %d layers (select_layer=%d).",
                len(layers),
                select_layer,
            )
        self.select_layer = select_layer
        self.requires_grad_(False)
        self._repair_rope_buffers()
        self._prefer_channels_last_patch_embed()
        self._perf_enabled = _PERF
        self._rope_fast_path = _FAST_ROPE
        self._bind_sync_free_flash_attention()

    def _bind_sync_free_flash_attention(self) -> None:
        if not self._perf_enabled:
            return
        visual = getattr(self.model, "visual", None)
        installed = 0
        for block in getattr(visual, "blocks", ()):
            attn = getattr(block, "attn", None)
            cfg = getattr(attn, "config", None)
            if cfg is not None and cfg._attn_implementation == "flash_attention_2":
                attn.forward = MethodType(_flash_attn_with_host_max_seqlen, attn)
                installed += 1
        if installed:
            logger.info("Bound sync-free max_seqlen FlashAttention on %d vision blocks", installed)

    def _prefer_channels_last_patch_embed(self) -> None:
        visual = getattr(self.model, "visual", None)
        if visual is None:
            return

        def _coerce_channels_last(_module, inputs):
            if not inputs or torch.jit.is_tracing():
                return inputs
            x = inputs[0]
            if isinstance(x, torch.Tensor) and x.dim() == 5:
                return (x.contiguous(memory_format=torch.channels_last_3d), *inputs[1:])
            return inputs

        patched = 0
        for module in visual.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
                module.register_forward_pre_hook(_coerce_channels_last)
                patched += 1
        if patched:
            logger.debug(
                "channels_last_3d applied to %d Conv3d patch-embed modules.", patched
            )

    def _repair_rope_buffers(self) -> None:
        config = getattr(self.model, "config", None)
        vision_hit = self._repair_vision_rope(config)
        text_hit = self._repair_text_rope()
        logger.debug(
            "RoPE inv_freq repair finished (vision=%s, text=%s).", vision_hit, text_hit
        )

    def _repair_vision_rope(self, config) -> bool:
        visual = getattr(self.model, "visual", None)
        rotary = getattr(visual, "rotary_pos_emb", None)
        if rotary is None or not hasattr(rotary, "inv_freq"):
            raise RuntimeError(
                "Vision rotary_pos_emb/inv_freq missing; refusing to run with "
                "uninitialized non-persistent RoPE buffers."
            )
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg is None or not all(
            hasattr(vision_cfg, key) for key in ("hidden_size", "num_heads")
        ):
            raise RuntimeError(
                "vision_config.hidden_size/num_heads missing; cannot rebuild vision RoPE."
            )
        head_dim = vision_cfg.hidden_size // vision_cfg.num_heads
        device = _first_material_device(visual)
        inv = _rebuild_vision_inv_freq(rotary, head_dim // 2, device)
        return _write_inv_freq(rotary, "inv_freq", inv, persistent=False)

    def _repair_text_rope(self) -> bool:
        language = getattr(self.model, "language_model", None)
        rotary = getattr(language, "rotary_emb", None)
        text_cfg = getattr(rotary, "config", None) or getattr(language, "config", None)
        if rotary is None or not hasattr(rotary, "inv_freq") or text_cfg is None:
            raise RuntimeError(
                "Language rotary_emb/inv_freq/config missing; refusing uninitialized text RoPE."
            )
        device = _first_material_device(language)
        inv, _ = _rebuild_text_inv_freq(rotary, text_cfg, device)
        changed = _write_inv_freq(rotary, "inv_freq", inv, persistent=False)
        if hasattr(rotary, "original_inv_freq"):
            changed = (
                _write_inv_freq(rotary, "original_inv_freq", inv.clone(), persistent=False)
                or changed
            )
        return changed

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        wanted = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        model_inputs = {key: vl_input[key] for key in wanted if key in vl_input}
        grid_cpu = getattr(self, "_vla_grid_thw_cpu", None)
        core = self.model.model

        restore_image_features = None
        restore_rope = None
        restore_fast_pos = None
        restore_rot_pos = None

        if self._perf_enabled and grid_cpu is not None:
            uniform = bool((grid_cpu == grid_cpu[0:1]).all())
            max_seqlen = int((grid_cpu[:, 1] * grid_cpu[:, 2]).max())
            for block in getattr(core.visual, "blocks", ()):
                block.attn._vla_max_seqlen = max_seqlen

            restore_image_features = core.get_image_features
            split_sizes = (grid_cpu.prod(-1) // core.visual.spatial_merge_size**2).tolist()

            def _split_image_features(model, pixel_values, image_grid_thw=None):
                pixel_values = pixel_values.type(model.visual.dtype)
                embeds, deepstack = model.visual(pixel_values, grid_thw=image_grid_thw)
                return torch.split(embeds, split_sizes), deepstack

            core.get_image_features = MethodType(_split_image_features, core)
            core._vla_grid_thw_cpu = grid_cpu

            if uniform:
                visual = core.visual
                cache_key = tuple(grid_cpu.flatten().tolist())
                if getattr(self, "_vla_vit_cache_key", None) != cache_key:
                    self._vla_vit_cache_key = cache_key
                    with torch.no_grad():
                        self._vla_cached_pos = visual.fast_pos_embed_interpolate(
                            model_inputs["image_grid_thw"]
                        )
                        self._vla_cached_rotary = visual.rot_pos_emb(model_inputs["image_grid_thw"])
                restore_fast_pos = visual.fast_pos_embed_interpolate
                restore_rot_pos = visual.rot_pos_emb
                visual.fast_pos_embed_interpolate = lambda _grid: self._vla_cached_pos
                visual.rot_pos_emb = lambda _grid: self._vla_cached_rotary

            if uniform and self._rope_fast_path:
                restore_rope = core.get_rope_index
                core.get_rope_index = MethodType(_image_only_mrope, core)

        try:
            outputs = self.model(**model_inputs, output_hidden_states=True)
        finally:
            if restore_fast_pos is not None:
                core.visual.fast_pos_embed_interpolate = restore_fast_pos
                core.visual.rot_pos_emb = restore_rot_pos
            if restore_rope is not None:
                core.get_rope_index = restore_rope
            if restore_image_features is not None:
                core.get_image_features = restore_image_features

        return BatchFeature(
            data={
                "backbone_features": outputs.hidden_states[self.select_layer],
                "backbone_attention_mask": vl_input["attention_mask"] == 1,
                "image_mask": vl_input["input_ids"] == self.model.config.image_token_id,
            }
        )

