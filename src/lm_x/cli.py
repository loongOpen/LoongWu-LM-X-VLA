"""龙悟LM-X optional inference-server command line."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lm-x-serve",
        description="Load a 龙悟LM-X checkpoint and serve inference requests over ZeroMQ.",
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint directory or model ID.")
    parser.add_argument(
        "--backbone-path",
        default=None,
        help="Optional local directory or model ID for the separately stored backbone.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network downloads and use only local or cached files.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional gated-resource token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Floating-point dtype used by inference parameters.",
    )
    parser.add_argument(
        "--embodiment-tag",
        default=None,
        help="Default checkpoint embodiment tag. Clients may provide one per request.",
    )
    parser.add_argument("--device", default="cuda:0", help="PyTorch device used for inference.")
    parser.add_argument("--host", default="127.0.0.1", help="Address on which to bind.")
    parser.add_argument("--port", type=_port, default=5555, help="TCP port on which to bind.")
    parser.add_argument(
        "--api-token",
        default=os.environ.get("LMX_API_TOKEN"),
        help="Optional request token. Defaults to LMX_API_TOKEN.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable input and output validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch

        from . import LMXPolicy, LMXServer
    except ImportError as exc:
        if isinstance(exc.__cause__, ModuleNotFoundError) and exc.__cause__.name in {
            "msgpack",
            "zmq",
        }:
            raise SystemExit(
                "Server dependencies are not installed. Run: uv sync --extra server"
            ) from exc
        raise

    print(f"Loading 龙悟LM-X checkpoint from {args.model_path} on {args.device}...")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    policy = LMXPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        backbone_path=args.backbone_path,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        token=args.hf_token,
        dtype=dtype,
        device=args.device,
        strict=args.strict,
    )

    endpoint = f"tcp://{args.host}:{args.port}"
    try:
        with LMXServer(
            policy=policy,
            host=args.host,
            port=args.port,
            api_token=args.api_token,
        ) as server:
            print(f"龙悟LM-X inference server listening on {endpoint}")
            server.run()
    except KeyboardInterrupt:
        print("Inference server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
