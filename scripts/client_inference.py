"""Run one synthetic request through an existing LM-X server."""

import argparse
import os

from lm_x import LMXClient, make_sample_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--embodiment-tag", default=None)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--instruction", default="move to the target")
    args = parser.parse_args()
    with LMXClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.timeout_ms,
        embodiment_tag=args.embodiment_tag,
        api_token=os.environ.get("LMX_API_TOKEN"),
    ) as client:
        if not client.ping():
            raise RuntimeError("Inference server is unavailable")
        spec = client.get_io_spec()
        observation = make_sample_observation(spec, instruction=args.instruction)
        actions, _ = client.get_action(observation)
        for key, value in actions.items():
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
