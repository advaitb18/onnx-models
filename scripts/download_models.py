#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HF_USER = 'addyAIMLprojects'


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf-user",
        default=DEFAULT_HF_USER,
        help="Hugging Face username/organization",
    )

    args = parser.parse_args()

    if (
        not args.hf_user
        or args.hf_user == "YOUR_HF_USERNAME"
    ):
        raise SystemExit(
            "Specify your Hugging Face account with "
            "--hf-user USERNAME"
        )

    repos = {
        "v03h": (
            f"{args.hf_user}/"
            "xauusd-v03h-sell"
        ),
        "v05h": (
            f"{args.hf_user}/"
            "xauusd-v05h-buy"
        ),
    }

    for local_name, repo_id in repos.items():

        destination = (
            ROOT
            / "models"
            / local_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Downloading {repo_id} "
            f"-> {destination}"
        )

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=destination,
        )

    print()
    print("MODEL DOWNLOAD = COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
