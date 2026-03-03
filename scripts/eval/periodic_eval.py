from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runner", default="niah")
    args = parser.parse_args()

    print(f"periodic_eval_hook checkpoint={args.checkpoint} runner={args.runner}")


if __name__ == "__main__":
    main()
