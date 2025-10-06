"""Train neural guidance model from episodic memory."""

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.neural.guidance import NeuralGuidance


def main() -> None:
    parser = argparse.ArgumentParser(description="Train guidance from episodes")
    parser.add_argument("--db", default="episodes.json", help="Episode database path")
    parser.add_argument("--out", default="guidance_model.json", help="Output model path")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    guidance = NeuralGuidance()
    guidance.train_from_episode_db(args.db, epochs=args.epochs)
    guidance.save_model(args.out)
    print(f"model saved to {args.out}")


if __name__ == "__main__":
    main()
