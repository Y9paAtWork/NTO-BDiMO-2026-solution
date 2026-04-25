from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution import evaluate_with_splits, fit_bundle, load_dataset, save_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=ROOT / "train.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model.pkl.gz",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_dataset(args.train_path)

    if not args.skip_validation:
        metrics = evaluate_with_splits(
            train,
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(
            "Validation "
            f"score={metrics['score_mean']:.4f}±{metrics['score_std']:.4f}, "
            f"department_f1={metrics['department_f1_mean']:.4f}, "
            f"category_accuracy={metrics['category_accuracy_mean']:.4f}"
        )

    model_bundle = fit_bundle(train)
    save_bundle(model_bundle, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
