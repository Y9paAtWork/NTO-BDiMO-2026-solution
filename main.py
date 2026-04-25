from __future__ import annotations

from pathlib import Path

from solution import predict_to_csv


def main() -> None:
    root = Path(__file__).resolve().parent
    model_path = root / "model.pkl.gz"
    test_path = root / "test.tsv"
    output_path = root / "prediction.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run `python3 fit/train.py` locally first."
        )

    predict_to_csv(model_path, test_path, output_path)


if __name__ == '__main__':
    main()
