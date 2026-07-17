"""Central pipeline coordinator (CLI entry point).

Ties together every backend stage of the AI-Powered Medical Image Analysis
System. Run the full sequence end-to-end, or invoke individual stages:

    python main.py all            # verify -> preprocess -> train -> evaluate
    python main.py verify         # audit the local dataset
    python main.py preprocess     # save CLAHE-enhanced scans
    python main.py train          # train & checkpoint the model
    python main.py evaluate       # metrics + plots on the test split
    python main.py predict PATH   # diagnose a single image + Grad-CAM

The script adds ``src/`` to the import path so the modular pipeline can be
imported regardless of the current working directory.
"""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import _env                              # noqa: E402  (must precede TF imports)
import evaluate as evaluate_mod          # noqa: E402
import preprocess as preprocess_mod      # noqa: E402
import predict as predict_mod            # noqa: E402
import train as train_mod                # noqa: E402
import verify_dataset as verify_mod      # noqa: E402
from logger import get_logger            # noqa: E402

logger = get_logger(__name__)


def run_verify() -> None:
    verify_mod.main()


def run_preprocess() -> None:
    preprocess_mod.main()


def run_train() -> None:
    train_mod.train()


def run_evaluate() -> None:
    evaluate_mod.evaluate()


def run_predict(path: str) -> None:
    target = Path(path)
    if not target.exists():
        raise SystemExit(f"Image not found: {target}")
    result = predict_mod.predict_and_record(target)
    logger.info("=== DIAGNOSIS ===")
    logger.info("File       : %s", target.name)
    logger.info("Prediction : %s", result["class_name"])
    logger.info("Confidence : %.4f", result["confidence"])
    logger.info("Latency    : %.1f ms", result["latency_ms"])


def run_all() -> None:
    logger.info(">>> [1/4] Verifying dataset")
    run_verify()
    logger.info(">>> [2/4] Preprocessing (saving enhanced scans)")
    run_preprocess()
    logger.info(">>> [3/4] Training model")
    run_train()
    logger.info(">>> [4/4] Evaluating model")
    run_evaluate()
    logger.info("Pipeline complete. Ready for dashboard / prediction.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Powered Medical Image Analysis - pipeline CLI"
    )
    parser.add_argument(
        "stage",
        choices=["all", "verify", "preprocess", "train", "evaluate", "predict"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help="Image path (required for the `predict` stage)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "all":
        run_all()
    elif args.stage == "verify":
        run_verify()
    elif args.stage == "preprocess":
        run_preprocess()
    elif args.stage == "train":
        run_train()
    elif args.stage == "evaluate":
        run_evaluate()
    elif args.stage == "predict":
        if not args.path:
            raise SystemExit("The `predict` stage requires an image path.")
        run_predict(args.path)


if __name__ == "__main__":
    main()
