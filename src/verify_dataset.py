"""Dataset audit tool.

Audits the local directory structure under ``data/raw/`` and reports image
counts per split (train/val/test) and per class (NORMAL/PNEUMONIA). This is
the first guardrail in the pipeline: it confirms the dataset is present and
correctly organised before any downstream preprocessing or training runs.
"""

from pathlib import Path

import numpy as np

from config import (
    CLASS_NAMES,
    RAW_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
)
from logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit_split(split_dir: Path) -> dict:
    """Return a dict of ``{class_name: image_count}`` for a split directory."""
    result = {}
    for class_name in CLASS_NAMES:
        result[class_name] = _count_images(split_dir / class_name)
    return result


def verify_dataset() -> dict:
    """Walk every split and class and return a structured report.

    Returns a dict with keys ``splits`` (per-split per-class counts) and a
    boolean ``ready`` flag indicating whether all expected splits/classes were
    found with at least one image.
    """
    splits = {
        "train": TRAIN_DIR,
        "val": VAL_DIR,
        "test": TEST_DIR,
    }

    report = {"splits": {}, "total": 0, "ready": True}

    for split_name, split_dir in splits.items():
        if not split_dir.exists():
            report["splits"][split_name] = {c: 0 for c in CLASS_NAMES}
            report["ready"] = False
            continue
        counts = audit_split(split_dir)
        report["splits"][split_name] = counts
        report["total"] += int(np.sum(list(counts.values())))
        if int(np.sum(list(counts.values()))) == 0:
            report["ready"] = False

    return report


def print_report(report: dict) -> None:
    line = "=" * 56
    if report["ready"]:
        status = "STATUS: READY - dataset found and correctly structured."
    else:
        status = (
            "STATUS: NOT READY - missing splits/classes or empty folders.\n"
            f"Expected raw root: {RAW_DIR}"
        )
    rows = []
    for split_name, counts in report["splits"].items():
        normal = counts.get("NORMAL", 0)
        pneumonia = counts.get("PNEUMONIA", 0)
        rows.append(
            f"{split_name:<10}{normal:>14}{pneumonia:>14}{normal + pneumonia:>18}"
        )
    block = "\n".join(
        [
            line,
            "DATASET AUDIT REPORT".center(56),
            line,
            f"{'Split':<10}{'NORMAL':>14}{'PNEUMONIA':>14}{'Total':>18}",
            "-" * 56,
            *rows,
            "-" * 56,
            f"{'GRAND TOTAL':<10}{'':>14}{'':>14}{report['total']:>18}",
            line,
            status,
            line,
        ]
    )
    logger.info(block)


def main() -> None:
    report = verify_dataset()
    print_report(report)
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
