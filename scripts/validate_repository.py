#!/usr/bin/env python3
"""Validate the committed notebook and reproducibility artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "nfl_draft_prediction_professional.ipynb"
OUTPUT_DIR = ROOT / "outputs"
INPUT_DIR = ROOT / "input"

REQUIRED_OUTPUTS = {
    "feature_importance.csv",
    "fold_results.csv",
    "model_comparison.csv",
    "oof_predictions.csv",
    "run_summary.json",
    "submission.csv",
}
REQUIRED_INPUTS = {"train.csv", "test.csv", "sample_submission.csv"}


def fail(message: str) -> None:
    raise AssertionError(message)


def validate_notebook() -> tuple[int, int]:
    if not NOTEBOOK.is_file():
        fail(f"Missing notebook: {NOTEBOOK.relative_to(ROOT)}")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        fail("Notebook must use nbformat 4.")

    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if not cells or not code_cells:
        fail("Notebook contains no executable analysis.")
    if any(cell.get("execution_count") is None for cell in code_cells):
        fail("At least one code cell has not been executed.")

    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        fail(f"Notebook contains {len(errors)} error output(s).")
    return len(cells), len(code_cells)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_outputs() -> tuple[int, float]:
    missing = sorted(name for name in REQUIRED_OUTPUTS if not (OUTPUT_DIR / name).is_file())
    if missing:
        fail(f"Missing output files: {', '.join(missing)}")

    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))
    required_summary_keys = {
        "seed", "cv_folds", "selected_candidate", "selected_members",
        "oof_roc_auc", "bootstrap_95_ci", "train_shape", "test_shape",
    }
    if not required_summary_keys.issubset(summary):
        fail("run_summary.json is missing required fields.")
    if summary["seed"] != 42 or summary["cv_folds"] != 5:
        fail("Unexpected reproducibility configuration in run_summary.json.")

    columns, rows = read_csv(OUTPUT_DIR / "submission.csv")
    if columns != ["Id", "Drafted"]:
        fail("submission.csv must have exactly the columns Id,Drafted.")
    if len(rows) != 696:
        fail(f"submission.csv must contain 696 rows, found {len(rows)}.")

    ids = [row["Id"] for row in rows]
    probabilities = [float(row["Drafted"]) for row in rows]
    if len(ids) != len(set(ids)):
        fail("submission.csv contains duplicate IDs.")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        fail("submission.csv contains invalid probabilities.")
    if len(set(probabilities)) <= 1:
        fail("submission.csv contains constant predictions.")

    return len(rows), float(summary["oof_roc_auc"])


def validate_inputs(required: bool) -> None:
    present = {path.name for path in INPUT_DIR.glob("*.csv")}
    if required and not REQUIRED_INPUTS.issubset(present):
        missing = sorted(REQUIRED_INPUTS - present)
        fail(f"Missing competition input files: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-input",
        action="store_true",
        help="also require the three locally supplied competition CSV files",
    )
    args = parser.parse_args()

    total_cells, code_cells = validate_notebook()
    submission_rows, oof_auc = validate_outputs()
    validate_inputs(args.require_input)

    print("Repository validation passed.")
    print(f"- Notebook: {total_cells} cells ({code_cells} code cells), no saved errors")
    print(f"- Outputs: {len(REQUIRED_OUTPUTS)} required artifacts")
    print(f"- Submission: {submission_rows} valid probability rows")
    print(f"- Selected OOF ROC-AUC: {oof_auc:.5f}")
    print(f"- Input files required: {'yes' if args.require_input else 'no'}")


if __name__ == "__main__":
    main()
