"""Copy generated exhibits into the manuscript repository.

The analysis writes its figures and tables into this project; the manuscript
lives in its own repository and includes them. Keeping the copy as an explicit,
idempotent step means the paper is never edited by a stray write from a notebook,
and re-running an analysis updates the paper by running one command rather than by
transcribing numbers.

Only the exhibits the manuscript actually uses are copied. Everything else stays
here, where it belongs.
"""

import argparse
import os
import shutil

from regimeaware.constants import DataConstants

# Figures the manuscript includes, in the order they appear.
FIGURES = [
    "state_calibration.pdf",      # Figure 1: choice of the number of regimes
    "regime_timeline.pdf",        # Figure 2: regimes against recessions and crashes
    "recovery_mktrf.pdf",         # Figure 3: parameter recovery, one panel per factor
    "recovery_smb.pdf",
    "recovery_hml.pdf",
    "recovery_rmw.pdf",
    "recovery_cma.pdf",
    "recovery_umd.pdf",
    "benchmark_ecdf.pdf",         # Figure 4: distribution of outcomes across benchmarks
    "cost_sensitivity.pdf",       # Figure 5: performance against assumed trading cost
    "detection_horizon.pdf",      # Figure 6: how long a history the test needs
    # Appendix
    "rolling_window_sensitivity.pdf",
    "regime_conditional.pdf",
    "ablation_increments.pdf",
    "ablation_phi.pdf",
    "ablation_ecdf.pdf",
]

TABLES = [
    "table_state_selection.tex",   # Table 1: choice of the number of regimes
    "table2_performance.tex",
    "table3_significance.tex",
    "table4_decomposition.tex",
    "table5a_regime_conditional.tex",
    "table5b_forecast_quality.tex",
    "table6_detection_horizon.tex",
]


def sync(src_root, dst_root, dry_run=False):
    """Copy exhibits from the analysis project into the manuscript repository."""
    copied, missing = [], []

    for subdir, names in [("img", FIGURES), ("tables", TABLES)]:
        os.makedirs(os.path.join(dst_root, subdir), exist_ok=True)
        for name in names:
            src = os.path.join(src_root, subdir, name)
            dst = os.path.join(dst_root, subdir, name)
            if not os.path.exists(src):
                missing.append(f"{subdir}/{name}")
                continue
            if not dry_run:
                shutil.copy2(src, dst)
            copied.append(f"{subdir}/{name}")

    return copied, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscript",
        default=DataConstants.MANUSCRIPT_DIR.value,
        help="Root of the manuscript repository.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    copied, missing = sync(DataConstants.WDIR.value, args.manuscript, args.dry_run)

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {len(copied)} exhibits to {args.manuscript}")
    for name in copied:
        print(f"  {name}")

    if missing:
        print(f"\nnot yet generated ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
