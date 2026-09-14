"""Copy generated exhibits into the manuscript repository.

The analysis writes its figures and tables into this project; the manuscript
lives in its own repository and includes them. Keeping the copy as an explicit,
idempotent step means the paper is never edited by a stray write from a notebook,
and re-running an analysis updates the paper by running one command rather than by
transcribing numbers.

Only the exhibits the manuscript actually uses are copied. Everything else stays
here, where it belongs.

Copying is also where the generated tables are inspected. Each is written only
when its own notebook happens to run, so they drift apart as the exporter gains
features: a panel rule that one table has and another does not, or a character
that prints as something else entirely. Three such defects reached a compiled
draft before being caught by eye, one of them invisible to the log and to text
extraction alike. The checks below are the mechanical version of noticing.
"""

import argparse
import os
import re
import shutil

from regimeaware.constants import DataConstants

EXPORTER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "core", "exhibits.py")

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
    "table_momentum_crashes.tex",
    "table2_performance.tex",
    "table3_significance.tex",
    "table4_decomposition.tex",
    "table5a_regime_conditional.tex",
    "table5b_forecast_quality.tex",
    "table6_detection_horizon.tex",
]


def inspect_table(path):
    """Defects a generated table can carry without failing to compile."""
    problems = []
    lines = open(path, encoding="utf-8").read().splitlines()
    rows = [l for l in lines if "&" in l and not l.lstrip().startswith("%")]

    # A bare angle bracket in text mode is typeset as inverted punctuation under
    # the OT1 encoding. Nothing warns, and the PDF text layer still reads it as
    # the bracket, so neither the log nor pdftotext can reveal it.
    for row in rows:
        outside_maths = row.split("$")[::2]
        if any(c in seg for seg in outside_maths for c in "<>"):
            problems.append(f"angle bracket outside maths: {row.strip()[:60]}")
            break

    # An unescaped per cent sign comments out the rest of the row, silently
    # merging it with the next one.
    for row in rows:
        if re.search(r"(?<!\\)%", row):
            problems.append(f"unescaped per cent: {row.strip()[:60]}")
            break

    # Panels are separated from their rows in every table or in none of them.
    panels = [i for i, l in enumerate(lines) if "multicolumn" in l and "Panel" in l]
    unruled = [i for i in panels if i + 1 >= len(lines)
               or "midrule" not in lines[i + 1]]
    if panels and unruled:
        problems.append(f"{len(unruled)} of {len(panels)} panel headings have no "
                        "rule beneath them")

    # Deliberately no staleness check on modification time. Every edit to the
    # exporter would flag every table, whether or not the change mattered, and a
    # warning that fires constantly is one nobody reads. The checks above name
    # the actual defect instead.
    return problems


def sync(src_root, dst_root, dry_run=False):
    """Copy exhibits from the analysis project into the manuscript repository."""
    copied, missing, flagged = [], [], {}

    for subdir, names in [("img", FIGURES), ("tables", TABLES)]:
        os.makedirs(os.path.join(dst_root, subdir), exist_ok=True)
        for name in names:
            src = os.path.join(src_root, subdir, name)
            dst = os.path.join(dst_root, subdir, name)
            if not os.path.exists(src):
                missing.append(f"{subdir}/{name}")
                continue
            if subdir == "tables":
                problems = inspect_table(src)
                if problems:
                    flagged[name] = problems
            if not dry_run:
                shutil.copy2(src, dst)
            copied.append(f"{subdir}/{name}")

    return copied, missing, flagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscript",
        default=DataConstants.MANUSCRIPT_DIR.value,
        help="Root of the manuscript repository.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    copied, missing, flagged = sync(
        DataConstants.WDIR.value, args.manuscript, args.dry_run
    )

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {len(copied)} exhibits to {args.manuscript}")
    for name in copied:
        print(f"  {name}")

    if missing:
        print(f"\nnot yet generated ({len(missing)}):")
        for name in missing:
            print(f"  {name}")

    # Copied regardless: a defect in a table is for the author to judge, and
    # refusing to sync would leave the manuscript with the older copy anyway.
    if flagged:
        print(f"\ncheck these tables ({len(flagged)}):")
        for name, problems in flagged.items():
            print(f"  {name}")
            for problem in problems:
                print(f"    - {problem}")
