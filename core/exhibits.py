"""Export of result tables as LaTeX for direct inclusion in the manuscript.

Every number that reaches the paper should come from here rather than being
copied by hand. Transcription is where exhibits drift apart: the submitted draft
reported Sharpe ratios in Table 2 on one definition and plotted them in Figure 3
on another, a discrepancy of up to 0.47 that no amount of care in the analysis
would have caught, because it happened after the analysis.

Files are written as bare ``tabular`` environments so the manuscript keeps
control of float placement, captions and labels, and so re-running an analysis
updates the paper without touching its structure.

Labels are emitted close to verbatim, because column headings like ``$1/N$`` or
``Costa \\& Kwon`` are deliberate LaTeX and must survive intact. The exception is
the handful of characters that are never intended as markup in a metric name --
see ``_escape_label``.
"""

import os
import re

import numpy as np
import pandas as pd

# How each reported quantity is rendered. Percentages are stored as fractions
# throughout the analysis and converted here, so a change of unit cannot happen
# silently in a notebook.
DEFAULT_FORMATS = {
    "Ann. Excess Return": "pct2",
    "Ann. Std. Deviation": "pct2",
    "Certainty Equivalent": "pct2",
    "Max. Drawdown": "pct2",
    "Value-at-Risk (95%)": "pct2",
    "Expected Shortfall": "pct2",
    "Mean return": "pct2",
    "Volatility": "pct2",
    "Return advantage (ann.)": "pct2",
    "Contribution to total (ann.)": "pct2",
    "5th pct month": "pct2",
    "5th pct improvement": "pct2",
    "Worst month": "pct2",
    "Skewness": "num2",
    "Kurtosis": "num2",
    "Sharpe Ratio": "num2",
    "Sharpe (arith.)": "num2",
    "Sharpe (gross)": "num2",
    "Sharpe (net)": "num2",
    "Portfolio Turnover": "num2",
    "Volatility ratio": "num2",
    "Cost drag": "num2",
    "Delta": "num3",
    "Std. Err.": "num3",
    "t-stat": "num2",
    "p (paired t)": "pval",
    "p (Wilcoxon)": "pval",
    "p (sign)": "pval",
    "p-value": "pval",
    "Avg. Num. Constituents": "num1",
    "Effective Num. Bets": "num1",
    "Months": "int",
    "N paths": "int",
    "Share of months": "pct1",
    "Frac. > 0": "pct1",
    "Reject (favourable)": "pct1",
    "Reject (adverse)": "pct1",
    "Inconclusive": "pct1",
    "Frac. never reversed": "pct1",
    "Frac. no gross lead": "pct1",
    r"LW reject 5\%": "pct1",
}


def _render(value, spec):
    """Format a single cell."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        # An infinite breakeven is a result, not a missing value: it means no
        # level of trading cost reverses the ranking.
        if isinstance(value, float) and value == np.inf:
            return r"$\infty$"
        return "--"

    if spec == "pct2":
        return f"{value * 100:.2f}\\%"
    if spec == "pct1":
        return f"{value * 100:.1f}\\%"
    if spec == "num1":
        return f"{value:.1f}"
    if spec == "num2":
        return f"{value:.2f}"
    if spec == "num3":
        return f"{value:.3f}"
    if spec == "int":
        return f"{int(round(value)):,}"
    if spec == "pval":
        # Below the resolution the Monte Carlo sample can support, quoting more
        # digits would overstate what 1000 paths establish.
        return "$<$0.001" if value < 0.001 else f"{value:.3f}"
    return str(value)


def _spec_for(label, formats):
    return (formats or {}).get(label, DEFAULT_FORMATS.get(label, "num2"))


def _escape_label(label):
    """Escape the characters in a label that are never meant as LaTeX markup.

    Labels are otherwise emitted verbatim, because column headings such as
    ``$1/N$`` and ``Costa \\& Kwon`` are deliberate LaTeX and must survive. But a
    per cent sign in a name like ``Value-at-Risk (95%)`` is always a per cent
    sign, and left alone it comments out the rest of the row -- silently merging
    it with the next one, which is the kind of defect that reaches print because
    the file still compiles.

    The angle brackets are the same kind of trap and quieter still. Under the
    default font encoding ``>`` and ``<`` in text mode are typeset as an inverted
    question mark and an inverted exclamation mark, so a heading like
    ``Frac. > 0`` compiles without complaint and prints as ``Frac. ¿ 0``. They
    are moved into maths mode, where they mean what they say.
    """
    text = str(label)
    for char in ("%", "#"):
        text = re.sub(rf"(?<!\\){re.escape(char)}", rf"\\{char}", text)

    # Only outside existing maths: a label may already carry its own $...$.
    segments = text.split("$")
    for i in range(0, len(segments), 2):
        for char in ("<", ">"):
            segments[i] = segments[i].replace(char, f"${char}$")
    return "$".join(segments)


def write_tabular(
    frame,
    path,
    formats=None,
    panel_level=None,
    panel_prefix=None,
    index_header="",
    notes=None,
    column_format=None,
    format_axis="index",
    full_width=False,
    rule_below_panel=True,
    title=None,
):
    """Write ``frame`` as a booktabs tabular.

    :param frame: rows are the quantities reported, columns the things compared.
        With ``panel_level`` set, the index is a MultiIndex whose named level
        splits the table into stacked panels.
    :param path: destination ``.tex`` file; parent directories are created.
    :param formats: overrides of the default rendering, keyed by row label.
    :param panel_level: index level to break into panels, e.g. risk aversion.
    :param index_header: heading above the row labels, usually blank.
    :param notes: lines appended after the table, inside the file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    n_cols = frame.shape[1]
    column_format = column_format or "l" + "r" * n_cols

    if full_width:
        # Stretch the columns evenly across the text block. A table whose natural
        # width falls short of the measure otherwise sits bunched against the left
        # margin, which reads as crowded however much white space surrounds it.
        #
        # ``True`` stretches to the full measure; a number stretches to that
        # fraction of it, which suits a table with few enough columns that the
        # full width would leave visible gaps between them.
        #
        # The stretch glue goes *after* the first column, not before it. Placed at
        # the head of the preamble it replaces the leading \tabcolsep, so the data
        # rows lose their left pad while a \multicolumn panel heading -- which does
        # not inherit that @{} -- keeps its own, leaving the heading indented
        # relative to the rows beneath it.
        environment = "tabular*"
        stretched = column_format[:1] + "@{\\extracolsep{\\fill}}" + column_format[1:]
        measure = ("\\textwidth" if full_width is True
                   else f"{float(full_width):g}\\textwidth")
        opening = f"\\begin{{tabular*}}{{{measure}}}{{{stretched}}}"
    else:
        environment = "tabular"
        opening = f"\\begin{{tabular}}{{{column_format}}}"

    lines = [opening, "\\toprule"]

    # A title set inside the tabular rather than above it, so the rules that
    # enclose it are the table's own and span exactly its width. Set outside, the
    # heading floats free of the rules and two stacked tables read as two
    # exhibits that happen to share a caption.
    if title is not None:
        lines.append(
            f"\\multicolumn{{{n_cols + 1}}}{{c}}{{\\textit{{{_escape_label(title)}}}}}"
            " \\\\"
        )
        lines.append("\\midrule")

    lines.append(
        f"{index_header} & "
        + " & ".join(_escape_label(c) for c in frame.columns)
        + " \\\\"
    )
    lines.append("\\midrule")

    if panel_level is None:
        blocks = [(None, frame)]
    else:
        values = frame.index.get_level_values(panel_level).unique()
        blocks = [(v, frame.xs(v, level=panel_level)) for v in values]

    for position, (panel, block) in enumerate(blocks):
        if panel is not None:
            if position:
                lines.append("\\midrule")
            # The risk-aversion level names the division on its own. Lettering it
            # as well spends Panel A on something that is not a panel of the
            # argument, and leaves nothing to call the halves of a table that
            # genuinely has two -- which then have to nest inside it.
            heading = f"$\\varphi = {panel}$"
            if panel_prefix:
                letter = chr(ord("A") + position)
                heading = f"{panel_prefix} {letter}: {heading}"
            lines.append(
                f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{{heading}}}}} \\\\"
            )
            if rule_below_panel:
                # Separates a panel heading from its rows, so a reader scanning a
                # long table can see at a glance where each panel begins. A full
                # \midrule rather than a \cmidrule: the latter trims its ends by
                # \cmidrulekern, leaving the rule visibly short of the table edges
                # that every other rule reaches.
                lines.append("\\midrule")

        for label, row in block.iterrows():
            if format_axis == "columns":
                # Tables that report several different quantities side by side
                # take their formatting from the column rather than the row.
                cells = " & ".join(
                    _render(v, _spec_for(c, formats))
                    for c, v in zip(frame.columns, row)
                )
            else:
                spec = _spec_for(label, formats)
                cells = " & ".join(_render(v, spec) for v in row)
            lines.append(f"{_escape_label(label)} & {cells} \\\\")

    lines.append("\\bottomrule")
    lines.append(f"\\end{{{environment}}}")

    if notes:
        lines.append("")
        lines.extend(f"% {line}" for line in np.atleast_1d(notes))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return path


def stars(p):
    """Conventional significance markers, for tables that report them inline."""
    if not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.1 else ""


def with_stars(estimates, pvalues, spec="num2"):
    """Render estimates with significance markers attached."""
    return pd.DataFrame(
        {
            col: [
                _render(v, spec) + stars(p)
                for v, p in zip(estimates[col], pvalues[col])
            ]
            for col in estimates.columns
        },
        index=estimates.index,
    )
