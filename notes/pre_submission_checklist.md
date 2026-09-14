# Pre-submission checklist

Production items deliberately deferred until the manuscript is otherwise
finished, because each one reflows the document or invalidates a reading pass.
None of them is a referee response; they are the things that should be done once,
at the end, in a single sweep.

## Blocking: the Ledoit and Wolf numbers are not reproducible yet

**This one changes reported numbers and must not be forgotten.**

The bootstrap was unseeded when Table 4 was generated, so the rejection rates it
reports cannot be reproduced by anyone re-running the code, including us.
`core.inference` now seeds it per path, but the table has not been regenerated
because doing so requires the full bootstrap, which takes hours.

When it runs, **the rejection rates will change**, and every one of them quoted
in the text has to be re-checked against the new table. As of this writing
section 4.5 quotes:

- $15.8\%$ of individual histories rejecting at the five per cent level
- $19.8\%$ and $17.6\%$ at the two higher levels of risk aversion, via the table
- the detection sequence $18.0$, $35.3$, $59.3$, $84.8$ and $100$ per cent,
  which comes from `detection_by_horizon` and is also bootstrap-based

Two tables are affected: `table3_significance.tex`, and `table6_detection_horizon.tex`
if it is ever restored. Both are produced by `notebooks/statistical_validation.py`.

The same run should pick up the panel relabelling described below, so that
Tables 3 and 4 match the convention the other tables already use.

## Typography

**Add `\usepackage[T1]{fontenc}`.** The document currently uses the OT1 default,
whose 128-slot table puts other characters where `>`, `<` and `|` should be: a
bare `>` in text mode prints as `¿`. This caused a real defect in Table 7, which
compiled without a warning and which `pdftotext` extracts as `>` — the text layer
is correct while the glyph on the page is not, so neither the log nor text
extraction can catch it.

`core.exhibits._escape_label` now moves angle brackets in generated tables into
maths mode, so the tables are protected at source. T1 is insurance for
hand-written prose, and it additionally allows accented words to hyphenate, which
OT1 refuses to do.

Deferred because it changes glyph metrics slightly, so line breaks and pagination
shift throughout. Do it before a final read-through, not during one.

Check afterwards, since both are legitimate uses of a bare `>` that must survive:
the TikZ arrow style near the top of the preamble, and the `array` column
specifiers in the questionnaire-style table.

## Exhibits

**Table file names no longer match their numbers.** `table4_decomposition.tex` is
Table 7, `table5a_regime_conditional.tex` is Table 8, and so on; only
`table3_significance.tex` still happens to coincide. Cross-references resolve
correctly through `\ref`, so nothing is broken, but hunting for the file behind a
table number is needlessly confusing. Rename once the layout has settled, and
update `routines/sync_exhibits.py` and the generating notebooks together.

**Table and figure float ordering.** LaTeX keeps separate queues for the two, so
a table can overtake a figure declared before it. Table 2 currently prints a page
ahead of Figure 2 although the text introduces the figure first.

**Narrow tables.** The regime-conditional table is stretched to four fifths of
the measure with `full_width=0.8`. Others are narrower than the measure but have
few enough columns that stretching them would leave visible gaps; left as they
are deliberately.

**Panel labelling.** Divisions by risk aversion are now headed `$\varphi = 10$`
rather than `Panel A: $\varphi = 10$`, so that `Panel A` and `Panel B` are free
for the two halves of the merged mechanism table. `table2_performance.tex`,
`table4_decomposition.tex` and `table5a_regime_conditional.tex` follow the new
convention. `table3_significance.tex` does not, because regenerating it requires
the bootstrap; it will be picked up by the run described above.

## Verification

**Re-check every number quoted in prose.** The tables regenerate themselves; the
sentences do not. Sections 4.2, 4.4, 4.5 and 4.6 and the introduction all carry
figures typed into the text. This matters most after the final refresh, when
every exhibit changes at once.

**Run `routines/sync_exhibits.py` and read its warnings.** It inspects each
generated table for the defects that still compile: angle brackets outside maths,
unescaped per cent signs, and panel headings missing their rule.
