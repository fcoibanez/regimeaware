# Pre-submission checklist

Production items deliberately deferred until the manuscript is otherwise
finished, because each one reflows the document or invalidates a reading pass.
None of them is a referee response; they are the things that should be done once,
at the end, in a single sweep.

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

**Narrow tables.** Table 8 is stretched to the text width with `full_width=True`.
Tables 1 and 6 are narrower than the measure but have few enough columns that
stretching them would leave visible gaps; left as they are deliberately.

## Verification

**Re-check every number quoted in prose.** The tables regenerate themselves; the
sentences do not. Sections 4.2, 4.4, 4.5 and 4.6 and the introduction all carry
figures typed into the text. This matters most after the final refresh, when
every exhibit changes at once.

**Run `routines/sync_exhibits.py` and read its warnings.** It inspects each
generated table for the defects that still compile: angle brackets outside maths,
unescaped per cent signs, and panel headings missing their rule.
