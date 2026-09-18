# Pre-submission checklist

Production items deliberately deferred until the manuscript is otherwise
finished, because each one reflows the document or invalidates a reading pass.
None of them is a referee response; they are the things that should be done once,
at the end, in a single sweep.

## Done, 14 September 2026

**The Ledoit and Wolf numbers now reproduce.** The bootstrap is seeded per path
and runs in parallel (`core.inference.ledoit_wolf_batch`, through joblib; the
serial run took about two hours, the parallel one under ten minutes). The
significance table and the detection-by-horizon sequence were regenerated and the
prose re-read against them: $15.6\%$ of five-year histories reject at the five
per cent level, and detection runs $17.3$, $37.3$, $59.3$, $85.6$ and $100$ per
cent at five, ten, twenty, forty and eighty years. A footnote now records why the
horizon exercise's five-year rate differs from the table's: it uses gross returns
over a fixed $150$ histories per horizon, the table net returns over all $1{,}000$.

**T1 encoding added**, and verified: the text sets in cm-super Type 1 fonts, and
the only Type 3 font in the document turned out to be `bbold`'s blackboard bold,
which is now dropped in favour of `amssymb`'s. Elsevier production rejects
Type 3 fonts, so this mattered independently of the encoding.

**Table files renamed** for what they hold (`performance.tex`, `significance.tex`,
`decomposition.tex`, `regime_conditional.tex`, `forecast_quality.tex`,
`momentum_crashes.tex`); `routines/sync_exhibits.py` and the notebooks updated
together. **Float order** fixed: the momentum-crash table is declared after the
HMM-fit table it follows in the text. **Panel labelling** is now uniform: every
risk-aversion division is headed by its `$\varphi$` alone, and the performance
and significance tables share the same abbreviated column headings.

## Remaining

**Recent citations** from EJOR and the OR journals this paper joins.

**Optional appendix** on `rolling_window_sensitivity.pdf`, justifying the
36-month rolling-OLS window (R3.3).

**Notebook pairing.** The `.ipynb` copies are stale relative to the `.py` scripts,
which are the live versions; jupytext is not installed in the `research` env.
Regenerate them or drop them before the repository is released.

**Figure 1 sub-period labels** are at year resolution (`1984--2004`, `2004--2024`)
and read as overlapping. The label is the cache key in
`notebooks/state_selection.py`, so changing it means refitting the grid.

## Open from the author's read-through (started 18 September 2026)

**The definition of $s$** -- deferred by the author, to be addressed later. The
author has begun replacing the bold process notation $\boldsymbol{s}$ with the
explicit state set $\{s_1, \dots, s_M\}$ (commit `7c6b821`). Three live uses of
$\boldsymbol{s}$ remain in Section 3.1: "an unobservable stochastic process
$\boldsymbol{s}$", "the hidden process $\boldsymbol{s}$ evolves", and "every
state in $\boldsymbol{s}$". The first two denote the process, where a distinct
symbol from a single state $s$ is standard; the third uses it as a set, the usage
already replaced elsewhere. Whatever is decided, the process, the set of states
and a single state should each have one symbol used consistently.

Awaiting the author's decision, raised during the read-through:

- **Table 3 caption, LW row.** Add that virtually all rejections favour RWLS
  (adverse rejections at most 0.7% in the gross breakdown) and that under the
  null the test rejects at close to its nominal 5%.
- **"Premultiplying" in Section 4.4.** The forecast is $\gamma_t \Pi$ with
  $\gamma_t$ a row vector, so it is postmultiplication; $\Pi\gamma_t$ does not
  conform. Relatedly, $\gamma^{(0)} \in \mathbb{R}^{M \times 1}$ in Section 3.1
  is a column vector, against the row convention for $\gamma_t$. Ties in with the
  definition of $s$ above.
- **`highlights.txt`.** The rewritten Sharpe highlight ("…against 2.11 for the
  strongest of five benchmarks") is uncommitted in the HMM repository.

## Not doing

**No final refresh** (decided 17 September 2026). The sample stays at July 1963
to December 2024 and the simulation at $1{,}000$ paths, as the manuscript states.
Extending the data to December 2025 and the paths to $5{,}000$ would have re-run
the whole pipeline and changed every number in the paper for a sample roughly
1.6% longer.

## Verification, after any regeneration

**Re-check every number quoted in prose.** The tables regenerate themselves; the
sentences do not.

**Run `routines/sync_exhibits.py` and read its warnings.** It inspects each
generated table for the defects that still compile: angle brackets outside maths,
unescaped per cent signs, and panel headings missing their rule.
