# Online appendices — candidates

Deferred until the main document is final (the author's read-through and the
corrections that follow it come first). Grouped by what each defends; the cost
column is the honest estimate of what producing it takes.

Status legend: **ready** (material exists in the repo), **arm** (one new
backtest arm on the existing simulation), **DGP** (a new simulation and every arm
re-run), **write-up** (no computation).

## A. Already computed

| # | Appendix | Defends | Material | Cost |
|---|----------|---------|----------|------|
| A1 | Number of regimes, full grid: the criterion values behind Figure 1 and the four-state ablation as a table (earns less, cuts volatility more) | §4.2 specification choice | `tables/state_selection.tex`; ablation run with `--states 4` | write-up |
| A2 | Power and size of the Ledoit–Wolf test: detection by horizon; size by block length (6.4/2.4/1.6/0.8 % vs nominal 5 %); the HAC variant that under-rejects | §4.5, block length of one | `img/detection_horizon.pdf`, `tables/detection_horizon.tex`, comment in `core/inference.py` | write-up, plus a small size run |
| A3 | Rolling-window sensitivity, 24/36/60 months | R3.3 | `img/rolling_window_sensitivity.pdf`; the 24m/60m arms are **not** in `results/` and must be re-run to refresh | arm ×2 |
| A4 | Incremental path as distributions across paths, not means | Table 5 | `img/ablation_increments.pdf`, `ablation_phi.pdf`, `ablation_ecdf.pdf` | write-up |
| A5 | Transaction-cost estimate: 0.1242 % from 2.07 M CRSP stock-months, the cap-weighted counterpart, the range in the literature | Figure 5 band, R2.4 | `notebooks/tcost_estimation` | write-up |
| A6 | Regime timing against NBER: lead/lag ratios (strongest at +2 months), the 63 episodes | §4.2 | `notebooks/regime_characterization` | write-up |

## B. Robustness to misspecification

| # | Appendix | Defends | Cost |
|---|----------|---------|------|
| B7 | Model uses fewer factors than the DGP (six in the DGP, three in the model): does conditioning on the regime still help when the regime is identified from an incomplete factor set? | Mechanism finding | arm (one new HMM fit) |
| B8 | Wrong number of regimes: DGP with 2 or 4 states, model fixed at 3 | §4.2, the referees' hardest push | DGP |
| B9 | Correlated factors within regime: non-diagonal $\Omega_s$ in the DGP, diagonal in the model | Stated limitation 3 | DGP |
| B10 | Fat-tailed emissions: Student-$t$ factor shocks, Gaussian HMM (Bulla et al. use $t$-HMMs) | Emission specification | DGP |
| B11 | Shorter in-sample histories, $T=240$ | Limitation 1, the DeMiguel caveat, the Reversal-state mass constraint | arm (truncate existing paths) |

B8, B9 and B10 can share one alternative simulation if it is generated once with
all three variations switched on separately.

## C. Claims made but not exhibited

| # | Appendix | Defends | Cost |
|---|----------|---------|------|
| C12 | Oracle bound: portfolio performance of `model_oracle` (known DGP regimes), the ceiling on what regime identification can deliver | R2.4 | **ready** — arm is run and aggregated; only its forecast quality is shown (Table 6 Panel B) |
| C13 | Scalability: solve time per rebalance and total run time as $N$ grows (200/500/1000), RWLS against a full asset-level regime-switching fit | The title | cheap for RWLS; the comparison is the point |
| C14 | Filtered instead of forecast probabilities in the optimizer ($\gamma_t$ for $\hat\gamma_{t+1}$ in eq. 10): a direct test of "no timing" | §4.6 mechanism | arm |
| C15 | Recovery of $\alpha_s$ and $\sigma^2_{i,s}$, not only the loadings | Figure 3 | write-up (data from the recovery exercise) |
| C16 | Finer risk-aversion grid than three values | "robust across preferences" | `ablation_phi.pdf` may already cover it |

## D. Not an appendix

| # | | |
|---|---|---|
| D17 | A real-data application. Every result is simulation calibrated to the estimated model, so the DeMiguel caveat applies to all of it. Not requested at this round; the most likely request at the next. | major |

## Suggested order

A1, A2, A5, C12 first (all data exist; about a day of writing). Then one B
item — B8 is the sharpest, and the shared DGP run makes B9 and B10 nearly free
once it exists. B7 and B11 are the cheap ones if compute is the constraint.
