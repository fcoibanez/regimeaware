"""Realised performance and trading activity of a backtested weight series."""

import numpy as np
import pandas as pd


def portfolio_returns(weights, asset_returns):
    """Returns of a portfolio rebalanced to ``weights`` at the end of each period.

    Weights decided at the close of period t earn the asset returns of t+1.

    :param weights: (T, N) target weights indexed by period.
    :param asset_returns: (T', N) asset returns covering those periods.
    :return: Series of portfolio returns.
    """
    return weights.shift(1).mul(asset_returns).dropna(how="all").sum(axis=1)


def drifted_weights(weights, asset_returns):
    """Weights carried into each rebalance date by the previous period's returns.

    Between rebalances the portfolio is not held at its target weights: positions
    drift with their own realised returns. The turnover actually incurred at a
    rebalance is the distance from these drifted weights to the new target, not
    from the previous target.

    :param weights: (T, N) target weights indexed by period.
    :param asset_returns: (T', N) asset returns covering those periods.
    :return: (T, N) drifted weights, aligned to ``weights``.
    """
    growth = asset_returns.reindex(weights.index).reindex(weights.columns, axis=1) + 1
    drifted = weights.shift(1).mul(growth)
    return drifted.div(drifted.sum(axis=1), axis=0)


def turnover(weights, asset_returns, initial_purchase=True):
    """Two-sided turnover per rebalance, accounting for weight drift.

    Returns the sum of absolute weight changes, so a complete replacement of the
    portfolio counts as 2.0 and a proportional cost applies at half the quoted
    bid-ask spread per unit.

    :param weights: (T, N) target weights indexed by period.
    :param asset_returns: (T', N) asset returns covering those periods.
    :param initial_purchase: charge the opening position, bought from cash.
    :return: Series of per-period turnover.
    """
    traded = (weights - drifted_weights(weights, asset_returns)).abs().sum(axis=1)

    if initial_purchase:
        # The first row has no predecessor to drift from: the whole portfolio is
        # established from cash, which is a one-sided trade of the full weight.
        traded.iloc[0] = weights.iloc[0].abs().sum()
    else:
        traded = traded.iloc[1:]

    return traded


def net_returns(weights, asset_returns, cost_per_unit):
    """Portfolio returns after proportional transaction costs.

    :param cost_per_unit: cost charged per unit of turnover, i.e. the half-spread
        when ``turnover`` is measured two-sided.
    """
    gross = portfolio_returns(weights, asset_returns)
    charge = turnover(weights, asset_returns) * cost_per_unit
    return gross.sub(charge.reindex(gross.index), fill_value=0.0)


def sharpe_ratio(returns, periods_per_year=12):
    """Annualised Sharpe ratio, as reported throughout.

    The numerator is the geometrically annualised return and the denominator the
    annualised standard deviation. Defined here once so that tables and figures
    cannot drift apart: these portfolios are strongly right-skewed, and
    annualising the mean arithmetically instead gives a visibly different number.

    Note that the Ledoit and Wolf (2008) test deliberately does *not* use this
    definition. Its asymptotic theory is derived for the ratio of the mean to the
    standard deviation, so ``core.inference`` works with that form.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan

    growth = 1.0 + r
    if np.any(growth <= 0):
        # The path is wiped out, so a geometric annualisation is undefined
        return np.nan

    ann_ret = growth.prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)

    return ann_ret / ann_vol if ann_vol > 0 else np.nan


def path_metrics(weights, asset_returns, phis, threshold=1e-4, periods_per_year=12):
    """Performance statistics for one arm, one row per (risk aversion, path).

    Computed path by path so that every reported statistic carries a Monte Carlo
    distribution rather than only a point estimate, which is what makes the
    paired tests in ``core.inference`` possible.

    :param weights: DataFrame indexed by (phi, iteration, period).
    :param asset_returns: dict of iteration -> (T, N) asset returns.
    :param phis: risk aversion levels to evaluate.
    :param threshold: weights below this count as no position when measuring
        concentration. It is deliberately not applied to turnover, where zeroing
        small positions would book a trade whenever a weight crossed it.
    :return: DataFrame indexed by (phi, iteration).
    """
    from scipy.stats import entropy

    rows = {}
    paths = weights.index.get_level_values("iteration").unique()

    for phi in phis:
        w_phi = weights.xs(phi, level="phi")
        for i in paths:
            w_raw = w_phi.xs(i, level="iteration")
            r = portfolio_returns(w_raw, asset_returns[i])
            r = pd.to_numeric(r, errors="coerce").dropna()
            if r.empty:
                continue

            n = len(r)
            ann_ret = (1 + r).prod() ** (periods_per_year / n) - 1
            ann_vol = r.std() * np.sqrt(periods_per_year)
            wealth = (1 + r).cumprod()
            var95 = np.percentile(r, 5)

            w_held = w_raw.copy()
            w_held[w_held < threshold] = 0

            rows[(phi, i)] = {
                "Ann. Excess Return": ann_ret,
                "Ann. Std. Deviation": ann_vol,
                "Skewness": r.skew(),
                "Kurtosis": r.kurt(),
                "Sharpe Ratio": sharpe_ratio(r, periods_per_year),
                "Certainty Equivalent": certainty_equivalent(r, phi, periods_per_year),
                "Max. Drawdown": (wealth / wealth.cummax() - 1).min(),
                "Value-at-Risk (95%)": var95,
                "Expected Shortfall": r[r <= var95].mean(),
                "Portfolio Turnover": turnover(w_raw, asset_returns[i]).iloc[1:].mean()
                * periods_per_year,
                "Avg. Num. Constituents": (w_held > 0).sum(axis=1).mean(),
                "Effective Num. Bets": w_held.apply(entropy, axis=1).apply(np.exp).mean(),
            }

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=["phi", "iteration"])
    return out.sort_index()


def certainty_equivalent(returns, phi, periods_per_year=12):
    """Certainty equivalent under the exponential utility actually optimised.

    Reviewer 2 asks for certainty equivalent returns. The usual mean-variance
    expression is an approximation; because the portfolios here maximise expected
    exponential utility, the exact certainty equivalent of that same utility is
    the internally consistent choice:

        CE = -(1/phi) * log E[exp(-phi * r)]

    which reduces to the mean-variance form when returns are Gaussian. It is
    reported alongside the Sharpe ratio because the two need not agree: a method
    that shifts the portfolio along the risk-return frontier can raise one and
    lower the other.

    :param returns: realised portfolio returns for one path.
    :param phi: absolute risk aversion.
    :return: annualised certainty equivalent.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]

    # Work on the log scale so that large -phi*r does not overflow
    from scipy.special import logsumexp

    log_expected_utility = logsumexp(-phi * r) - np.log(len(r))
    return -(1 / phi) * log_expected_utility * periods_per_year


def _gross_and_charge(weights, asset_returns):
    """Gross returns and per-period turnover, aligned. Neither depends on cost."""
    gross = portfolio_returns(weights, asset_returns)
    charge = turnover(weights, asset_returns).reindex(gross.index).fillna(0.0)
    return gross.to_numpy(), charge.to_numpy()


def net_sharpe_curve(weights, asset_returns, costs, periods_per_year=12):
    """Annualised net Sharpe ratio across a grid of proportional cost levels.

    Reports performance as a *function* of the assumed trading cost rather than
    at one assumed level. Estimates of the effective spread vary by an order of
    magnitude depending on the sample period and the weighting scheme, so showing
    the whole curve keeps that choice from driving the conclusion.

    :param costs: cost per unit of turnover, i.e. the one-way cost when turnover
        is measured two-sided.
    :return: array of net Sharpe ratios, one per entry of ``costs``.
    """
    gross, charge = _gross_and_charge(weights, asset_returns)
    costs = np.atleast_1d(np.asarray(costs, float))

    net = gross[None, :] - charge[None, :] * costs[:, None]

    # Same definition as sharpe_ratio, evaluated across the whole cost grid at
    # once. The geometric annualisation is taken in logs for stability, and is
    # undefined on any row where the path is wiped out.
    growth = 1.0 + net
    valid = (growth > 0).all(axis=1)

    ann_ret = np.full(len(costs), np.nan)
    log_growth = np.log(np.where(valid[:, None], growth, 1.0))
    ann_ret[valid] = (
        np.exp(periods_per_year / net.shape[1] * log_growth[valid].sum(axis=1)) - 1
    )

    ann_vol = net.std(axis=1, ddof=1) * np.sqrt(periods_per_year)

    return np.where(valid & (ann_vol > 0), ann_ret / ann_vol, np.nan)


def cost_drag(weights, asset_returns, cost, periods_per_year=12):
    """Annualised return given up to trading, at a given cost per unit turnover."""
    _, charge = _gross_and_charge(weights, asset_returns)
    return charge.mean() * cost * periods_per_year


def breakeven_cost(weights_a, weights_b, asset_returns, periods_per_year=12):
    """Cost per unit of turnover at which two strategies deliver equal Sharpe.

    Answers the question Reviewer 2 poses directly: how expensive would trading
    have to be before the more active strategy stops being worthwhile. Returns
    ``inf`` when the ranking never reverses, and ``0.0`` when ``weights_a`` does
    not lead on a gross basis to begin with.
    """

    # Gross returns and turnover are both independent of the cost level, so they
    # are computed once and the search below is arithmetic on two short vectors.
    gross_a, charge_a = _gross_and_charge(weights_a, asset_returns)
    gross_b, charge_b = _gross_and_charge(weights_b, asset_returns)

    def sharpe_gap(cost):
        """Sharpe advantage of a over b, on the same definition as the tables."""
        sa = sharpe_ratio(gross_a - charge_a * cost, periods_per_year)
        sb = sharpe_ratio(gross_b - charge_b * cost, periods_per_year)
        if not (np.isfinite(sa) and np.isfinite(sb)):
            return np.nan
        return sa - sb

    if not sharpe_gap(0.0) > 0:
        return 0.0

    lo, hi = 0.0, 0.0001
    while sharpe_gap(hi) > 0:
        lo, hi = hi, hi * 2
        if hi > 1.0:
            # Costs of this size are not economically meaningful; the ranking is
            # effectively never reversed by trading frictions.
            return np.inf

    for _ in range(60):
        mid = (lo + hi) / 2
        if sharpe_gap(mid) > 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
