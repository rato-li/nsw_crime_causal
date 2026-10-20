"""Effect estimation: DoWhy identification + linear backdoor, EconML CausalForestDML.

For every factor adjacent to the outcome in the discovered DAG:
  1. compute the backdoor adjustment set from the DAG;
  2. DoWhy: identify + estimate ATE (linear backdoor; robust across versions);
  3. EconML: CausalForestDML ATE + CATE (heterogeneity), when installed.
A pure-numpy OLS backdoor is the final fallback so the pipeline always
produces numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from .discovery import adjustment_set


def _ols_backdoor(df: pd.DataFrame, treatment: str, outcome: str,
                  adj: list[str]) -> tuple[float, float, float]:
    """OLS of Y on [T, adj]; returns (ate, ci_low, ci_high) for T.

    Uses statsmodels when present, otherwise a pure-numpy OLS with the
    classical standard-error formula.
    """
    x_cols = [treatment, *adj]
    X = df[x_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    y = df[outcome].to_numpy(dtype=float)

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    beta = float(beta[1])
    se = float(se[1])
    return beta, beta - 1.96 * se, beta + 1.96 * se


def _dowhy_ate(df: pd.DataFrame, treatment: str, outcome: str,
               adj: list[str]) -> tuple[float, str]:
    from dowhy import CausalModel

    lines = ["digraph {"]
    for a in adj:
        lines.append(f"  {a} -> {treatment};")
        lines.append(f"  {a} -> {outcome};")
    lines.append(f"  {treatment} -> {outcome};")
    lines.append("}")
    model = CausalModel(data=df, treatment=treatment, outcome=outcome,
                        graph="\n".join(lines))
    ident = model.identify_effect()
    est = model.estimate_effect(ident, method_name="backdoor.linear_regression")
    return float(est.get_ate_signal()), ident


def _econml_cate(df: pd.DataFrame, treatment: str, outcome: str, adj: list[str],
                 settings: Settings) -> tuple[float, float, float, np.ndarray]:
    """CausalForestDML: returns (ate, ci_low, ci_high, cate per row)."""
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    y = df[outcome].to_numpy(dtype=float)
    t = df[treatment].to_numpy(dtype=float)
    w = df[adj].to_numpy(dtype=float) if adj else None

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(random_state=settings.seed),
        model_t=GradientBoostingRegressor(random_state=settings.seed),
        discrete_treatment=False,
        n_estimators=200,
        random_state=settings.seed,
        n_jobs=1,
    )
    est.fit(y, t, X=df[[treatment]].to_numpy(dtype=float), W=w)
    x_test = df[[treatment]].to_numpy(dtype=float)
    cate = est.effect(x_test, W=w)
    ate = float(np.mean(cate))
    try:
        inf = est.ate_inference(y, t, X=x_test, W=w)
        lo, hi = (float(v) for v in inf.point_interval(alpha=0.05))
    except Exception:  # noqa: BLE001 - inference needs extra fit machinery
        lo, hi = np.nan, np.nan
    return ate, lo, hi, cate


def estimate_effects(wide: pd.DataFrame, dag, outcome: str, treatments: list[str],
                     settings: Settings, use_econml: bool = True) -> pd.DataFrame:
    """Estimate ATE per treatment; returns tidy DataFrame sorted by |ATE|."""
    rows = []
    for t in treatments:
        adj = adjustment_set(dag, t, outcome)
        cols = sorted({t, outcome, *adj})
        df = wide[cols].dropna()
        if len(df) < 30:
            rows.append(dict(treatment=t, method="skipped", ate=np.nan,
                             ci_low=np.nan, ci_high=np.nan, n_obs=len(df),
                             adjustment=adj))
            continue

        ate = ci_lo = ci_hi = np.nan
        method = "ols-backdoor"
        try:
            ate, ci_lo, ci_hi = _ols_backdoor(df, t, outcome, adj)
        except Exception as exc:  # noqa: BLE001
            print(f"  {t}: OLS backdoor failed ({exc})")

        try:
            dw_ate, _ = _dowhy_ate(df, t, outcome, adj)
            if np.isfinite(dw_ate):
                ate = dw_ate
                method = "dowhy-backdoor-linear"
        except Exception:  # noqa: BLE001 - DoWhy version drift; OLS result stands
            pass

        if use_econml:
            try:
                cf_ate, cf_lo, cf_hi, cate = _econml_cate(df, t, outcome, adj, settings)
                rows.append(dict(treatment=t, method="econml-causal-forest-dml",
                                 ate=cf_ate, ci_low=cf_lo, ci_high=cf_hi,
                                 ate_linear=ate, n_obs=len(df), adjustment=adj,
                                 cate_mean=float(np.mean(cate)),
                                 cate_sd=float(np.std(cate))))
            except Exception as exc:  # noqa: BLE001
                print(f"  {t}: CausalForestDML unavailable ({exc})")
                rows.append(dict(treatment=t, method=method, ate=ate,
                                 ci_low=ci_lo, ci_high=ci_hi, ate_linear=np.nan,
                                 n_obs=len(df), adjustment=adj,
                                 cate_mean=np.nan, cate_sd=np.nan))
        else:
            rows.append(dict(treatment=t, method=method, ate=ate,
                             ci_low=ci_lo, ci_high=ci_hi, ate_linear=np.nan,
                             n_obs=len(df), adjustment=adj,
                             cate_mean=np.nan, cate_sd=np.nan))
        print(f"  {t:35s} ATE={ate: .3f} [{ci_lo: .3f}, {ci_hi: .3f}]  adj={len(adj)}")

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_ate"] = out["ate"].abs()
        out = out.sort_values("abs_ate", ascending=False).drop(columns="abs_ate")
    return out.reset_index(drop=True)
