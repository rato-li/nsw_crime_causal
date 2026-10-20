"""Refutation tests (DoWhy) for the top effect estimates.

For each selected treatment we run four standard falsification tests:
  * random_common_cause      - add a synthetic unobserved confounder
  * placebo_treatment_refuter - replace treatment with random noise (effect -> 0)
  * data_subset_refuter      - re-estimate on random 50% subsets
  * leave_one_out_random_cause - drop a random confounder and re-estimate
A robust estimate barely moves under these perturbations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from .discovery import adjustment_set

REFUTERS = [
    "random_common_cause",
    "placebo_treatment_refuter",
    "data_subset_refuter",
    "leave_one_out_random_cause",
]


def refute_effects(wide: pd.DataFrame, dag, outcome: str, treatments: list[str],
                   settings: Settings) -> pd.DataFrame:
    rows = []
    for t in treatments:
        adj = adjustment_set(dag, t, outcome)
        cols = sorted({t, outcome, *adj})
        df = wide[cols].dropna()
        if len(df) < 50:
            continue
        try:
            from dowhy import CausalModel

            lines = ["digraph {"]
            for a in adj:
                lines.append(f"  {a} -> {t};")
                lines.append(f"  {a} -> {outcome};")
            lines.append(f"  {t} -> {outcome};")
            lines.append("}")
            model = CausalModel(data=df, treatment=t, outcome=outcome,
                                graph="\n".join(lines))
            ident = model.identify_effect()
            est = model.estimate_effect(ident, method_name="backdoor.linear_regression")
            base = float(est.get_ate_signal())
        except Exception as exc:  # noqa: BLE001
            print(f"  {t}: base estimate for refutation failed ({exc})")
            continue

        for refuter in REFUTERS:
            try:
                res = model.refute_estimate(
                    ident, est, method_name=refuter,
                    num_simulations=20, random_state=settings.seed)
                ref = res.refutation
                new_effect = float(ref.get("new_effect", np.nan))
                p_value = ref.get("p_value", None)
                rows.append(dict(treatment=t, refuter=refuter,
                                 base_effect=base, new_effect=new_effect,
                                 p_value=(float(p_value) if p_value is not None else np.nan),
                                 interpretation=str(ref.get("interpretation", ""))))
            except Exception as exc:  # noqa: BLE001 - refuters vary by version
                rows.append(dict(treatment=t, refuter=refuter, base_effect=base,
                                 new_effect=np.nan, p_value=np.nan,
                                 interpretation=f"failed: {exc}"))
        print(f"  refuted {t}: {len(REFUTERS)} tests")

    return pd.DataFrame(rows)
