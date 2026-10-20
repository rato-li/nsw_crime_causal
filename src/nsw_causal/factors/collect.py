"""Data collection: public-dataset loaders + a clearly-flagged synthetic panel.

Real ingestion from ABS / NSW SBS / AIHW is geography- and licence-specific,
so this module ships:
  * ``load_panel_csv``  - long CSV (factor_id, region, period, value[, dataset])
    produced by whatever downloader you point at the public sources listed in
    the factor catalog proxies;
  * ``synthetic_panel`` - a plausible correlated panel with PLANTED causal
    effects, used to validate the pipeline end-to-end (never for conclusions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..factors.schema import FactorCatalog

PANEL_COLUMNS = ["factor_id", "region", "period", "value", "dataset", "quality"]


def load_panel_csv(path: str | Path) -> pd.DataFrame:  # noqa: F821
    df = pd.read_csv(path)
    missing = {"factor_id", "region", "period", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"panel CSV missing columns: {sorted(missing)}")
    if "dataset" not in df.columns:
        df["dataset"] = "external"
    if "quality" not in df.columns:
        df["quality"] = "ok"
    return df[PANEL_COLUMNS]


def synthetic_panel(catalog: FactorCatalog, settings: Settings | None = None,
                    n_regions: int = 88, years: range = range(2013, 2025),
                    seed: int | None = None) -> pd.DataFrame:
    """Generate a correlated SA2 x year panel with planted effects on crime_rate.

    Latent structure: a socioeconomic latent drives income/education/SEIFA-like
    factors; a built-environment latent drives housing factors; crime_rate is
    generated from a few *planted* direct effects plus noise, so downstream
    causal discovery has ground truth to recover.
    """
    settings = settings or get_settings()
    rng = np.random.default_rng(seed if seed is not None else settings.seed)
    n_years = len(years)
    n = n_regions * n_years

    ses = rng.normal(size=n)
    built = rng.normal(size=n)
    region_effect = np.repeat(rng.normal(size=n_regions), n_years)

    # planted direct effects on the outcome
    planted = {
        "unemployment_rate": 0.55,
        "youth_population_share": 0.45,
        "rent_burden": 0.35,
        "alcohol_outlets_density": 0.30,
        "median_household_income": -0.50,
        "education_completion": -0.35,
    }

    rows = []
    factor_loadings = {
        "median_household_income": (0.9, "ses"),
        "education_completion": (0.8, "ses"),
        "seifa_disadvantage": (-0.85, "ses"),
        "unemployment_rate": (-0.7, "ses"),
        "income_inequality": (-0.3, "ses"),
        "public_housing_share": (-0.5, "ses"),
        "dwelling_crowding": (0.75, "built"),
        "rent_burden": (0.7, "built"),
        "rental_vacancy_shortage": (0.6, "built"),
        "public_transport_access": (-0.5, "built"),
    }

    values: dict[str, np.ndarray] = {}
    for f in catalog.factors:
        fid = f.factor_id
        if fid in factor_loadings:
            load, latent = factor_loadings[fid]
            base = ses if latent == "ses" else built
            values[fid] = load * base + 0.4 * region_effect + rng.normal(scale=0.5, size=n)
        elif fid == "youth_population_share":
            values[fid] = 0.3 * region_effect + rng.normal(scale=0.8, size=n)
        elif fid in planted:
            values[fid] = rng.normal(scale=1.0, size=n) + 0.3 * region_effect
        else:  # generic factor: weak ses link + noise
            values[fid] = 0.2 * ses + 0.3 * region_effect + rng.normal(scale=1.0, size=n)

    crime = np.zeros(n)
    for fid, beta in planted.items():
        if fid in values:
            crime = crime + beta * values[fid]
    crime = crime + 0.4 * ses + 0.5 * region_effect + rng.normal(scale=0.7, size=n)
    values["crime_rate"] = crime

    for fid, arr in values.items():
        for i in range(n_regions):
            for j, year in enumerate(years):
                idx = i * n_years + j
                rows.append({
                    "factor_id": fid,
                    "region": f"SA2_{1000 + i}",
                    "period": str(year),
                    "value": round(float(arr[idx]), 4),
                    "dataset": "synthetic-panel",
                    "quality": "synthetic",
                })
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


def panel_to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Long panel -> wide (index=(region, period), columns=factor_id)."""
    wide = panel.pivot_table(index=["region", "period"], columns="factor_id",
                             values="value")
    wide.columns = [str(c) for c in wide.columns]
    return wide
