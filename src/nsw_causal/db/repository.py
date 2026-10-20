"""Data Vault 2.0 repository: hash keys, hub/link/satellite writes, panel reads.

Every write is insert-only (Data Vault principle): hubs and links are MERGED
on their hash key, satellites append a new version row only when the HASHDIFF
of the descriptive attributes changes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable
from importlib import resources

import pandas as pd

from ..factors.schema import Factor, FactorEdge

# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def hash_key(*parts: object) -> str:
    """Deterministic 32-char hash of the business key parts (Data Vault HK)."""
    raw = "|".join(str(p).strip().upper() for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest().upper()


def hash_diff(values: Iterable[object]) -> str:
    """Hashdiff over satellite attributes; None and "" collapse to one token."""
    raw = "\x1f".join("" if v is None else str(v) for v in values)
    return hashlib.md5(raw.encode("utf-8")).hexdigest().upper()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def init_schema(conn) -> None:
    from .mssql import run_script

    for name in ("001_schema_datavault.sql", "002_views.sql"):
        sql = (resources.files("nsw_causal.sql") / name).read_text(encoding="utf-8")
        run_script(conn, sql)


# ---------------------------------------------------------------------------
# Hub / link / satellite writes
# ---------------------------------------------------------------------------

def ensure_hub(conn, table: str, hk_col: str, key_col: str, hk: str,
               key: str, source: str, extra_cols: dict | None = None) -> None:
    extra_cols = extra_cols or {}
    cols = [hk_col, key_col, "LOAD_DTS", "LOAD_SRC", *extra_cols.keys()]
    vals = [hk, key, _now(), source, *extra_cols.values()]
    placeholders = ", ".join("?" for _ in cols)
    update_cols = ", ".join(f"{c} = inserted.{c}" for c in extra_cols) or None
    insert_cols = ", ".join(cols)
    merge = (
        f"MERGE {table} WITH (HOLDLOCK) AS tgt "
        f"USING (SELECT {placeholders}) AS src ({insert_cols}) "
        f"ON tgt.{hk_col} = src.{hk_col} "
        f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({', '.join('src.' + c for c in cols)})"
    )
    if update_cols:
        merge += f" WHEN MATCHED THEN UPDATE SET {update_cols};"
    else:
        merge += ";"
    cur = conn.cursor()
    cur.execute(merge, *vals)
    conn.commit()
    cur.close()


def _latest_hashdiff(conn, table: str, hk_col: str, hk: str) -> str | None:
    cur = conn.cursor()
    cur.execute(
        f"SELECT TOP 1 HASHDIFF FROM {table} WHERE {hk_col} = ? ORDER BY LOAD_DTS DESC",
        hk,
    )
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def _sat_insert(conn, table: str, hk_col: str, hk: str, hashdiff: str,
                source: str, attrs: dict) -> None:
    cols = [hk_col, "LOAD_DTS", "HASHDIFF", "LOAD_SRC", *attrs.keys()]
    vals = [hk, _now(), hashdiff, source, *attrs.values()]
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    cur = conn.cursor()
    cur.execute(sql, *vals)
    conn.commit()
    cur.close()


def upsert_factor(conn, factor: Factor, source: str = "llm-discovery") -> str:
    """Write a factor to HUB_FACTOR + SAT_FACTOR. Returns the hash key."""
    hk = hash_key(factor.factor_id)
    ensure_hub(conn, "dv.HUB_FACTOR", "HK_FACTOR", "FACTOR_KEY", hk,
               factor.factor_id[:64], source)
    attrs = {
        "FACTOR_NAME": factor.name,
        "CATEGORY": factor.category,
        "DEFINITION": factor.definition,
        "EXPECTED_SIGN": factor.expected_sign,
        "UNIT": factor.unit,
        "CITATION": " || ".join(factor.citations),
        "EVIDENCE": factor.evidence,
        "IS_ACTIVE": True,
    }
    hd = hash_diff(attrs.values())
    if _latest_hashdiff(conn, "dv.SAT_FACTOR", "HK_FACTOR", hk) != hd:
        _sat_insert(conn, "dv.SAT_FACTOR", "HK_FACTOR", hk, hd, source, attrs)
    return hk


def upsert_relation(conn, edge: FactorEdge, source: str = "llm-discovery") -> None:
    hk_from, hk_to = hash_key(edge.source), hash_key(edge.target)
    hk = hash_key(edge.source, edge.target, edge.relation)
    sql = (
        "MERGE dv.LINK_FACTOR_RELATION WITH (HOLDLOCK) AS tgt "
        "USING (SELECT ? AS HK_RELATION) AS src ON tgt.HK_RELATION = src.HK_RELATION "
        "WHEN NOT MATCHED THEN INSERT (HK_RELATION, HK_FACTOR_FROM, HK_FACTOR_TO, "
        "RELATION_TYPE, CONFIDENCE, PROVENANCE, LOAD_DTS, LOAD_SRC) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
    )
    now = _now()
    cur = conn.cursor()
    cur.execute(sql, hk, hk, hk_from, hk_to, edge.relation,
                float(edge.confidence), edge.provenance, now, source)
    conn.commit()
    cur.close()


def insert_observation(conn, factor_id: str, region_key: str, period_key: str,
                       dataset_key: str, value: float, quality: str = "ok",
                       source: str = "etl", period_year: int | None = None,
                       period_type: str = "YEAR") -> None:
    """Insert one factor/region/period observation (hub+link+satellite)."""
    hk_f = hash_key(factor_id)
    hk_r = hash_key(region_key)
    hk_p = hash_key(period_key)
    hk_d = hash_key(dataset_key)
    ensure_hub(conn, "dv.HUB_FACTOR", "HK_FACTOR", "FACTOR_KEY", hk_f, factor_id[:64], source)
    ensure_hub(conn, "dv.HUB_REGION", "HK_REGION", "REGION_KEY", hk_r, region_key[:64], source)
    ensure_hub(conn, "dv.HUB_PERIOD", "HK_PERIOD", "PERIOD_KEY", hk_p, period_key[:32],
               source, {"PERIOD_YEAR": period_year, "PERIOD_TYPE": period_type})
    ensure_hub(conn, "dv.HUB_DATASET", "HK_DATASET", "DATASET_KEY", hk_d, dataset_key[:128], source)

    hk_o = hash_key(factor_id, region_key, period_key, dataset_key)
    now = _now()
    cur = conn.cursor()
    cur.execute(
        "MERGE dv.LINK_OBSERVATION WITH (HOLDLOCK) AS tgt "
        "USING (SELECT ? AS HK_OBSERVATION) AS src ON tgt.HK_OBSERVATION = src.HK_OBSERVATION "
        "WHEN NOT MATCHED THEN INSERT (HK_OBSERVATION, HK_FACTOR, HK_REGION, HK_PERIOD, "
        "HK_DATASET, LOAD_DTS, LOAD_SRC) VALUES (?, ?, ?, ?, ?, ?, ?);",
        hk_o, hk_o, hk_f, hk_r, hk_p, hk_d, now, source,
    )
    conn.commit()
    cur.close()

    attrs = {"OBS_VALUE": None if value is None else float(value),
             "OBS_TEXT": None, "QUALITY_FLAG": quality}
    hd = hash_diff(attrs.values())
    if _latest_hashdiff(conn, "dv.SAT_OBSERVATION", "HK_OBSERVATION", hk_o) != hd:
        _sat_insert(conn, "dv.SAT_OBSERVATION", "HK_OBSERVATION", hk_o, hd, source, attrs)


def insert_observations(conn, frame: pd.DataFrame, source: str = "etl",
                        batch_log_every: int = 500) -> int:
    """Bulk-insert a long panel: columns factor_id, region, period, value,
    (optional) dataset, quality. Returns rows written."""
    n = 0
    for row in frame.itertuples(index=False):
        dataset = getattr(row, "dataset", None) or "synthetic-panel"
        quality = getattr(row, "quality", None) or "ok"
        period = str(row.period)
        year = int(period[:4]) if period[:4].isdigit() else None
        insert_observation(conn, row.factor_id, str(row.region), period,
                           dataset, row.value, quality, source, year)
        n += 1
        if n % batch_log_every == 0:
            print(f"  inserted {n} observations ...")
    return n


def save_embeddings(conn, embeddings: dict[str, list[float]],
                    model_version: str, source: str = "gnn-embed") -> int:
    """Persist factor embeddings as JSON rows in SAT_FACTOR_EMBEDDING."""
    import json

    n = 0
    for factor_id, vec in embeddings.items():
        hk = hash_key(factor_id)
        ensure_hub(conn, "dv.HUB_FACTOR", "HK_FACTOR", "FACTOR_KEY", hk,
                   factor_id[:64], source)
        payload = json.dumps([round(float(x), 6) for x in vec])
        attrs = {"MODEL_VERSION": model_version, "EMBED_DIM": len(vec),
                 "EMBEDDING": payload}
        hd = hash_diff(attrs.values())
        if _latest_hashdiff(conn, "dv.SAT_FACTOR_EMBEDDING", "HK_FACTOR", hk) != hd:
            _sat_insert(conn, "dv.SAT_FACTOR_EMBEDDING", "HK_FACTOR", hk, hd,
                        source, attrs)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def read_panel(conn) -> pd.DataFrame:
    """Read the panel as a wide DataFrame: index=(region, period), columns=factor."""
    cur = conn.cursor()
    df = pd.read_sql_query(
        "SELECT FACTOR_KEY, REGION_KEY, PERIOD_KEY, OBS_VALUE FROM dv.V_FACTOR_PANEL",
        cur,
    )
    cur.close()
    if df.empty:
        return df
    wide = df.pivot_table(index=["REGION_KEY", "PERIOD_KEY"],
                          columns="FACTOR_KEY", values="OBS_VALUE")
    wide.columns = [str(c) for c in wide.columns]
    return wide


def read_relations(conn) -> pd.DataFrame:
    cur = conn.cursor()
    df = pd.read_sql_query(
        "SELECT FACTOR_FROM, FACTOR_TO, RELATION_TYPE, CONFIDENCE, PROVENANCE "
        "FROM dv.V_FACTOR_RELATIONS",
        cur,
    )
    cur.close()
    return df


def read_factor_meta(conn) -> pd.DataFrame:
    cur = conn.cursor()
    df = pd.read_sql_query("SELECT * FROM dv.V_FACTOR_META", cur)
    cur.close()
    return df
