"""Multi-angle LLM factor discovery with merge + relation proposal.

Pipeline:
  1. Ask the local model the same question from several literature angles.
  2. Merge/dedupe candidates (deterministic id merge, optional LLM merge pass).
  3. Ask for conservative causal edges between surviving factors.
  4. Validate everything against the pydantic catalog schema.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

from ..config import Settings, get_settings
from ..factors.schema import Factor, FactorCatalog, FactorEdge
from . import prompts


def load_seed_catalog(path: Path | str | None = None) -> FactorCatalog:
    """Load the curated seed catalog shipped with the package."""
    if path is not None:
        return FactorCatalog.load(path)
    raw = (resources.files("nsw_causal") / "data" / "factor_catalog_seed.json") \
        .read_text(encoding="utf-8")
    return FactorCatalog.model_validate(json.loads(raw))


def _norm_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:48]


def _parse_factors(payload: object) -> list[Factor]:
    out: list[Factor] = []
    if isinstance(payload, dict):
        payload = payload.get("factors", [])
    if not isinstance(payload, list):
        return out
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            f = Factor.model_validate(item)
        except Exception:  # noqa: BLE001 - drop malformed candidates, keep the rest
            continue
        if f.factor_id != f.factor_id.lower():
            f.factor_id = _norm_id(f.factor_id)
        out.append(f)
    return out


def _merge_factors(cands: list[Factor]) -> list[Factor]:
    by_id: dict[str, Factor] = {}
    for f in cands:
        if f.factor_id in by_id:
            kept = by_id[f.factor_id]
            kept.proxies = (kept.proxies + f.proxies)[:6]
            kept.citations = list(dict.fromkeys(kept.citations + f.citations))[:8]
        else:
            by_id[f.factor_id] = f
    return list(by_id.values())


def discover_catalog(settings: Settings | None = None,
                     max_factors: int = 30,
                     use_merge_pass: bool = True) -> FactorCatalog:
    """Run the full LLM discovery loop. Requires the local Strata endpoint."""
    from .client import LLMClient

    settings = settings or get_settings()
    llm = LLMClient(settings)

    candidates: list[Factor] = []
    for angle in prompts.ANGLES:
        user = (angle["instruction"]
                + "\n\n"
                + prompts.FACTOR_JSON_CONTRACT.format(max_factors=max_factors // 2))
        payload = llm.chat_json(prompts.SYSTEM, user)
        found = _parse_factors(payload)
        print(f"  angle '{angle['id']}': {len(found)} factors")
        candidates.extend(found)

    merged = _merge_factors(candidates)

    if use_merge_pass and len(merged) > max_factors:
        user = prompts.MERGE_INSTRUCTION.format(
            max_factors=max_factors,
            candidates=json.dumps([f.model_dump() for f in merged]),
        )
        merged = _parse_factors(llm.chat_json(prompts.SYSTEM, user, temperature=0.0)) or merged

    merged = merged[:max_factors]
    ids = [f.factor_id for f in merged] + [settings.outcome_factor_id]

    edges: list[FactorEdge] = []
    try:
        user = prompts.RELATIONS_INSTRUCTION.format(
            factor_ids=json.dumps(ids), max_edges=3 * len(ids))
        payload = llm.chat_json(prompts.SYSTEM, user, temperature=0.0)
        valid = set(ids)
        if isinstance(payload, dict):
            payload = payload.get("edges", [])
        for item in payload if isinstance(payload, list) else []:
            try:
                e = FactorEdge.model_validate(item)
            except Exception:  # noqa: BLE001
                continue
            if e.source in valid and e.target in valid and e.source != e.target:
                edges.append(e)
    except Exception as exc:  # noqa: BLE001 - edges are a prior, not critical
        print(f"  warning: relation proposal failed ({exc}); continuing without priors")

    return FactorCatalog(
        outcome=settings.outcome_factor_id,
        notes=f"LLM-discovered catalog via {settings.llm_model} @ {settings.llm_base_url}",
        factors=merged,
        edges=edges,
    )
