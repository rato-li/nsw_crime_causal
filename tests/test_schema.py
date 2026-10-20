"""Schema + catalog validation tests (no heavy deps)."""

import pytest

from nsw_causal.db.repository import hash_diff, hash_key
from nsw_causal.factors.schema import FactorCatalog
from nsw_causal.llm.factor_discovery import load_seed_catalog


def test_seed_catalog_validates():
    catalog = load_seed_catalog()
    assert len(catalog.factors) >= 10
    ids = catalog.factor_ids()
    assert len(ids) == len(set(ids)), "duplicate factor ids"
    assert catalog.outcome not in ids, "outcome must not be a factor"
    known = set(ids) | {catalog.outcome}
    for e in catalog.edges:
        assert e.source in known and e.target in known


def test_catalog_roundtrip(tmp_path):
    catalog = load_seed_catalog()
    path = catalog.save(tmp_path / "cat.json")
    again = FactorCatalog.load(path)
    assert again.factor_ids() == catalog.factor_ids()
    assert len(again.edges) == len(catalog.edges)


def test_hash_key_deterministic_and_case_insensitive():
    a = hash_key("unemployment_rate")
    b = hash_key("  UNEMPLOYMENT_RATE ")
    assert a == b
    assert len(a) == 32


def test_hash_diff_collapses_none_and_empty():
    assert hash_diff(["a", None]) == hash_diff(["a", ""])


def test_llm_json_extraction_edge_cases():
    from nsw_causal.llm.client import extract_json

    assert extract_json('```json\n[1, 2]\n```') == [1, 2]
    assert extract_json('noise {"a": 1} noise') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")
