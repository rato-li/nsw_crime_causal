"""Data Vault persistence layer."""

from .mssql import connect, get_engine, run_script
from .repository import (
    hash_diff,
    hash_key,
    init_schema,
    insert_observation,
    insert_observations,
    read_factor_meta,
    read_panel,
    read_relations,
    save_embeddings,
    upsert_factor,
    upsert_relation,
)

__all__ = [
    "connect", "get_engine", "run_script",
    "hash_key", "hash_diff", "init_schema",
    "upsert_factor", "upsert_relation",
    "insert_observation", "insert_observations", "save_embeddings",
    "read_panel", "read_relations", "read_factor_meta",
]
