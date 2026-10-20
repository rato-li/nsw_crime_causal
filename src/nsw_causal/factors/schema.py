"""Pydantic models for the factor catalog produced by LLM discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Proxy(BaseModel):
    """A public-dataset proxy through which a factor can be measured."""

    dataset: str
    variable: str
    geography: str = "SA2"
    frequency: str = "annual"
    url: str | None = None


class Factor(BaseModel):
    factor_id: str = Field(description="snake_case unique id, e.g. 'unemployment_rate'")
    name: str
    category: str = Field(description="economics | demographics | housing | health | "
                                      "environment | infrastructure | policing | social")
    definition: str
    expected_sign: Literal["+", "-", "?"] = "?"
    unit: str = ""
    proxies: list[Proxy] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    evidence: str = Field(default="moderate", description="strong | moderate | weak")
    notes: str = ""


class FactorEdge(BaseModel):
    source: str
    target: str
    relation: Literal["causes", "correlates", "proxy_of"] = "correlates"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance: Literal["llm", "correlation", "manual"] = "llm"


class FactorCatalog(BaseModel):
    version: str = "1.0"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    outcome: str = "crime_rate"
    outcome_definition: str = ("Recorded criminal offence rate per 100,000 residents "
                               "(NSW SBS / NSW Police open data), by statistical area "
                               "and year")
    notes: str = ""
    factors: list[Factor] = Field(default_factory=list)
    edges: list[FactorEdge] = Field(default_factory=list)

    # ------------------------------------------------------------------ I/O
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> FactorCatalog:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, factor_id: str) -> Factor | None:
        return next((f for f in self.factors if f.factor_id == factor_id), None)

    def factor_ids(self) -> list[str]:
        return [f.factor_id for f in self.factors]
