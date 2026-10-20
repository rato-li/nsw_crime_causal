"""Prompt templates for LLM-assisted causal factor discovery.

Each *angle* attacks the question from a different literature tradition so the
union of answers covers the space better than a single prompt. The model must
answer with a JSON array of factor objects matching ``factors.schema.Factor``.
"""

from __future__ import annotations

SYSTEM = (
    "You are a quantitative criminologist specialising in Australian and New South "
    "Wales crime research. You reason about causal structures (confounders, "
    "mediators, instruments) and you only propose factors that can be measured "
    "from PUBLIC Australian datasets (ABS Census, ABS Labour Force, NSW SBS "
    "crimnology statistics, AIHW, SEIFA, NSW open data). You answer with strict "
    "JSON and you never invent statistics; when unsure you lower 'evidence'."
)

FACTOR_JSON_CONTRACT = """
Answer with a JSON array. Each element:
{
  "factor_id": "snake_case_id",
  "name": "Human readable name",
  "category": "economics|demographics|housing|health|environment|infrastructure|policing|social",
  "definition": "1-2 sentence precise definition",
  "expected_sign": "+ or - effect on recorded crime rate",
  "unit": "e.g. % of population, per 100,000",
  "proxies": [
    {"dataset": "ABS Census 2021", "variable": "Income (weekly, median)",
     "geography": "SA2", "frequency": "one-off|annual|quarterly", "url": "optional"}
  ],
  "citations": ["Author (Year) Title, Venue"],
  "evidence": "strong|moderate|weak",
  "notes": "causal role: confounder / mediator / mechanism / proxy; caveats"
}
Return at most {max_factors} factors. Do NOT include the outcome itself
(crime rate) as a factor.
"""

ANGLES: list[dict[str, str]] = [
    {
        "id": "theory",
        "instruction": (
            "Enumerate causal factors for the recorded crime rate of a NSW local "
            "area derived from criminological THEORY: social disorganisation, "
            "strain/anomie, routine activity, social control, and rational "
            "choice. For each factor state the causal mechanism in 'notes'."
        ),
    },
    {
        "id": "datasets",
        "instruction": (
            "Enumerate factors that have been EMPIRICALLY linked to property and "
            "violent offending in Australian studies (AIC, NSW SBS, criminology "
            "journals), and that can be measured at SA2/LGA level from public "
            "Australian datasets. Prefer factors with concrete Census/ABS/AIHW "
            "variables as proxies."
        ),
    },
    {
        "id": "confounders",
        "instruction": (
            "You are the devil's advocate for internal validity. List the most "
            "dangerous CONFOUNDERS and common causes that would bias a naive "
            "regression of crime rate on any single policy-relevant factor "
            "(e.g. policing, welfare, migration). Include demographic structure, "
            "youth density, tourism/visitor flux, and reporting-propensity "
            "factors."
        ),
    },
    {
        "id": "context",
        "instruction": (
            "Enumerate NSW-SPECIFIC contextual factors: the Sydney apartment "
            "boom and construction workforce, FIFO mining workforce flows, "
            "holiday/tourist regions, public housing precincts, transport "
            "deserts, liquor and gambling venue density, and heatwaves. Only "
            "keep those measurable from public data."
        ),
    },
]

MERGE_INSTRUCTION = """
Below are candidate causal factors proposed by several analysts (JSON arrays).
Merge duplicates and near-duplicates (keep the best definition, union the
proxies and citations), drop anything not measurable from public Australian
data, and return the final JSON array of at most {max_factors} factors using
the same schema.

CANDIDATES:
{candidates}
"""

RELATIONS_INSTRUCTION = """
Given this list of factor ids for a NSW crime-rate causal study:
{factor_ids}

Output a JSON array of causal relationships between them (and to the outcome
"crime_rate"). Each element:
{{"source": id, "target": id, "relation": "causes|correlates|proxy_of",
  "confidence": 0.0-1.0}}
Use "causes" only where established theory/literature supports a direct
causal arrow (source -> target). Be conservative: omit edges you are not at
least 0.5 confident about. At most {max_edges} edges.
"""
