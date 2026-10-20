# Methodology — Causal Model of NSW Crime Rates

*LLM-assisted factor discovery → Data Vault 2.0 (MSSQL) → GNN message-passing
embeddings → causal discovery → DoWhy/EconML inference → refutation.*

---

## 0. Scope, estimand, and the data constraint

**Question.** Which publicly measurable area-level factors have a statistically
and materially significant *causal* impact on the recorded crime rate in NSW
statistical areas?

**Estimand.** For each candidate factor `T`: the area-average treatment effect
of a one-standard-deviation shift in `T` on the offences-per-100k rate, under
the backdoor adjustment set implied by the discovered DAG; plus a CATE
(heterogeneous effect) from causal forest.

**The constraint that shapes everything.** NSW publishes *offence counts*
(NSW Police / SBS Crimstats) but **not** offender attributes (age, gender,
education) at any analysable level. Therefore:

1. The model is **ecological** (SA2/LGA × year panels). Findings describe
   places, not individuals; individual-level mechanisms (e.g. "low education
   causes offending") are *not* identified and are only proxied by area
   composition (e.g. share completing Year 12).
2. Age/education effects enter only as **contextual composition variables**
   (`youth_population_share`, `education_completion`), and this is stated in
   every report.
3. Ethnicity is deliberately excluded as a predictor; structural disadvantage
   is represented by SEIFA indices instead (ethical-governance decision,
   documented in the catalog).

## 1. Factor discovery with the local LLM

The GPU box runs Qwen via Strata behind an OpenAI-compatible endpoint, so all
LLM traffic stays on-premises (no crime-adjacent prompts leave the LAN).

Discovery is **multi-angle** (`llm/prompts.py`): the same question is asked
from four independent traditions —

1. **Theory** (social disorganisation, strain, routine activity, social
   control);
2. **Empirics** (AIC / NSW SBS findings with SA2-measurable proxies);
3. **Confounder hunt** (devil's-advocate pass for dangerous common causes,
   including *reporting propensity*);
4. **NSW context** (apartment boom, FIFO flows, tourist regions, public-housing
   precincts, heat).

Each answer must conform to a strict JSON contract (factor id, definition,
expected sign, unit, **public-dataset proxies**, citations, evidence strength).
Candidates are merged deterministically by normalised id, optionally by an LLM
merge pass, then a second LLM call proposes **conservative causal edges**
(`causes | correlates | proxy_of` with confidence). Everything is validated
against a pydantic schema; malformed items are dropped, never guessed.

A curated seed catalog (18 factors, literature-cited) ships with the package
and doubles as (a) the offline fallback and (b) a regression test that the LLM
output stays sane. **Human review gate:** the catalog is written to
`artifacts/catalog.json` for expert sign-off before any causal claim.

## 2. Storage: Data Vault 2.0 on MSSQL (LAN)

Raw factor values are loaded into `DataVault` on `192.168.1.185,1433`
(login `ai`) using the classic hub-link-satellite pattern (schema `dv`):

- **Hubs**: `FACTOR`, `REGION`, `PERIOD`, `DATASET` — MD5 hash of the business
  key; insert-only.
- **Links**: `LINK_OBSERVATION` (factor × region × period × dataset — one row
  per measurement) and `LINK_FACTOR_RELATION` (the factor graph the GNN reads).
- **Satellites**: `SAT_FACTOR` (definitions, citations, expected sign),
  `SAT_OBSERVATION` (**raw numeric values**, quality flags), `SAT_REGION`,
  `SAT_FACTOR_EMBEDDING` (GNN vectors, JSON, versioned by model version).
  Versioning is `LOAD_DTS + HASHDIFF`: re-running the ETL never overwrites
  history, so every causal run is reproducible from the vault.
- **Views** (`V_FACTOR_PANEL`, …) give the pipeline a flat consumption layer.

Why a vault? Crime/factor data arrives from ABS releases on different vintages
and geographies; the vault keeps provenance (`LOAD_SRC`), handles late
revisions via satellite versioning, and lets the causal layer stay stateless.

## 3. GNN factor embeddings (message passing) — before the causal model

Rationale: factors are not isolated columns — they form a semantic/empirical
web (income ↔ SEIFA ↔ crowding …). Embedding them on their graph *before*
causal modelling does three jobs:

1. **Proxy de-duplication.** K-means on the embeddings groups near-duplicate
   proxies (e.g. SEIFA disadvantage vs low income); each cluster contributes
   one medoid representative to adjustment sets, preventing collinear padding
   of the backdoor set and stabilising DML nuisances.
2. **Regularised features.** Embeddings summarise a factor's whole graph
   neighbourhood, giving discovery/estimation smoother inputs than raw
   aggregates.
3. **Persistence.** Vectors go back into `SAT_FACTOR_EMBEDDING`, so any other
   tool in the vault ecosystem can reuse them.

Mechanics (`gnn/`): nodes = factors (incl. outcome); edges = LLM priors ∪
|Spearman ρ| ≥ threshold on the panel; symmetric-normalised adjacency with
self-loops. A GraphSAGE-style layer computes
`h' = LayerNorm(W_self·h + W_nbr·(Â h))`; 2 layers, residual connection, and a
linear decoder. Training is **self-supervised link prediction** (dot-product
decoder, positive edges vs sampled negatives) plus a feature-reconstruction
term — no labels needed. Pure PyTorch (dense adjacency is exact at ~20 nodes);
the RTX 5080 is used via the cu128 build, though CPU is fine at this scale.
This deliberately mirrors the CFR-GNN design notes in this workspace.

## 4. Causal discovery

`causal/discovery.py` runs **PC with Fisher-Z** tests (causal-learn) on the
standardised panel, then merges literature/LLM prior edges: priors are added
only where PC found nothing and no cycle would be created; PC-oriented edges
are never overridden, and conflicts are flagged for review. Undirected PC
edges are stored as bidirected `pc-unknown` so downstream adjustment is
conservative (both endpoints treated as parents). A transparent
correlation-skeleton fallback keeps the pipeline running when causal-learn is
absent, and the report always prints which engine produced the graph.

Notes: with census-vintage factors (5-yearly) interpolation is required —
interpolation induces autocorrelation, so PC results should be re-run on
bootstrapped resamples before trusting edge presence. FCI (also in
causal-learn) is the recommended upgrade when latent confounding is suspected.

## 5. Causal inference (DoWhy + EconML)

For every factor adjacent to `crime_rate` in the DAG:

1. **Adjustment set** = parents of the treatment that are not descendants of
   the outcome (backdoor criterion), with GNN-cluster representatives replacing
   near-duplicate proxies.
2. **DoWhy** builds the explicit graph, runs `identify()` (backdoor), and
   estimates a linear backdoor ATE — the *modelling discipline* is the point:
   assumptions are written down, not implicit in a regression call.
3. **EconML `CausalForestDML`** estimates the nonlinear/heterogeneous effect
   (nuisances: gradient boosting; honest forest; ATE + CATE + CI). DML's
   cross-fitting handles the regularisation bias from many correlated factors.
4. A pure-numpy OLS backdoor is the final fallback so the pipeline always
   emits numbers, with the method column stating which estimator produced them.

Panel structure (region × year) supports two upgrades flagged in code:
two-way fixed effects (region + year FE) and lagged treatments (1 year) to
dampen simultaneity — especially for `police_resources`, where crime drives
deployment as much as deployment drives crime.

## 6. Refutation and sensitivity

For the top-k effects, DoWhy refuters: `random_common_cause`,
`placebo_treatment_refuter`, `data_subset_refuter`,
`leave_one_out_random_cause` (20 sims each). A factor is reported as
*credible* only if (i) the placebo effect collapses to ~0, (ii) subset and
leave-one-out estimates stay within the original CI, and (iii) the random
common cause moves the estimate by less than ~20% at plausible strength.
Unmeasured confounding can never be refuted — only bounded — so the report
pairs every ATE with these stress tests and the sensitivity narrative.

## 7. Threats to validity (stated in every report)

| Threat | Mitigation / honest statement |
|---|---|
| Ecological fallacy | Area-level effects only; never restated as individual effects |
| Reporting propensity | Recorded crime ≠ offending; confounder pass includes policing/reporting factors |
| Reverse causation | Lags for police/economic variables; DAG conflicts flagged |
| Latent confounding | Conservative adjustment; FCI/sensitivity recommended |
| Census interpolation | Autocorrelation caveat; bootstrap stability check |
| LLM hallucination | JSON contracts, citations required, curated seed catalog, human sign-off gate |
| Denominator drift | Transient-population factor; population estimates from ABS ERP |

## 8. Reproducibility

Fixed seeds throughout; Data Vault satellites version every input; artifacts
(catalog, DAG dot, effects CSV, embeddings NPZ) are content-addressed per run;
CI runs lint + tests without needing the DB or the LLM.

## 9. Self-evaluation (strengths, weaknesses, improvements)

The project maintains a standing critical review of this methodology in
[SELF_EVALUATION.md](SELF_EVALUATION.md). Summary position:

- **Strengths.** The DAG is constructed explicitly and auditably (multi-angle
  LLM priors + PC + refutation) rather than assumed; the LLM solves the
  cold-start variable problem created by the absence of offender attributes;
  the GNN de-duplicates the latent-disadvantage web before adjustment; the
  Data Vault makes every claim replayable; falsification is automated.
- **Weaknesses vs best practice.** Ecological/MAUP limits; recorded crime ≠
  offending; PC assumes no latent confounding; discovery uncertainty does not
  propagate into CIs; no multiple-testing control, robust SEs, lags or FEs yet;
  LLM citations unverified; embedding features carry an outcome-correlation
  circularity; no spatial spillover model; no external quasi-experimental
  validation; governance artifacts (model card, fairness audit) not yet
  produced.
- **Improvement roadmap.** Quick wins (robust/clustered SEs, BH-FDR, bootstrap
  edge stability, LLM response logging, lags + two-way FE) → medium term (FCI +
  proximal adjustment, RAG-grounded discovery with verified citations, semantic
  node features, region-graph GNN for displacement, victimisation-survey
  calibration) → structural (incident-level/linked microdata access,
  quasi-experimental benchmarks such as the Kings Cross lockout laws, formal
  sensitivity suite, published model card + fairness audit).

Full detail and justification: [SELF_EVALUATION.md](SELF_EVALUATION.md).
