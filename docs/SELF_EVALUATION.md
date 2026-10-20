# Self-Evaluation — Strengths, Weaknesses, and Improvement Plan

*Critical review of the project approach (LLM + GNN + Causal Inference for NSW
crime), for justification and continuous improvement. Companion to
[METHODOLOGY.md](METHODOLOGY.md).*

---

## 1. Strengths of the methodology

### 1.1 It attacks the weakest link in applied causal inference: the graph, not the arithmetic
Most published "causal" studies in criminology are regressions with a hopeful
adjustment set. Here the DAG is constructed explicitly, from four independent
elicitation angles (theory, empirics, confounder hunt, NSW context), validated
by schema, and merged with data-driven PC output. The DoWhy discipline of
*identify → estimate → refute* means every assumption is written down and
falsifiable before any number is produced. That auditability is exactly what a
justice-adjacent analysis needs.

### 1.2 The LLM solves the cold-start problem the data cannot
Because NSW withholds offender attributes (age, gender, education of
offenders), the variable set must be *invented* from the literature —
precisely what an LLM with criminological priors is good at. Multi-angle
querying plus a strict JSON contract (proxies must name a real ABS/SBS/AIHW
variable) turns free-form generation into a structured, reviewable instrument.
The local Strata deployment keeps crime-adjacent prompts on-premises, and the
curated seed catalog doubles as a hallucination tripwire: LLM output that
drifts far from the literature baseline is visible in review.

### 1.3 The GNN matches the actual structure of the problem
Factors are not independent columns — income, SEIFA, crowding, rent burden are
one latent web of disadvantage seen through different lenses. Message passing
over the factor graph shares statistical strength across a sparse panel
(census variables are 5-yearly), and the embedding clusters let us collapse
near-duplicate proxies into one adjustment-set representative instead of
padding the backdoor set with collinear noise. This is the same deconfounding
logic as NetDeconf/GCF from the workspace CFR-GNN notes, applied at the
variable level where the data actually lives.

### 1.4 The Data Vault makes it evidence-grade
Insert-only satellites with `LOAD_DTS + HASHDIFF` mean every causal claim can
be replayed against the exact inputs that produced it — ABS releases get
revised, and provenance is what separates an analysis from an anecdote.
Combined with per-stage artifacts (catalog, DAG dot, effects CSV, embeddings),
the whole chain is reviewable by a third party.

### 1.5 Falsification is institutionalised, not optional
Placebo treatments, random common causes, subset and leave-one-out refuters run
automatically on the top effects. In a domain where motivated reasoning is a
professional hazard (policing budgets, welfare debates), automated
"try-to-break-your-result" machinery is a genuine advance over the status quo.

---

## 2. Where the methodology falls short of best practice

### 2.1 Data (the deepest limitations)
- **Ecological fallacy + MAUP.** SA2 boundaries are arbitrary; results can
  shift with aggregation granularity (modifiable areal unit problem). Best
  practice runs the model at multiple geographies (SA2/LGA/postcode) and checks
  stability.
- **Recorded crime ≠ crime.** Reporting propensity varies with policing
  intensity and community trust — the very factors in the model. Best practice
  calibrates against the ABS Personal Safety Survey / NSW victimisation
  surveys, or models reporting separately.
- **Area-level is not a life sentence.** NSW Police publishes *incident-level*
  approximate locations (NSW Crime Data portal, ~1.5 km² precision, monthly),
  and NSW has a mature administrative-data-linkage pathway (Centre for Health
  Record Linkage, population data centres). Best practice eventually moves to
  point-pattern models (Poisson point processes, risk-terrain analysis) and
  linked microdata under governance.
- **Census interpolation** for 5-yearly factors induces fake autocorrelation
  that inflates apparent significance.

### 2.2 LLM layer
- **Citations are asserted, not verified.** Hallucinated references are the
  known failure mode; a verified-citation pass (RAG over actual AIC/SBS
  publications, DOI resolution) is the standard fix.
- **No uncertainty on the prior.** Best practice samples the elicitation many
  times (self-consistency / temperature ensembles) and reports *edge
  probabilities*, which then feed discovery as weighted priors — not binary
  edges. Raw LLM transcripts should be logged for audit, not just the merged
  catalog.

### 2.3 GNN layer
- **Circularity risk.** Node features include correlation with the outcome, and
  the embeddings then choose adjustment representatives — the selection partly
  encodes the thing it is meant to deconfound. Fix: exclude outcome-linked
  features from the embedding used for adjustment selection, or use *semantic*
  node features (sentence embeddings of factor definitions) combined with the
  structural graph.
- **~20 nodes is tiny.** Message passing there is defensible but marginal; the
  far more valuable GNN is the one the CFR-GNN notes describe — over the
  *region* graph, because crime has documented spatial spillover and
  displacement. That interference is currently invisible to the model.

### 2.4 Causal layer
- **PC assumes no latent confounding — false here.** FCI (outputs a PAG,
  admits hidden causes) plus proximal causal inference (DoWhy supports proxies)
  is the honest engine; we have exactly the proxy-rich setting proximal methods
  were built for.
- **The DAG is treated as known.** Discovery uncertainty never propagates into
  the CIs. Best practice: bootstrap discovery, report edge stability, and
  average estimates over plausible structures.
- **No multiple-testing control** across 14+ treatments (apply
  Benjamini–Hochberg); OLS CIs are not robust/clustered (use HC3 + clustering
  by region); two-way fixed effects / lagged treatments are flagged in code but
  not implemented — critical for `police_resources` simultaneity.
- **No external validation.** The synthetic harness proves the plumbing; it
  does not prove the model. NSW offers ready-made benchmarks with published
  answers: the Kings Cross lockout laws (NSW SBS evaluated ~25–40% drops in
  alcohol-related violence), the gun buyback, mandatory sentencing. A
  methodology that cannot reproduce a known quasi-experiment should not be
  trusted on new questions.

### 2.5 Governance
- Area-level models of "disadvantage → crime" carry real
  **redlining/stigmatisation risk**. Best practice adds fairness audits (do
  estimates survive within-cluster permutation of disadvantaged areas?), a
  model card, and alignment with the NSW AI Assurance Standard — currently only
  caveats in the report.
- **Reproducibility gaps:** seeds are fixed, but data versions (ABS vintages)
  are not content-addressed (no DVC/containerisation), and LLM outputs are not
  fully logged.

---

## 3. Improvement roadmap

### 3.1 Quick wins (days)
1. Robust (HC3) and region-clustered standard errors.
2. Benjamini–Hochberg FDR correction across treatment estimates.
3. Bootstrap edge-stability on causal discovery; report edge frequencies.
4. Log raw LLM responses alongside the merged catalog (audit trail).
5. Lag treatments (1 year) and add a two-way fixed-effects estimator.
6. Replace k-means with hierarchical clustering + stability selection.

### 3.2 Medium term (weeks)
1. FCI discovery + proximal adjustment with negative-control exposures.
2. RAG-grounded factor discovery with citation verification and
   edge-probability ensembles from repeated elicitation.
3. Semantic (text-embedding) node features to remove the outcome-correlation
   circularity in adjustment selection.
4. Spatial-lag / region-graph GNN for crime displacement and spillover
   (connects to the CFR-GNN design notes).
5. Victimisation-survey calibration of reporting propensity.
6. Multi-granularity robustness checks (SA2 / LGA / postcode) against MAUP.

### 3.3 Structural (the real best-practice gap)
1. Move beyond open data: apply for incident-level crime data and linked
   administrative microdata through the proper ethics / data-trust channels
   (NSW Population Data Linkage, CHRL).
2. Quasi-experimental validation benchmarks: lockout laws, gun buyback,
   staggered policy changes — synthetic control / DiD against published
   effect sizes.
3. Formal sensitivity suite (E-values, Cinelli–Hazlett robustness values)
   alongside the refuters.
4. Publish model card + fairness/redlining audit with every report; align with
   the NSW AI Assurance Standard and Australian AI Ethics Principles.

---

## 4. Bottom line

The architecture is stronger than typical practice because it makes
assumptions explicit, auditable, and falsifiable at every layer. Its honest
ceiling is *"well-defended ecological associations with credible causal
structure"*. Closing the gap to best practice is less about algorithms and
more about: (1) admitting latent confounding (FCI / proximal), (2) spatial and
interference modelling, (3) grounding the LLM in verified literature, and
(4) earning trust through quasi-experimental validation and governance
artifacts.
