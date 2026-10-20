# Original Project Prompt (verbatim)

*Captured 2026-10-19 as the founding requirement for this project. Reproduced
word-for-word except for one redaction: the MSSQL password is replaced with
`[REDACTED]` — real credentials live only in the git-ignored `.env`.*

---

> you are an expert in integrating AI LLM with private datasets. You also
> specialises in Causal Discovery and Inference algorithms, specially using
> packages like DoWhy and EconML. You have installed DeepSeek Harness on a
> local pc with RTX 5080 GPU card. You have also successfully installed and run
> the QWen3.8-Flash-Next model locally via Strata. Describe your methodology in
> conducting a Causal Model on NSW crime, to identify the factors which impacts
> significantly the crime rate. You do not have the detail dataset on NSW
> criminal offences and attributes on offenders like age, gender, education.
> Using the LLM queries, identify the potential factors from other publicly
> available research findings and public datasets. Built this project in Gibhub,
> using Python as your main language. Before the factors are used to construct
> the causal model, compute the embedding of each using the Graph Neural Network
> mechanism, eg message passing. The raw factor values are stored inside a Data
> Vault relational database with the hub, link, and satellite tables. Create
> project that can be deployed to Github easily, using Python, SQL, and MSSQL
> server running in the same local network. MSSQL
> servername=192.168.1.185,1433,database=DataVault,username=ai,pwd=[REDACTED].

---

## Requirement traceability

| # | Requirement in the prompt | Where it is implemented |
|---|---|---|
| 1 | Describe the methodology | [METHODOLOGY.md](METHODOLOGY.md) |
| 2 | DoWhy + EconML causal discovery/inference | `src/nsw_causal/causal/` (PC discovery, DoWhy identification, EconML CausalForestDML, refutation) |
| 3 | Local LLM (Qwen via Strata) on RTX 5080 | `src/nsw_causal/llm/client.py` (OpenAI-compatible endpoint, `NSWCAUSAL_LLM_*` env vars); cu128 torch build for Blackwell |
| 4 | NSW crime causal model — factors significantly impacting crime rate | `nswcausal estimate` → `artifacts/effects.csv`, `reports/report.md` |
| 5 | No detailed offence/offender dataset (age, gender, education unavailable) | Area-level design decision, `METHODOLOGY.md` §0; catalog notes; ecological caveats in every report |
| 6 | LLM queries identify factors from public research + public datasets | `llm/factor_discovery.py` + `llm/prompts.py` (4-angle elicitation, JSON contract, ABS/SBS/AIHW proxies); curated seed catalog `src/nsw_causal/data/factor_catalog_seed.json` |
| 7 | GitHub-deployable Python project | `pyproject.toml`, `.github/workflows/ci.yml`, README, MIT license, git repo |
| 8 | GNN message-passing embeddings before the causal model | `src/nsw_causal/gnn/` (factor graph, GraphSAGE-style message passing, link-prediction training, proxy clustering) |
| 9 | Data Vault relational DB with hub/link/satellite tables | `src/nsw_causal/sql/001_schema_datavault.sql`, `db/repository.py` (insert-only, HASHDIFF versioning) |
| 10 | Python + SQL + MSSQL on the local network (192.168.1.185,1433/DataVault, user `ai`) | `config.py` defaults, `.env.example`, `db/mssql.py` (pyodbc/SQLAlchemy) |
| 11 | Self-evaluation of approach and justification | [SELF_EVALUATION.md](SELF_EVALUATION.md) |

## Related documents

- Methodology: [METHODOLOGY.md](METHODOLOGY.md)
- Strengths / weaknesses / improvements: [SELF_EVALUATION.md](SELF_EVALUATION.md)
- Usage and quickstart: [../README.md](../README.md)
