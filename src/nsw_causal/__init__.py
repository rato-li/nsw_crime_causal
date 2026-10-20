"""NSW crime causal-model pipeline.

LLM-assisted factor discovery -> DataVault 2.0 (MSSQL) storage ->
GNN message-passing factor embeddings -> causal discovery (PC + priors) ->
DoWhy identification + EconML estimation -> refutation & reporting.
"""

__version__ = "0.1.0"
