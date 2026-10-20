"""Central configuration, loaded from environment variables / .env file.

All variables use the ``NSWCAUSAL_`` prefix, e.g. ``NSWCAUSAL_MSSQL_HOST``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NSWCAUSAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- MSSQL / Data Vault -------------------------------------------------
    mssql_host: str = "192.168.1.185"
    mssql_port: int = 1433
    mssql_database: str = "DataVault"
    mssql_user: str = "ai"
    mssql_password: str = "CHANGE_ME"  # real value goes in .env (git-ignored)
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_trust_cert: bool = True

    # --- Local LLM (Strata, OpenAI-compatible) -------------------------------
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "qwen3.8-flash-next"
    llm_api_key: str = "local"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 3000
    llm_timeout_s: float = 300.0

    # --- Paths ----------------------------------------------------------------
    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    reports_dir: Path = Field(default=Path("reports"))

    # --- Pipeline knobs --------------------------------------------------------
    seed: int = 42
    corr_edge_threshold: float = 0.35
    gnn_layers: int = 2
    gnn_hidden: int = 64
    gnn_dim: int = 32
    gnn_epochs: int = 300
    gnn_lr: float = 5e-3
    gnn_neg_ratio: int = 2
    gnn_device: str = "auto"  # auto | cpu | cuda
    embedding_model_version: str = "mp-v1"
    pc_alpha: float = 0.05
    outcome_factor_id: str = "crime_rate"

    # ---------------------------------------------------------------------------
    @property
    def odbc_connection_string(self) -> str:
        trust = "yes" if self.mssql_trust_cert else "no"
        return (
            f"DRIVER={{{self.mssql_driver}}};"
            f"SERVER={self.mssql_host},{self.mssql_port};"
            f"DATABASE={self.mssql_database};"
            f"UID={self.mssql_user};PWD={self.mssql_password};"
            f"Encrypt=yes;TrustServerCertificate={trust};"
            f"ApplicationName=nsw-crime-causal;"
        )

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mssql+pyodbc://{self.mssql_user}:{quote_plus(self.mssql_password)}"
            f"@{self.mssql_host}:{self.mssql_port}/{self.mssql_database}"
            f"?driver={quote_plus(self.mssql_driver)}"
            f"&TrustServerCertificate={'yes' if self.mssql_trust_cert else 'no'}"
        )

    def resolve(self, p: Path | str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (Path.cwd() / path)


@lru_cache
def get_settings() -> Settings:
    return Settings()
