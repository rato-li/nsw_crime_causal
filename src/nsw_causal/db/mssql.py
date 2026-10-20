"""MSSQL connectivity for the Data Vault."""

from __future__ import annotations

from ..config import Settings


def connect(settings: Settings):
    """Return a pyodbc connection to the Data Vault database."""
    import pyodbc

    return pyodbc.connect(settings.odbc_connection_string, timeout=20)


def get_engine(settings: Settings):
    """Return a SQLAlchemy engine (same DSN as :func:`connect`)."""
    import sqlalchemy as sa

    return sa.create_engine(settings.sqlalchemy_url, pool_pre_ping=True)


def run_script(conn, sql_text: str) -> None:
    """Execute a T-SQL script, splitting batches on lines that are just GO."""
    batches: list[str] = []
    current: list[str] = []
    for line in sql_text.splitlines():
        if line.strip().upper() == "GO":
            batches.append("\n".join(current))
            current = []
        else:
            current.append(line)
    batches.append("\n".join(current))

    cur = conn.cursor()
    try:
        for batch in batches:
            if batch.strip():
                cur.execute(batch)
        conn.commit()
    finally:
        cur.close()
