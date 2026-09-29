"""DuckDB + Parquet analytical store.

Prefer DuckDB/Polars for multi-year panels. All cleaned intermediates are
snappy-compressed Parquet, partitioned by ``year=YYYY`` or ``symbol=XYZ``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, Literal, Sequence

import duckdb
import polars as pl

from src.data.paths import DUCKDB_PATH, PROCESSED_DIR, ensure_data_dirs

logger = logging.getLogger(__name__)

PartitionBy = Literal["year", "symbol"]


def connect(db_path: Path | None = None, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (or create) the project DuckDB database."""
    ensure_data_dirs()
    path = db_path or DUCKDB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def parquet_glob(dataset_dir: Path, partition_by: PartitionBy | None = None) -> str:
    """Build a DuckDB-friendly glob for a partitioned dataset."""
    if partition_by == "year":
        return str(dataset_dir / "year=*" / "*.parquet")
    if partition_by == "symbol":
        return str(dataset_dir / "symbol=*" / "*.parquet")
    return str(dataset_dir / "**" / "*.parquet")


def register_parquet_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    dataset_dir: Path,
    *,
    partition_by: PartitionBy | None = "year",
    hive_partitioning: bool = True,
) -> str:
    """Create or replace a view over a Parquet dataset directory."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    glob = parquet_glob(dataset_dir, partition_by)
    hive = "true" if hive_partitioning else "false"
    sql = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT * FROM read_parquet('{glob}', hive_partitioning={hive}, union_by_name=true)
    """
    con.execute(sql)
    logger.info("Registered view %s -> %s", view_name, glob)
    return view_name


def write_partitioned_parquet(
    df: pl.DataFrame,
    dest_dir: Path,
    *,
    partition_by: PartitionBy = "year",
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    compression: str = "snappy",
    overwrite: bool = True,
) -> list[Path]:
    """Write a Polars frame as snappy Parquet partitions.

    Parameters
    ----------
    partition_by:
        ``year`` uses ``trade_date.year``; ``symbol`` partitions by ticker.
    overwrite:
        If True, replace existing partition directories for keys present in ``df``.
    """
    ensure_data_dirs()
    dest_dir.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        return []

    work = df
    if partition_by == "year":
        if date_col not in work.columns:
            raise KeyError(f"date_col={date_col!r} missing for year partitioning")
        work = work.with_columns(pl.col(date_col).dt.year().cast(pl.Int32).alias("_part"))
        part_prefix = "year"
    elif partition_by == "symbol":
        if symbol_col not in work.columns:
            raise KeyError(f"symbol_col={symbol_col!r} missing for symbol partitioning")
        work = work.with_columns(pl.col(symbol_col).cast(pl.Utf8).alias("_part"))
        part_prefix = "symbol"
    else:
        raise ValueError(f"Unsupported partition_by={partition_by!r}")

    written: list[Path] = []
    for key, part in work.group_by("_part", maintain_order=True):
        part_key = key[0] if isinstance(key, tuple) else key
        out_dir = dest_dir / f"{part_prefix}={part_key}"
        if overwrite and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "part.parquet"
        part.drop("_part").write_parquet(out, compression=compression)
        written.append(out)
    return written


def read_partitioned_parquet(
    dataset_dir: Path,
    *,
    years: Sequence[int] | None = None,
    symbols: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read a partitioned Parquet dataset with optional year/symbol filters."""
    if not dataset_dir.exists():
        return pl.DataFrame()

    paths: list[Path] = []
    if years is not None:
        for y in years:
            paths.extend((dataset_dir / f"year={y}").glob("*.parquet"))
    elif symbols is not None:
        for s in symbols:
            paths.extend((dataset_dir / f"symbol={s}").glob("*.parquet"))
    else:
        paths = list(dataset_dir.glob("**/part.parquet")) + list(
            dataset_dir.glob("**/*.parquet")
        )
        # de-dupe while preserving order
        seen: set[Path] = set()
        uniq: list[Path] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        paths = uniq

    if not paths:
        return pl.DataFrame()

    df = pl.concat([pl.read_parquet(p, columns=list(columns) if columns else None) for p in paths], how="diagonal_relaxed")
    return df


def query_df(
    sql: str,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    db_path: Path | None = None,
) -> pl.DataFrame:
    """Run SQL and return a Polars DataFrame."""
    own = con is None
    connection = con or connect(db_path)
    try:
        arrow = connection.execute(sql).fetch_arrow_table()
        return pl.from_arrow(arrow)
    finally:
        if own:
            connection.close()


def register_standard_views(
    con: duckdb.DuckDBPyConnection,
    *,
    processed_dir: Path | None = None,
) -> list[str]:
    """Register common project views if their Parquet dirs exist."""
    root = processed_dir or PROCESSED_DIR
    mapping = {
        "bhavcopy": (root / "bhavcopy", "year"),
        "returns": (root / "returns", "year"),
        "fundamentals": (root / "fundamentals", "year"),
        "universe": (root / "universe", "year"),
        "risk_free": (root / "risk_free", None),
    }
    registered: list[str] = []
    for name, (path, part) in mapping.items():
        if not path.exists():
            continue
        has_parquet = any(path.rglob("*.parquet"))
        if not has_parquet:
            continue
        register_parquet_view(
            con,
            name,
            path,
            partition_by=part,  # type: ignore[arg-type]
            hive_partitioning=part is not None,
        )
        registered.append(name)
    return registered


def list_partitions(dataset_dir: Path, partition_by: PartitionBy = "year") -> list[str]:
    """Return sorted partition key strings present on disk."""
    if not dataset_dir.exists():
        return []
    prefix = f"{partition_by}="
    keys = [p.name[len(prefix) :] for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(keys)


def assert_snappy_parquet(paths: Iterable[Path]) -> None:
    """Raise if any path is missing or not a Parquet file (extension check)."""
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() != ".parquet":
            raise ValueError(f"Expected Parquet artifact, got {p}")
