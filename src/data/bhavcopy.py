"""NSE equity Bhavcopy download + dual-format parsers.

Supports:
  * Old format: ``cm<DD><MMM><YYYY>bhav.csv.zip``
  * UDiFF CM: ``BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip``
  * PR daily package: ``PR<DD><MM><YY>.zip`` (equity CSV inside, if present)

Canonical output schema (EQ/BE only)::

    trade_date, symbol, series, open, high, low, close, last,
    prev_close, tottrdqty, tottrdval, timestamp
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Iterable, Literal

import polars as pl
import requests
from tqdm import tqdm

from src.data.paths import RAW_BHAVCOPY_DIR, PROCESSED_BHAVCOPY_DIR, ensure_data_dirs
from src.data.symbols import SymbolMap

logger = logging.getLogger(__name__)

ALLOWED_SERIES: frozenset[str] = frozenset({"EQ", "BE"})
UDIFF_START: date = date(2024, 7, 8)

CANONICAL_COLUMNS: list[str] = [
    "trade_date",
    "symbol",
    "series",
    "open",
    "high",
    "low",
    "close",
    "last",
    "prev_close",
    "tottrdqty",
    "tottrdval",
    "timestamp",
]

FormatKind = Literal["old", "udiff", "pr", "auto"]

_OLD_COLMAP: dict[str, str] = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "LAST": "last",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "tottrdqty",
    "TOTTRDVAL": "tottrdval",
    "TIMESTAMP": "timestamp",
    "ISIN": "isin",
}

_UDIFF_COLMAP: dict[str, str] = {
    "TradDt": "trade_date",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "tottrdqty",
    "TtlTrfVal": "tottrdval",
    "ISIN": "isin",
}

_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.nseindia.com/all-reports",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_DEFAULT_HEADERS)
    return s


def old_bhavcopy_filename(d: date) -> str:
    """Legacy archive name, e.g. ``cm02JAN2020bhav.csv.zip``."""
    return f"cm{d.strftime('%d')}{d.strftime('%b').upper()}{d.strftime('%Y')}bhav.csv.zip"


def udiff_bhavcopy_filename(d: date) -> str:
    return f"BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"


def pr_filename(d: date) -> str:
    return f"PR{d.strftime('%d%m%y')}.zip"


def old_bhavcopy_url(d: date) -> str:
    yyyy = d.strftime("%Y")
    mmm = d.strftime("%b").upper()
    fname = f"cm{d.strftime('%d')}{mmm}{yyyy}bhav.csv.zip"
    return (
        "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
        f"{yyyy}/{mmm}/{fname}"
    )


def udiff_bhavcopy_url(d: date) -> str:
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"{udiff_bhavcopy_filename(d)}"
    )


def detect_format(columns: Iterable[str]) -> FormatKind:
    cols = set(columns)
    if {"TckrSymb", "SctySrs", "OpnPric"} <= cols or {"TradDt", "TckrSymb"} <= cols:
        return "udiff"
    if {"SYMBOL", "SERIES", "OPEN", "CLOSE"} <= cols:
        return "old"
    upper = {c.upper() for c in cols}
    if {"SYMBOL", "SERIES", "OPEN", "CLOSE"} <= upper:
        return "old"
    return "auto"


def _normalize_numeric(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    exprs = []
    for c in cols:
        if c in df.columns:
            exprs.append(
                pl.col(c)
                .cast(pl.Utf8, strict=False)
                .str.replace_all(",", "")
                .cast(pl.Float64, strict=False)
                .alias(c)
            )
    return df.with_columns(exprs) if exprs else df


def _finalize_canonical(
    df: pl.DataFrame,
    *,
    trade_date: date | None,
    symbol_map: SymbolMap,
) -> pl.DataFrame:
    """Map to canonical schema, filter EQ/BE, normalize symbols."""
    # Ensure required columns exist
    for col, dtype in (
        ("last", pl.Float64),
        ("prev_close", pl.Float64),
        ("tottrdqty", pl.Float64),
        ("tottrdval", pl.Float64),
        ("timestamp", pl.Utf8),
    ):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    if "trade_date" not in df.columns or df["trade_date"].null_count() == df.height:
        if trade_date is None:
            raise ValueError("trade_date missing and no fallback date provided")
        df = df.with_columns(pl.lit(trade_date).alias("trade_date"))
    else:
        df = df.with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8, strict=False)
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .fill_null(
                pl.col("trade_date")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, format="%d-%b-%Y", strict=False)
            )
            .fill_null(
                pl.col("trade_date")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, format="%d-%m-%Y", strict=False)
            )
            .alias("trade_date")
        )
        if trade_date is not None:
            df = df.with_columns(pl.col("trade_date").fill_null(trade_date))

    df = _normalize_numeric(
        df,
        ["open", "high", "low", "close", "last", "prev_close", "tottrdqty", "tottrdval"],
    )
    df = symbol_map.normalize_frame(df, "symbol")
    df = df.with_columns(pl.col("series").cast(pl.Utf8).str.strip_chars().str.to_uppercase())
    df = df.filter(pl.col("series").is_in(list(ALLOWED_SERIES)))

    # Prefer EQ over BE when both exist for same symbol/date
    df = (
        df.with_columns(
            pl.when(pl.col("series") == "EQ")
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_series_rank")
        )
        .sort(["trade_date", "symbol", "_series_rank"])
        .unique(subset=["trade_date", "symbol"], keep="first")
        .drop("_series_rank")
    )

    out_cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    return df.select(out_cols).select(CANONICAL_COLUMNS)


def parse_old_bhavcopy(
    source: str | Path | BinaryIO | bytes,
    *,
    trade_date: date | None = None,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Parse legacy ``cm*bhav.csv`` (or its zip) into the canonical schema."""
    sm = symbol_map or SymbolMap()
    df = _read_csv_from_source(source)
    # Column names are often uppercase with trailing spaces
    df = df.rename({c: c.strip().upper() for c in df.columns})
    rename = {k: v for k, v in _OLD_COLMAP.items() if k in df.columns}
    df = df.rename(rename)
    return _finalize_canonical(df, trade_date=trade_date, symbol_map=sm)


def parse_udiff_bhavcopy(
    source: str | Path | BinaryIO | bytes,
    *,
    trade_date: date | None = None,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Parse UDiFF CM Bhavcopy CSV/zip into the canonical schema."""
    sm = symbol_map or SymbolMap()
    df = _read_csv_from_source(source)
    df = df.rename({c: c.strip() for c in df.columns})
    rename = {k: v for k, v in _UDIFF_COLMAP.items() if k in df.columns}
    df = df.rename(rename)
    return _finalize_canonical(df, trade_date=trade_date, symbol_map=sm)


def parse_bhavcopy(
    source: str | Path | BinaryIO | bytes,
    *,
    trade_date: date | None = None,
    fmt: FormatKind = "auto",
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Auto-detect format and parse to canonical EQ/BE panel."""
    sm = symbol_map or SymbolMap()
    # PR zips may contain multiple members — extract best equity CSV first
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".zip" and path.name.upper().startswith("PR"):
            return parse_pr_zip(path, trade_date=trade_date, symbol_map=sm)

    df_raw = _read_csv_from_source(source, peek_only=False)
    kind = fmt if fmt != "auto" else detect_format(df_raw.columns)
    if kind == "udiff":
        return parse_udiff_bhavcopy(source, trade_date=trade_date, symbol_map=sm)
    if kind in ("old", "auto"):
        # Re-parse via old mapper (handles case folding)
        return parse_old_bhavcopy(source, trade_date=trade_date, symbol_map=sm)
    raise ValueError(f"Unsupported bhavcopy format: {kind}")


def parse_pr_zip(
    path: Path,
    *,
    trade_date: date | None = None,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Parse ``PR<DD><MM><YY>.zip``, locating an equity bhav-like CSV inside."""
    sm = symbol_map or SymbolMap()
    with zipfile.ZipFile(path, "r") as zf:
        candidates = [
            n
            for n in zf.namelist()
            if n.lower().endswith((".csv", ".txt")) and not n.endswith("/")
        ]
        if not candidates:
            raise ValueError(f"No CSV/TXT members in PR zip: {path}")

        # Prefer names that look like bhav / security / Pd files
        def score(name: str) -> int:
            u = name.upper()
            s = 0
            for token in ("BHAV", "PD", "SEC", "CM", "EQ"):
                if token in u:
                    s += 1
            return s

        candidates.sort(key=score, reverse=True)
        last_err: Exception | None = None
        for name in candidates:
            try:
                data = zf.read(name)
                return parse_bhavcopy(
                    data, trade_date=trade_date, fmt="auto", symbol_map=sm
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise ValueError(f"Could not parse equity CSV from {path}: {last_err}")


def download_bhavcopy(
    d: date,
    *,
    dest_dir: Path | None = None,
    fmt: FormatKind = "auto",
    session: requests.Session | None = None,
    timeout: float = 30.0,
    skip_if_present: bool = True,
) -> Path:
    """Download one trading day's bhavcopy zip into ``data/raw/bhavcopy``."""
    ensure_data_dirs()
    dest_dir = dest_dir or RAW_BHAVCOPY_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = session or _session()

    # Prefer UDiFF on/after switch date; else old cm zip.
    if fmt == "auto":
        order: list[tuple[FormatKind, str, str]] = []
        if d >= UDIFF_START:
            order.append(
                ("udiff", udiff_bhavcopy_url(d), udiff_bhavcopy_filename(d))
            )
        order.append(("old", old_bhavcopy_url(d), old_bhavcopy_filename(d)))
    elif fmt == "udiff":
        order = [("udiff", udiff_bhavcopy_url(d), udiff_bhavcopy_filename(d))]
    elif fmt == "old":
        order = [("old", old_bhavcopy_url(d), old_bhavcopy_filename(d))]
    elif fmt == "pr":
        # PR packages are typically served via NSE reports UI; try archives path pattern
        fname = pr_filename(d)
        url = f"https://nsearchives.nseindia.com/archives/equities/bhavcopy/{fname}"
        order = [("pr", url, fname)]
    else:
        raise ValueError(f"Unknown format: {fmt}")

    last_err: Exception | None = None
    for kind, url, fname in order:
        out = dest_dir / fname
        if out.exists() and skip_if_present and out.stat().st_size > 0:
            return out
        try:
            # Cookie warm-up
            sess.get("https://www.nseindia.com", timeout=timeout)
            resp = sess.get(url, timeout=timeout)
            if resp.status_code != 200 or not resp.content:
                last_err = RuntimeError(
                    f"HTTP status={resp.status_code} url={url}"
                )
                continue
            # Basic zip magic check
            if not resp.content.startswith(b"PK"):
                last_err = RuntimeError(f"Not a zip from {url}")
                continue
            out.write_bytes(resp.content)
            logger.info("Downloaded %s (%s)", out.name, kind)
            return out
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Download failed for %s via %s: %s", d, kind, exc)

    raise RuntimeError(f"Failed to download bhavcopy for {d}: {last_err}")


def download_bhavcopy_range(
    start: date,
    end: date,
    *,
    dest_dir: Path | None = None,
    skip_weekends: bool = True,
    skip_if_present: bool = True,
) -> list[Path]:
    """Download bhavcopies over an inclusive date range (best-effort per day)."""
    paths: list[Path] = []
    d = start
    dates: list[date] = []
    while d <= end:
        if not (skip_weekends and d.weekday() >= 5):
            dates.append(d)
        d += timedelta(days=1)

    sess = _session()
    for day in tqdm(dates, desc="bhavcopy"):
        try:
            paths.append(
                download_bhavcopy(
                    day,
                    dest_dir=dest_dir,
                    session=sess,
                    skip_if_present=skip_if_present,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", day, exc)
    return paths


def load_bhavcopy_file(
    path: Path,
    *,
    trade_date: date | None = None,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Parse a single on-disk bhavcopy zip/csv."""
    sm = symbol_map or SymbolMap()
    inferred = trade_date or _infer_date_from_filename(path.name)
    return parse_bhavcopy(path, trade_date=inferred, symbol_map=sm)


def load_bhavcopy_dir(
    directory: Path | None = None,
    *,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Parse all bhavcopy archives in a directory into one panel."""
    sm = symbol_map or SymbolMap()
    directory = directory or RAW_BHAVCOPY_DIR
    files = sorted(
        list(directory.glob("*.zip"))
        + list(directory.glob("*.csv"))
        + list(directory.glob("*.CSV"))
    )
    if not files:
        return pl.DataFrame(schema={c: pl.Float64 for c in CANONICAL_COLUMNS}).clear()

    frames = [load_bhavcopy_file(p, symbol_map=sm) for p in files]
    return pl.concat(frames, how="vertical_relaxed").sort(["trade_date", "symbol"])


def write_bhavcopy_parquet(
    df: pl.DataFrame,
    dest_dir: Path | None = None,
) -> list[Path]:
    """Write canonical bhavcopy panel as year-partitioned snappy Parquet."""
    ensure_data_dirs()
    dest_dir = dest_dir or PROCESSED_BHAVCOPY_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        return []

    df = df.with_columns(pl.col("trade_date").dt.year().alias("year"))
    written: list[Path] = []
    for key, part in df.group_by("year", maintain_order=True):
        y = key[0] if isinstance(key, tuple) else key
        out_dir = dest_dir / f"year={y}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "part.parquet"
        part.drop("year").write_parquet(out, compression="snappy")
        written.append(out)
    return written


def _read_csv_from_source(
    source: str | Path | BinaryIO | bytes,
    *,
    peek_only: bool = False,  # noqa: ARG001 — reserved
) -> pl.DataFrame:
    if isinstance(source, bytes):
        return _read_bytes(source)
    if hasattr(source, "read") and not isinstance(source, (str, Path)):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _read_bytes(data)
    path = Path(source)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            name = _pick_csv_member(zf.namelist())
            return _read_bytes(zf.read(name))
    return pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)


def _read_bytes(data: bytes) -> pl.DataFrame:
    if data.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            name = _pick_csv_member(zf.namelist())
            data = zf.read(name)
    return pl.read_csv(io.BytesIO(data), infer_schema_length=10_000, ignore_errors=True)


def _pick_csv_member(names: list[str]) -> str:
    csvs = [n for n in names if n.lower().endswith(".csv") and not n.endswith("/")]
    if not csvs:
        # Some archives use .CSV or bare names
        csvs = [n for n in names if not n.endswith("/")]
    if not csvs:
        raise ValueError("Zip archive contains no readable members")
    return csvs[0]


def _infer_date_from_filename(name: str) -> date | None:
    upper = name.upper()
    # cm02JAN2020bhav.csv.zip
    if upper.startswith("CM") and "BHAV" in upper:
        core = upper[2:].split("BHAV")[0]
        try:
            return datetime.strptime(core, "%d%b%Y").date()
        except ValueError:
            pass
    # BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip
    if "BHAVCOPY_NSE_CM" in upper.replace("-", "_"):
        for token in upper.replace(".CSV.ZIP", "").replace(".ZIP", "").split("_"):
            if len(token) == 8 and token.isdigit():
                try:
                    return datetime.strptime(token, "%Y%m%d").date()
                except ValueError:
                    pass
    # PR020124.zip
    if upper.startswith("PR") and len(upper) >= 8:
        token = upper[2:8]
        try:
            return datetime.strptime(token, "%d%m%y").date()
        except ValueError:
            pass
    return None
