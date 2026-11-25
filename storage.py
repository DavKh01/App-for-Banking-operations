"""Parquet-backed dataset storage plus Polars/DuckDB helpers.

The Streamlit app keeps only a small :class:`Dataset` descriptor in memory.
All row-level data lives in a compressed Parquet file on disk, and every heavy
aggregation is pushed down to DuckDB, which scans that file directly.
"""
from __future__ import annotations

import hashlib
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

CACHE_DIR = Path(tempfile.gettempdir()) / "banking_aml_parquet"
_LOCK = threading.Lock()

# Formats accepted by the date parser.  DuckDB/Polars try them in order.
# Single source of truth: preprocessing, risk_engine and graph_analysis import
# this tuple so an added format can never be missed in one code path.
DATE_FORMATS = (
    # day-first (pandas ``dayfirst=True``) — must come before month-first
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M.%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S.%f",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M.%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
    "%d.%m.%Y %H:%M:%S.%f",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M.%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
    # Year first: pandas still puts the DAY first when both middle fields
    # are <= 12, e.g. 2026-11-01 -> 2026-01-11.  Try %Y-%d-%m first and let
    # TRY_STRPTIME fall back to %Y-%m-%d when the day is > 12.
    "%Y-%d-%m %H:%M:%S.%f",
    "%Y-%d-%m %H:%M:%S",
    "%Y-%d-%m %H:%M.%S",
    "%Y-%d-%m %H:%M",
    "%Y-%d-%m",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M.%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%d/%m %H:%M:%S.%f",
    "%Y/%d/%m %H:%M:%S",
    "%Y/%d/%m %H:%M.%S",
    "%Y/%d/%m %H:%M",
    "%Y/%d/%m",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M.%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%d-%mT%H:%M:%S.%f",
    "%Y-%d-%mT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M.%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
)
_DATE_FORMATS = DATE_FORMATS  # backwards-compatible alias
_DATE_FORMAT_SQL = "[" + ", ".join("'" + f + "'" for f in DATE_FORMATS) + "]"

# The dots after HH:MM are hundredths of a MINUTE in the source data
# (e.g. 00:01.66 => 00:01:39), exactly like pandas' ``format="mixed"``
# with dayfirst=True.  Such values can be >= 60, so they must never be
# parsed as seconds.  Parse the base without the fraction, then add
# floor(hundredths * 0.6) seconds.
DATE_BASE_FORMATS = tuple(f for f in DATE_FORMATS if "%M.%S" not in f)
_DATE_BASE_FORMAT_SQL = "[" + ", ".join("'" + f + "'" for f in DATE_BASE_FORMATS) + "]"
# DuckDB expects ``.%f`` for fractional seconds, while Polars/chrono expects
# ``%.f``.  Keep one canonical list and translate only at the Polars boundary
# so no ``ChronoFormatWarning`` is emitted.
_POLARS_DATE_BASE_FORMATS = tuple(f.replace(".%f", "%.f") for f in DATE_BASE_FORMATS)
# Anchor to the start of the time part so ``HH:MM:SS.xx`` (fractional
# SECONDS) is never mistaken for ``HH:MM.xx`` (hundredths of a MINUTE).
_DOT_TIME_RE = r"(^|[ T])([0-9]{1,2}:[0-9]{2})\.[0-9]{1,2}$"
_DOT_FRAC_RE = r"(^|[ T])[0-9]{1,2}:[0-9]{2}\.([0-9]{1,2})$"


def ensure_cache_dir() -> Path:
    """Create the Parquet scratch directory if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def digest(*parts: Any) -> str:
    """Stable short hash used for cache keys and file names."""
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, bytes):
            h.update(part)
        else:
            h.update(repr(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def parquet_path(signature: str, suffix: str = "data") -> str:
    """Deterministic path of the Parquet file backing a dataset."""
    ensure_cache_dir()
    safe_suffix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in suffix)
    return str(CACHE_DIR / f"{signature}_{safe_suffix}.parquet")


def prune_cache(keep: int = 200) -> None:
    """Keep the newest Parquet files so the scratch folder cannot grow forever."""
    with _LOCK:
        ensure_cache_dir()
        files = sorted(CACHE_DIR.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass


def write_parquet(frame: pl.DataFrame, path: str) -> str:
    """Write a Polars frame to Parquet (zstd) atomically enough for local use."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=True)
    return path


@dataclass(frozen=True)
class Dataset:
    """Lightweight descriptor for a dataframe stored as Parquet."""

    path: str
    signature: str
    columns: tuple[str, ...]
    rows: int
    size_bytes: int = 0

    def lazy(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path)

    def frame(self) -> pl.DataFrame:
        return pl.read_parquet(self.path)

    def head(self, n: int = 100) -> pl.DataFrame:
        return pl.read_parquet(self.path, n_rows=n)

    def pandas_head(self, n: int = 100):
        return self.head(n).to_pandas()

    def estimated_size_mb(self) -> float:
        sample = self.head(1000)
        if not self.rows:
            return 0.0
        return float(sample.estimated_size("mb") / max(len(sample), 1) * self.rows)

    def __len__(self) -> int:
        return int(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.rows), len(self.columns)


def make_dataset(frame: pl.DataFrame, signature: str, suffix: str = "data") -> Dataset:
    """Persist a frame to Parquet and return its descriptor."""
    path = parquet_path(signature, suffix)
    write_parquet(frame, path)
    return Dataset(
        path=path,
        signature=signature,
        columns=tuple(frame.columns),
        rows=frame.height,
        size_bytes=Path(path).stat().st_size,
    )


def to_polars(frame: Any) -> pl.DataFrame:
    """Normalize pandas / Polars / LazyFrame inputs to a Polars DataFrame."""
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    if isinstance(frame, pl.DataFrame):
        return frame
    return pl.from_pandas(frame)


def to_pandas(frame: Any):
    """Convert a Polars frame to pandas only at the UI/plot boundary."""
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    return frame


def q(column: str) -> str:
    """Quote a (possibly Armenian / spaced) column identifier for SQL."""
    return '"' + str(column).replace('"', '""') + '"'


def lit(value: Any) -> str:
    """Quote a string literal for SQL."""
    return "'" + str(value).replace("'", "''") + "'"


def sql_source(ds: Dataset) -> str:
    """DuckDB table expression reading the dataset's Parquet file."""
    path = ds.path.replace("\\", "/").replace("'", "''")
    return f"read_parquet('{path}')"


def num_expr(column: str) -> str:
    """SQL expression matching utils.safe_numeric (strip commas and spaces)."""
    c = q(column)
    cleaned = f"REPLACE(REPLACE(CAST({c} AS VARCHAR), ',', ''), ' ', '')"
    return f"TRY_CAST(NULLIF(TRIM({cleaned}), '') AS DOUBLE)"


def date_expr(column: str) -> str:
    """SQL expression matching pandas safe_datetime (day-first, errors -> null).

    A trailing ``HH:MM.xx`` fraction is hundredths of a minute, not seconds.
    """
    raw = f"TRIM(CAST({q(column)} AS VARCHAR))"
    base = f"REGEXP_REPLACE({raw}, '{_DOT_TIME_RE}', '\\1\\2')"
    frac = f"COALESCE(TRY_CAST(REGEXP_EXTRACT({raw}, '{_DOT_FRAC_RE}', 2) AS DOUBLE), 0)"
    parsed = f"TRY_STRPTIME({base}, {_DATE_BASE_FORMAT_SQL})"
    return f"({parsed} + to_seconds(FLOOR({frac} * 0.6)))"


def date_col_expr(column: str, dtype: Any | None = None) -> pl.Expr:
    """Polars equivalent of :func:`date_expr` (pandas-compatible dot fraction)."""
    raw = pl.col(column)
    if isinstance(dtype, (pl.Datetime, pl.Date)):
        return raw.cast(pl.Datetime, strict=False)
    text = raw.cast(pl.String).str.strip_chars()
    base = text.str.replace(_DOT_TIME_RE, "$1$2")
    fraction = text.str.extract(_DOT_FRAC_RE, 2).cast(pl.Float64).fill_null(0.0)
    parsed = pl.coalesce([
        base.str.strptime(pl.Datetime, format=fmt, strict=False)
        for fmt in _POLARS_DATE_BASE_FORMATS
    ])
    return parsed + pl.duration(seconds=(fraction * 0.6).floor())


def query(sql: str, frames: dict[str, Any] | None = None) -> pl.DataFrame:
    """Run a DuckDB query and return a Polars DataFrame."""
    with duckdb.connect() as con:
        for name, frame in (frames or {}).items():
            con.register(name, to_polars(frame))
        try:
            return con.sql(sql).pl()
        except duckdb.Error as exc:
            if "PlainSkip" not in str(exc):
                raise
            # DuckDB 1.5 cannot skip rows in an all-null / PLAIN-encoded
            # Parquet column during filter pushdown.  The same query works
            # when filters are applied after the scan, so retry that way
            # instead of failing the whole analysis.
            con.execute("SET disabled_optimizers='filter_pushdown'")
            return con.sql(sql).pl()
