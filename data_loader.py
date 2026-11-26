"""Fast loading of Excel, CSV and TSV files into Parquet-backed datasets.

Excel is read with Calamine, delimited text with DuckDB, and the result is
always materialized as a compressed Parquet file that the rest of the app
scans lazily with Polars/DuckDB.
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import duckdb
import polars as pl
import streamlit as st

from storage import (
    CACHE_DIR,
    Dataset,
    digest,
    lit,
    parquet_path,
    prune_cache,
    write_parquet,
)

# DuckDB only needs a bounded sample to infer CSV types.  Scanning the whole
# file (-1) roughly quadruples load time on large files.
CSV_SAMPLE_ROWS = 100_000


@dataclass
class LoadOptions:
    delimiter: str = ","
    encoding: str = "utf-8"
    decimal: str = "."
    thousands: Optional[str] = None
    sheet_name: Union[str, int, None] = 0
    header: Optional[int] = 0
    skiprows: int = 0
    nrows: Optional[int] = None
    load_as_text: bool = False
    parse_dates: Optional[List[str]] = None


def _file_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    raise TypeError("Unsupported file source")


def _normalize_text(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the app's Armenian spelling fix on string columns only."""
    string_cols = [c for c, dtype in zip(frame.columns, frame.dtypes) if dtype == pl.String]
    if not string_cols:
        return frame
    return frame.with_columns(
        pl.col(c).str.replace_all("եվ", "և") for c in string_cols
    )


def _dataset_from_parquet(path: str, signature: str) -> Dataset:
    lf = pl.scan_parquet(path)
    names = tuple(lf.collect_schema().names())
    rows = lf.select(pl.len()).collect().item()
    return Dataset(path=path, signature=signature, columns=names, rows=int(rows),
                   size_bytes=Path(path).stat().st_size)


def _excel_rows(data: bytes, options: LoadOptions) -> List[list]:
    """Read the selected sheet with Calamine (no pandas/openpyxl involved)."""
    from python_calamine import CalamineWorkbook

    workbook = CalamineWorkbook.from_filelike(io.BytesIO(data))
    names = list(workbook.sheet_names)
    sheet_name = options.sheet_name
    if isinstance(sheet_name, int):
        index = sheet_name if sheet_name >= 0 else len(names) + sheet_name
        index = max(0, min(index, max(len(names) - 1, 0)))
        sheet = workbook.get_sheet_by_index(index)
    elif sheet_name in names:
        sheet = workbook.get_sheet_by_name(sheet_name)
    elif names:
        sheet = workbook.get_sheet_by_index(0)
    else:
        raise ValueError("The workbook has no worksheets.")

    skiprows = int(options.skiprows or 0)
    header = options.header
    # A preview only needs a handful of rows: stop the Calamine iterator early
    # instead of materializing a whole workbook on every Streamlit rerun.
    stop_after = None
    if options.nrows and options.nrows > 0:
        stop_after = skiprows + (int(header) + 1 if header is not None and header >= 0 else 0)
        stop_after += int(options.nrows)
    if stop_after is not None:
        rows = []
        for index, row in enumerate(sheet.iter_rows()):
            rows.append(list(row))
            if index + 1 >= stop_after:
                break
    else:
        rows = [list(r) for r in sheet.to_python()]
    rows = rows[skiprows:]

    if header is not None and header >= 0:
        header = int(header)
        if header >= len(rows):
            raise ValueError("Header row is outside the sheet.")
        columns = [str(v) for v in rows[header]]
        body = rows[header + 1:]
    else:
        width = max((len(r) for r in rows), default=0)
        columns = [f"column_{i + 1}" for i in range(width)]
        body = rows

    # Calamine returns "" for blank cells.  Turning them into real nulls keeps
    # Polars from appending a str to a numeric/date builder when a blank shows
    # up after the first 100 rows.
    width = len(columns)
    body = [
        [None if v == "" else v for v in (list(r) + [None] * width)[:width]]
        for r in body
    ]
    return [columns] + body


def _frame_from_excel(data: bytes, options: LoadOptions) -> pl.DataFrame:
    rows = _excel_rows(data, options)
    columns, body = rows[0], rows[1:]
    try:
        # Infer from every row: an isolated blank or stray value late in the
        # sheet must not break a type chosen from the first 100 rows.
        frame = pl.DataFrame(body, orient="row", schema=columns,
                             strict=False, infer_schema_length=None)
    except Exception:
        # Mixed dates/numbers/text cannot share one typed builder; a string
        # column is lossless here because all analytics cast from VARCHAR.
        frame = pl.DataFrame(body, orient="row",
                             schema={c: pl.String for c in columns}, strict=False)
    if options.load_as_text:
        frame = frame.select(pl.all().cast(pl.String))
    return frame


def _sniff_delimiter(data: bytes) -> Optional[str]:
    try:
        sample = data[:131072].decode("utf-8", "replace")
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        return None


def _read_csv_to_parquet(data: bytes, options: LoadOptions, separator: str,
                         path: str) -> None:
    """Let DuckDB stream a delimited file straight into Parquet."""
    source = CACHE_DIR / f"source_{digest(data, separator, options)}.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)

    parts = [f"read_csv({lit(source.as_posix())}", f"delim={lit(separator)}",
             f"skip={int(options.skiprows or 0)}",
             f"header={'true' if options.header is not None and options.header >= 0 else 'false'}",
             f"encoding={lit(options.encoding or 'utf-8')}",
             "quote='\"'", "escape='\"'", f"sample_size={CSV_SAMPLE_ROWS}",
             "ignore_errors=true",
             f"all_varchar={'true' if options.load_as_text else 'false'}"]
    if options.decimal and options.decimal != ".":
        parts.append(f"decimal_separator={lit(options.decimal)}")
    if options.thousands:
        parts.append(f"thousands={lit(options.thousands)}")

    select = "SELECT * FROM " + ", ".join(parts) + ")"
    if options.nrows and options.nrows > 0:
        select += f" LIMIT {int(options.nrows)}"
    try:
        duckdb.sql(f"COPY ({select}) TO {lit(path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        try:
            source.unlink()
        except OSError:
            pass


def _normalize_parquet(path: str) -> None:
    """Apply the Armenian spelling fix as a streaming Parquet pass."""
    lazy = pl.scan_parquet(path)
    string_cols = [c for c, dtype in lazy.collect_schema().items() if dtype == pl.String]
    if not string_cols:
        return
    tmp = f"{path}.norm"
    lazy.with_columns(
        pl.col(c).str.replace_all("եվ", "և") for c in string_cols
    ).sink_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


class DataLoader:
    """Load banking datasets with user-controlled options."""

    @staticmethod
    def excel_sheets(uploaded_file) -> List[str]:
        from python_calamine import CalamineWorkbook

        data = _file_bytes(uploaded_file)
        return list(CalamineWorkbook.from_filelike(io.BytesIO(data)).sheet_names)

    @staticmethod
    @st.cache_data(show_spinner="Loading data...", max_entries=8)
    def load(file_bytes: bytes, file_name: str, options: LoadOptions) -> Dataset:
        signature = digest(file_bytes, file_name, options, "parquet-v4")
        path = parquet_path(signature, "data")

        if Path(path).exists():
            return _dataset_from_parquet(path, signature)

        lower = file_name.lower()
        if lower.endswith((".xlsx", ".xls", ".xlsm")):
            frame = _frame_from_excel(file_bytes, options)
            write_parquet(frame, path)
        else:
            separator = "\t" if lower.endswith(".tsv") else (options.delimiter or ",")
            try:
                _read_csv_to_parquet(file_bytes, options, separator, path)
            except Exception:
                guessed = _sniff_delimiter(file_bytes)
                if not guessed or guessed == separator:
                    raise
                separator = guessed
                _read_csv_to_parquet(file_bytes, options, separator, path)
            #_normalize_parquet(path)

        prune_cache()
        return _dataset_from_parquet(path, signature)
