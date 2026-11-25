"""Shared utility helpers (Polars/DuckDB backed)."""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, Iterable, Optional

import polars as pl
import streamlit as st

from storage import Dataset, date_expr, q, query, sql_source, to_pandas, to_polars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def normalize_mapping(mapping: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Remove empty selections from a Streamlit column mapping."""
    return {k: v for k, v in mapping.items() if v and v != "<Not available>"}


def get_col(mapping: Dict[str, str], key: str) -> Optional[str]:
    """Return mapped dataframe column for canonical field key."""
    return mapping.get(key)


def safe_numeric(values: Any) -> pl.Series:
    """Convert values to Float64 safely (commas and spaces are ignored)."""
    if isinstance(values, pl.Series):
        s = values
    else:
        s = pl.Series(values)
    return (
        s.cast(pl.String)
        .str.replace_all(",", "")
        .str.replace_all(" ", "")
        .cast(pl.Float64, strict=False)
    )


def safe_datetime(values: Any) -> pl.Series:
    """Convert values to datetime safely using the shared day-first parser."""
    s = values if isinstance(values, pl.Series) else pl.Series(values)
    frame = pl.DataFrame({"_value": s})
    result = query(f"SELECT {date_expr('_value')} AS _value FROM frame", {"frame": frame})
    return result["_value"]


def memory_usage_mb(df: Any) -> float:
    """Return dataset memory usage in MB (Parquet-backed estimate)."""
    if isinstance(df, Dataset):
        return df.estimated_size_mb()
    return float(to_polars(df).estimated_size("mb"))


_PANDAS_DTYPES = {
    pl.String: "object",
    pl.Utf8: "object",
    pl.Int64: "int64",
    pl.Int32: "int32",
    pl.Int16: "int16",
    pl.Int8: "int8",
    pl.UInt64: "uint64",
    pl.UInt32: "uint32",
    pl.Float64: "float64",
    pl.Float32: "float32",
    pl.Boolean: "bool",
    pl.Date: "object",
    pl.Datetime: "datetime64[ns]",
}


def _dtype_label(dtype: pl.DataType) -> str:
    for cls, label in _PANDAS_DTYPES.items():
        try:
            if dtype == cls or isinstance(dtype, cls):
                return label
        except TypeError:
            continue
    return str(dtype)


def dataframe_info(df: Any) -> pl.DataFrame:
    """Create a dataframe similar to pandas info output, in a single scan."""
    if isinstance(df, Dataset):
        frame = df
    else:
        frame = to_polars(df)
    schema = frame.lazy().collect_schema()
    columns = list(schema.names())
    source = sql_source(frame) if isinstance(frame, Dataset) else "frame"

    aggregates = []
    for i, column in enumerate(columns):
        c = q(column)
        aggregates += [
            f"count({c}) AS nn_{i}",
            f"count(DISTINCT {c}) AS u_{i}",
        ]
    frames = None if isinstance(frame, Dataset) else {"frame": frame}
    total = query(f"SELECT count(*) AS n, {', '.join(aggregates)} FROM {source}", frames) if columns \
        else query(f"SELECT count(*) AS n FROM {source}", frames)
    row = total.row(0, named=True) if total.height else {"n": 0}
    n = int(row["n"])

    rows = []
    for i, column in enumerate(columns):
        non_null = int(row[f"nn_{i}"] or 0)
        nulls = n - non_null
        rows.append({
            "column": column,
            "dtype": _dtype_label(schema[column]),
            "non_null": non_null,
            "nulls": nulls,
            "null_pct": round(nulls / n * 100, 2) if n else 0.0,
            "unique": int(row[f"u_{i}"] or 0),
        })
    return pl.DataFrame(rows)


def download_dataframe(df: Any, filename_stem: str = "export") -> None:
    """Excel download built with xlsxwriter's constant-memory mode.

    Polars' ``write_excel`` always creates an Excel *table*, which xlsxwriter
    does not allow in constant-memory mode (the sheet ends up empty).  Writing
    the rows directly keeps memory flat and produces a valid workbook.
    """
    buffer = io.BytesIO()
    frame = to_polars(df) if not isinstance(df, Dataset) else df.frame()
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    frame = frame.select(pl.all().cast(pl.String))
    try:
        import xlsxwriter

        book = xlsxwriter.Workbook(buffer, {"constant_memory": True, "use_zip64": True})
        sheet = book.add_worksheet("Data")
        sheet.write_row(0, 0, [str(c) for c in frame.columns])
        max_rows = 1_048_575  # Excel sheet limit, excluding the header row
        for row_no, row in enumerate(frame.iter_rows(), start=1):
            if row_no > max_rows:
                st.warning(f"Excel export truncated to {max_rows:,} rows.")
                break
            sheet.write_row(row_no, 0, ["" if v is None else str(v) for v in row])
        book.close()
    except Exception:  # pragma: no cover - fallback when xlsxwriter is missing
        buffer = io.BytesIO()
        frame.write_csv(buffer)
        st.download_button("Download CSV", buffer.getvalue(), f"{filename_stem}.csv", "text/csv")
        return

    st.download_button(
        "Download Excel",
        buffer.getvalue(),
        f"{filename_stem}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def download_csv(df: Any, filename_stem: str = "export") -> None:
    """CSV download for the same data that :func:`download_dataframe` exports."""
    frame = to_polars(df) if not isinstance(df, Dataset) else df.frame()
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    buffer = io.BytesIO()
    frame.write_csv(buffer)
    st.download_button("Download CSV", buffer.getvalue(), f"{filename_stem}.csv", "text/csv")


def require_columns(mapping: Dict[str, str], keys: Iterable[str]) -> bool:
    """Check whether required canonical fields are mapped."""
    missing = [k for k in keys if k not in mapping]
    if missing:
        st.warning(f"Map required fields first: {', '.join(missing)}")
        return False
    return True


def dataframe_pdf_bytes(df: Any, title: str = "AML Report") -> bytes:
    """Create a compact PDF preview report from a dataframe."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    frame = to_pandas(to_polars(df).head(40)).astype(str)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    data = [frame.columns.tolist()] + frame.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
    ]))
    rows = len(df) if not isinstance(df, Dataset) else len(df)
    cols = frame.shape[1]
    doc.build([Paragraph(title, styles["Title"]), Spacer(1, 8),
               Paragraph(f"Rows: {rows:,} | Columns: {cols:,}", styles["Normal"]),
               Spacer(1, 8), table])
    return buffer.getvalue()


def dataframe_pptx_bytes(df: Any, title: str = "AML Analytics Report") -> bytes:
    """Create a basic PowerPoint report with dataset summary."""
    from pptx import Presentation
    from pptx.util import Inches

    rows = len(df)
    cols = len(df.columns) if hasattr(df, "columns") else 0
    preview = to_pandas(to_polars(df).head(10)).astype(str)

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = f"Rows: {rows:,} | Columns: {cols:,} | Generated by Streamlit AML app"
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Data Preview"
    n_rows, n_cols = preview.shape[0] + 1, min(preview.shape[1], 6)
    table = slide.shapes.add_table(n_rows, n_cols, Inches(0.3), Inches(1.2),
                                   Inches(9.0), Inches(4.5)).table
    for j, col in enumerate(preview.columns[:n_cols]):
        table.cell(0, j).text = str(col)
    for i, (_, row) in enumerate(preview.iterrows(), start=1):
        for j, val in enumerate(row.iloc[:n_cols]):
            table.cell(i, j).text = str(val)[:40]
    buffer = io.BytesIO()
    prs.save(buffer)
    return buffer.getvalue()
