
"""Robust loading of CSV, TSV and Excel files."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, List

import pandas as pd
import streamlit as st


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


class DataLoader:
    """Load banking datasets with user-controlled options."""

    @staticmethod
    def excel_sheets(uploaded_file) -> List[str]:
        xls = pd.ExcelFile(uploaded_file)
        return xls.sheet_names

    @staticmethod
    @st.cache_data(show_spinner="Loading data...")
    def load(file_bytes: bytes, file_name: str, options: LoadOptions) -> pd.DataFrame:
        import io
        dtype = str if options.load_as_text else None
        nrows = options.nrows if options.nrows and options.nrows > 0 else None
        header = options.header if options.header is not None and options.header >= 0 else None
        thousands = options.thousands if options.thousands else None
        parse_dates = options.parse_dates or False
        buffer = io.BytesIO(file_bytes)
        lower = file_name.lower()
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(
                buffer, sheet_name=options.sheet_name, header=header, skiprows=options.skiprows,
                nrows=nrows, dtype=dtype, parse_dates=parse_dates, engine="openpyxl" if lower.endswith(".xlsx") else None,
            )
        sep = "\t" if lower.endswith(".tsv") else options.delimiter
        return pd.read_csv(
            buffer, sep=sep, encoding=options.encoding, decimal=options.decimal,
            thousands=thousands, header=header, skiprows=options.skiprows, nrows=nrows,
            dtype=dtype, parse_dates=parse_dates, low_memory=False,
        )
