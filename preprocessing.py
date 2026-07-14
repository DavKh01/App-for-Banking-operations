
"""Data cleaning and transformation service."""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd

from utils import safe_datetime, safe_numeric


class DataPreprocessor:
    """Apply configurable cleaning actions to a dataframe."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> "DataPreprocessor":
        self.df = self.df.drop_duplicates(subset=subset)
        return self
    def remove_rows(self, target : str) -> "DataPreprocessor":
        self.df.drop(
            self.df[self.df.map(lambda x: target in str(x).lower()).any(axis=1)].index, inplace=True
        )
        return self
    def fill_missing(self, columns: List[str], value: Any) -> "DataPreprocessor":
        for col in columns:
            self.df[col] = self.df[col].fillna(value)
        return self

    def remove_missing(self, subset: Optional[List[str]] = None) -> "DataPreprocessor":
        self.df = self.df.dropna(subset=subset)
        return self

    def trim_spaces(self, columns: List[str]) -> "DataPreprocessor":
        for col in columns:
            self.df[col] = self.df[col].astype(str).str.strip()
        return self

    def change_case(self, columns: List[str], mode: str) -> "DataPreprocessor":
        for col in columns:
            s = self.df[col].astype(str)
            self.df[col] = s.str.upper() if mode == "upper" else s.str.lower()
        return self

    def replace_values(self, column: str, old: str, new: str) -> "DataPreprocessor":
        self.df[column] = self.df[column].replace(old, new)
        return self

    def rename_columns(self, mapping: Dict[str, str]) -> "DataPreprocessor":
        self.df = self.df.rename(columns={k: v for k, v in mapping.items() if v})
        return self
    def delete_rows(self, column: str, target : str) -> "DataPreprocessor":
        self.df = self.df[~self.df[column].str.lower().str.contains(target)]
        return self

    def convert_dtype(self, column: str, dtype: str) -> "DataPreprocessor":
        if dtype == "numeric":
            self.df[column] = safe_numeric(self.df[column])
        elif dtype == "datetime":
            self.df[column] = safe_datetime(self.df[column])
        elif dtype == "string":
            self.df[column] = self.df[column].astype("string")
        elif dtype == "category":
            self.df[column] = self.df[column].astype("category")
        return self

    def normalize_currency(self, amount_col: str, currency_col: str, rates: Dict[str, float], target_col: str = "normalized_amount") -> "DataPreprocessor":
        amounts = safe_numeric(self.df[amount_col])
        self.df[target_col] = amounts * self.df[currency_col].map(rates).fillna(1.0)
        return self

    def result(self) -> pd.DataFrame:
        return self.df
