"""Data cleaning and transformation service (Polars, Parquet-backed)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import polars as pl

from storage import Dataset, date_col_expr, digest, make_dataset


class DataPreprocessor:
    """Apply configurable cleaning actions to a dataset."""

    def __init__(self, df: Dataset):
        self.df = df
        self._lazy = df.lazy()
        self._ops: List[Any] = []

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> "DataPreprocessor":
        self._lazy = self._lazy.unique(subset=subset, keep="first")
        self._ops.append(("dedupe", subset))
        return self

    def remove_rows(self, target: str) -> "DataPreprocessor":
        needle = str(target).lower()
        hit = pl.any_horizontal(
            pl.all().cast(pl.String).fill_null("nan").str.to_lowercase().str.contains(needle, literal=True)
        ).fill_null(False)
        self._lazy = self._lazy.filter(~hit)
        self._ops.append(("remove_rows", needle))
        return self

    def fill_missing(self, columns: List[str], value: Any) -> "DataPreprocessor":
        self._lazy = self._lazy.with_columns(pl.col(c).fill_null(value) for c in columns)
        self._ops.append(("fill", columns, value))
        return self

    def remove_missing(self, subset: Optional[List[str]] = None) -> "DataPreprocessor":
        self._lazy = self._lazy.drop_nulls(subset=subset)
        self._ops.append(("drop_nulls", subset))
        return self

    def trim_spaces(self, columns: List[str]) -> "DataPreprocessor":
        self._lazy = self._lazy.with_columns(
            pl.col(c).cast(pl.String).fill_null("nan").str.strip_chars() for c in columns
        )
        self._ops.append(("trim", columns))
        return self

    def change_case(self, columns: List[str], mode: str) -> "DataPreprocessor":
        exprs = []
        for c in columns:
            s = pl.col(c).cast(pl.String).fill_null("nan")
            exprs.append(s.str.to_uppercase() if mode == "upper" else s.str.to_lowercase())
        self._lazy = self._lazy.with_columns(exprs)
        self._ops.append(("case", columns, mode))
        return self

    def replace_values(self, column: str, old: str, new: str) -> "DataPreprocessor":
        self._lazy = self._lazy.with_columns(
            pl.when(pl.col(column) == old).then(pl.lit(new)).otherwise(pl.col(column)).alias(column)
        )
        self._ops.append(("replace", column, old, new))
        return self

    def rename_columns(self, mapping: Dict[str, str]) -> "DataPreprocessor":
        rename = {k: v for k, v in mapping.items() if v}
        self._lazy = self._lazy.rename(rename)
        self._ops.append(("rename", rename))
        return self

    def delete_rows(self, column: str, target: str) -> "DataPreprocessor":
        hit = pl.col(column).cast(pl.String).str.to_lowercase().str.contains(target).fill_null(False)
        self._lazy = self._lazy.filter(~hit)
        self._ops.append(("delete_rows", column, target))
        return self

    def convert_dtype(self, column: str, dtype: str) -> "DataPreprocessor":
        if dtype == "numeric":
            expr = pl.col(column).cast(pl.String).str.replace_all(",", "").str.replace_all(" ", "").cast(pl.Float64, strict=False)
        elif dtype == "datetime":
            expr = date_col_expr(column, self._lazy.collect_schema().get(column))
        elif dtype == "string":
            expr = pl.col(column).cast(pl.String)
        else:
            expr = pl.col(column).cast(pl.Categorical)
        self._lazy = self._lazy.with_columns(expr.alias(column))
        self._ops.append(("dtype", column, dtype))
        return self

    def normalize_currency(self, amount_col: str, currency_col: str, rates: Dict[str, float],
                           target_col: str = "normalized_amount") -> "DataPreprocessor":
        rate = pl.col(currency_col).cast(pl.String).replace_strict(
            {str(k): float(v) for k, v in rates.items()}, default=1.0, return_dtype=pl.Float64
        )
        self._lazy = self._lazy.with_columns(
            (pl.col(amount_col).cast(pl.String).str.replace_all(",", "").str.replace_all(" ", "")
             .cast(pl.Float64, strict=False) * rate).alias(target_col)
        )
        self._ops.append(("normalize_currency", amount_col, currency_col, rates))
        return self

    def result(self) -> Dataset:
        frame = self._lazy.collect()
        signature = digest(self.df.signature, "clean", self._ops)
        return make_dataset(frame, signature, "clean")
