"""Configurable AML risk scoring engine (vectorized with Polars)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import polars as pl
import streamlit as st

from config import DEFAULT_RISK_LEVELS
from storage import Dataset, date_col_expr
from utils import get_col


def _level_expr(score: pl.Expr, levels: Dict[str, tuple]) -> pl.Expr:
    expr = None
    for level, (lo, hi) in levels.items():
        cond = (score >= float(lo)) & (score <= float(hi))
        expr = pl.when(cond).then(pl.lit(level)) if expr is None else expr.when(cond).then(pl.lit(level))
    return expr.otherwise(
        pl.when(score > 100).then(pl.lit("Critical")).otherwise(pl.lit("Low"))
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _score(ds: Dataset, mapping: Dict[str, str], rules: Dict[str, Dict[str, Any]],
           levels: Dict[str, tuple]) -> pl.DataFrame:
    lazy = ds.lazy()
    columns = set(ds.columns)

    amount_col = get_col(mapping, "amount")
    date_col = get_col(mapping, "transaction_date")
    debit = get_col(mapping, "debit_account")
    credit = get_col(mapping, "credit_account")
    doc = get_col(mapping, "document_number")
    cash = get_col(mapping, "cash_flag")
    currency = get_col(mapping, "currency")
    customer = get_col(mapping, "customer_id") or debit

    amt = None
    if amount_col and amount_col in columns:
        amt = (
            pl.col(amount_col).cast(pl.String)
            .str.replace_all(",", "").str.replace_all(" ", "")
            .cast(pl.Float64, strict=False).abs()
        )
    dates = None
    if date_col and date_col in columns:
        dates = date_col_expr(date_col, lazy.collect_schema().get(date_col))

    score_expr = pl.lit(0.0)
    reasons: List[pl.Expr] = []

    def apply_rule(name: str, mask: pl.Expr | None) -> None:
        nonlocal score_expr
        rule = rules.get(name, {})
        if mask is None or not rule.get("enabled", False):
            return
        weight = float(rule.get("weight", 0))
        desc = rule.get("description", name)
        mask = mask.fill_null(False)
        score_expr = score_expr + pl.when(mask).then(pl.lit(weight)).otherwise(pl.lit(0.0))
        reasons.append(
            pl.when(mask)
            .then(pl.lit(f"{name}: {desc} (+{weight:g})"))
            .otherwise(pl.lit(""))
        )

    if amt is not None:
        threshold = float(rules.get("large_amount", {}).get("threshold", 0))
        apply_rule("large_amount", amt >= threshold)
        round_threshold = float(rules.get("round_amount", {}).get("threshold", 1000))
        apply_rule("round_amount", (amt >= round_threshold) & ((amt % round_threshold) == 0))

    if cash and cash in columns:
        apply_rule("cash_transaction", pl.col(cash).is_not_null())

    if dates is not None:
        apply_rule("night_activity", dates.dt.hour().is_between(0, 5, closed="both"))
        # Polars weekday(): Monday=1 ... Saturday=6, Sunday=7
        # (DuckDB dayofweek() in analysis.py uses Sunday=0 ... Saturday=6)
        apply_rule("weekend_activity", dates.dt.weekday().is_in([6, 7]))

    if debit and debit in columns:
        cnt = pl.col(debit).count().over(debit)
        threshold = float(rules.get("high_frequency_sender", {}).get("threshold", 20))
        apply_rule("high_frequency_sender", (cnt >= threshold) & pl.col(debit).is_not_null())

    if debit and credit and debit in columns and credit in columns:
        cpty = pl.col(credit).drop_nulls().n_unique().over(debit)
        threshold = float(rules.get("many_counterparties", {}).get("threshold", 15))
        apply_rule("many_counterparties", (cpty >= threshold) & pl.col(debit).is_not_null())

    if currency and customer and currency in columns and customer in columns:
        ccount = pl.col(currency).drop_nulls().n_unique().over(customer)
        threshold = float(rules.get("multiple_currencies", {}).get("threshold", 2))
        apply_rule("multiple_currencies", (ccount >= threshold) & pl.col(customer).is_not_null())

    if reasons:
        joined = (
            pl.concat_list(reasons)
            .list.filter(pl.element().str.len_chars() > 0)
            .list.join("; ")
        )
        reason_expr = pl.when(joined.str.len_chars() > 0).then(joined).otherwise(
            pl.lit("No enabled rule triggered")
        )
    else:
        reason_expr = pl.lit("No enabled rule triggered")

    final_score = score_expr.clip(0, 100).round(2)
    result = lazy.select(
        pl.all(),
        final_score.alias("risk_score"),
        _level_expr(final_score, levels).alias("risk_level"),
        reason_expr.alias("risk_reasons"),
    ).collect()

    # Match the original pandas ``sort_values(..., ascending=False)`` exactly.
    # Its default quicksort is not stable, so an ordinary stable sort by score
    # would order equal scores differently.  argsort on the negated scores is
    # the same operation pandas performs, and numpy keeps this off the UI path.
    order = np.argsort(-result["risk_score"].to_numpy(), kind="quicksort")
    result = result.gather(order)

    if doc and doc in columns:
        result = result.unique(subset=[doc], keep="first", maintain_order=True)
    return result


class RiskEngine:
    """Assign AML risk scores from configurable rules."""

    def __init__(self, rules: Dict[str, Dict[str, Any]] | None = None,
                 levels: Dict[str, tuple] | None = None):
        self.rules = rules
        self.levels = levels or DEFAULT_RISK_LEVELS

    def score(self, df: Dataset, mapping: Dict[str, str]) -> pl.DataFrame:
        return _score(df, mapping, self.rules or {}, dict(self.levels))
