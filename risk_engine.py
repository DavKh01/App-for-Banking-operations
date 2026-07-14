
"""Configurable AML risk scoring engine."""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from config import DEFAULT_RULES, DEFAULT_RISK_LEVELS
from utils import get_col, safe_numeric, safe_datetime


class RiskEngine:
    """Assign AML risk scores from configurable rules."""

    def __init__(self, rules: Dict[str, Dict[str, Any]] | None = None, levels: Dict[str, tuple] | None = None):
        self.rules = rules or DEFAULT_RULES.copy()
        self.levels = levels or DEFAULT_RISK_LEVELS

    def _level(self, score: float) -> str:
        for level, (lo, hi) in self.levels.items():
            if lo <= score <= hi:
                return level
        return "Critical" if score > 100 else "Low"

    def score(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        out = df.copy()
        score = pd.Series(0.0, index=out.index)
        reasons: List[List[str]] = [[] for _ in range(len(out))]

        amount_col = get_col(mapping, "amount")
        date_col = get_col(mapping, "transaction_date")
        debit = get_col(mapping, "debit_account")
        credit = get_col(mapping, "credit_account")
        doc = get_col(mapping, "document_number")
        cash = get_col(mapping, "cash_flag")
        blacklist = get_col(mapping, "blacklist_flag")
        currency = get_col(mapping, "currency")
        customer = get_col(mapping, "customer_id") or debit
        debit_balance = get_col(mapping, "debit_balance")

        amt = safe_numeric(out[amount_col]).abs() if amount_col else pd.Series(0, index=out.index)
        dates = safe_datetime(out[date_col]) if date_col else pd.Series(pd.NaT, index=out.index)

        def apply_rule(name: str, mask: pd.Series):
            rule = self.rules.get(name, {})
            if not rule.get("enabled", False):
                return
            weight = float(rule.get("weight", 0))
            desc = rule.get("description", name)
            m = mask.fillna(False).astype(bool)
            score.loc[m] += weight
            for pos in np.where(m.values)[0]:
                reasons[pos].append(f"{name}: {desc} (+{weight:g})")

        if amount_col:
            apply_rule("large_amount", amt >= float(self.rules["large_amount"].get("threshold", 0)))
            round_threshold = float(self.rules["round_amount"].get("threshold", 1000))
            apply_rule("round_amount", (amt >= round_threshold) & (amt % round_threshold == 0))
        if cash:
            apply_rule("cash_transaction", out[cash].notna())
        if blacklist:
            apply_rule("blacklist_flag", out[blacklist].astype(str).str.lower().str.contains("1|true|yes|black", regex=True, na=False))
        if date_col:
            apply_rule("night_activity", dates.dt.hour.between(0, 5))
            apply_rule("weekend_activity", dates.dt.weekday >= 5)
        if doc:
            apply_rule("duplicate_document", out[doc].notna() & out.duplicated(doc, keep=False))
        if debit:
            cnt = out.groupby(debit)[debit].transform("count")
            apply_rule("high_frequency_sender", cnt >= float(self.rules["high_frequency_sender"].get("threshold", 20)))
        if debit and credit:
            cpty_count = out.groupby(debit)[credit].transform("nunique")
            apply_rule("many_counterparties", cpty_count >= float(self.rules["many_counterparties"].get("threshold", 15)))
        if currency and customer:
            ccnt = out.groupby(customer)[currency].transform("nunique")
            apply_rule("multiple_currencies", ccnt >= float(self.rules["multiple_currencies"].get("threshold", 2)))
        if debit_balance and amount_col:
            bal = safe_numeric(out[debit_balance]).abs()
            threshold = float(self.rules["balance_anomaly"].get("threshold", .9))
            apply_rule("balance_anomaly", (bal > 0) & ((amt / bal) >= threshold))

        out["risk_score"] = score.clip(0, 100).round(2)
        out["risk_level"] = out["risk_score"].apply(self._level)
        out["risk_reasons"] = ["; ".join(r) if r else "No enabled rule triggered" for r in reasons]
        return out.sort_values("risk_score", ascending=False)
