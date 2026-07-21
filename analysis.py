
"""Statistical, AML, customer and account analytics."""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from utils import safe_numeric, safe_datetime, get_col


class AnalyticsEngine:
    """Reusable banking analytics calculations."""

    def __init__(self, df: pd.DataFrame, mapping: Dict[str, str]):
        self.df = df.copy()
        self.mapping = mapping
        self.amount_col = get_col(mapping, "amount")
        self.date_col = get_col(mapping, "transaction_date")
    def prepared(self) -> pd.DataFrame:
        df = self.df.copy()
        if self.amount_col:
            df["_amount"] = safe_numeric(df[self.amount_col])
        if self.date_col:
            df["_date"] = safe_datetime(df[self.date_col])
        return df

    def benfords_law(self) -> pd.DataFrame:

        df = self.prepared()

        if "_amount" not in df.columns:
            return pd.DataFrame()

        amounts = (
            pd.to_numeric(df["_amount"], errors="coerce")
            .abs()
            .dropna()
        )

        amounts = amounts[amounts > 0]

        if amounts.empty:
            return pd.DataFrame()


        first_digits = (amounts / (10 ** np.floor(np.log10(amounts)))).astype(int)

        observed = (
            first_digits.value_counts(normalize=True)
            .sort_index()
            .reindex(range(1, 10), fill_value=0)
        )

        expected = pd.Series(
            {d: np.log10(1 + 1 / d) for d in range(1, 10)}
        )

        result = pd.DataFrame({
            "digit": range(1, 10),
            "observed_rate": observed.values,
            "expected_rate": expected.values,
        })

        result["difference"] = result["observed_rate"] - result["expected_rate"]
        result["absolute_difference"] = result["difference"].abs()

        return result

    def dashboard_metrics(self) -> Dict[str, object]:
        df = self.prepared()
        amt = df.get("_amount", pd.Series(dtype=float)).dropna()
        debit = get_col(self.mapping, "debit_account")
        credit = get_col(self.mapping, "credit_account")
        customer = get_col(self.mapping, "document_number")
        date = df.get("_date", pd.Series(dtype="datetime64[ns]"))
        accounts = pd.concat([df[debit], df[credit]], ignore_index=True).nunique() if debit and credit else None
        return {
            "transactions": len(df),
            "unique_customers": df[customer].nunique() if customer else None,
            "accounts": accounts,
            "total_turnover": float(amt.abs().sum()) if not amt.empty else None,
            "total_incoming": float(amt[amt > 0].sum()) if not amt.empty else None,
            "total_outgoing": float(abs(amt[amt < 0].sum())) if not amt.empty else None,
            "average": float(amt.mean()) if not amt.empty else None,
            "median": float(amt.median()) if not amt.empty else None,
            "max": float(amt.max()) if not amt.empty else None,
            "min": float(amt.min()) if not amt.empty else None,
            "std": float(amt.std()) if len(amt) > 1 else None,
            "variance": float(amt.var()) if len(amt) > 1 else None,
            "date_range": f"{date.min()} – {date.max()}" if not date.dropna().empty else None,
        }

    def amount_statistics(self) -> pd.DataFrame:
        df = self.prepared()
        amt = df["_amount"].dropna() if "_amount" in df else pd.Series(dtype=float)
        percentiles = amt.quantile([.01, .05, .10, .25, .50, .75, .90, .95, .99]) if not amt.empty else pd.Series(dtype=float)
        out = amt.describe().to_frame("value") if not amt.empty else pd.DataFrame()
        out.loc["skewness", "value"] = amt.skew() if len(amt) > 2 else np.nan
        out.loc["kurtosis", "value"] = amt.kurtosis() if len(amt) > 3 else np.nan
        for idx, val in percentiles.items():
            out.loc[f"p{int(idx*100)}", "value"] = val
        return out.reset_index(names="metric")

    def time_series(self, freq: str = "D") -> pd.DataFrame:
        df = self.prepared().dropna(subset=["_date", "_amount"])
        if df.empty:
            return pd.DataFrame()
        return df.set_index("_date")["_amount"].resample(freq).agg(["count", "sum", "mean", "median", "max"]).reset_index()

    def aml_indicators(self) -> Dict[str, pd.DataFrame]:
        df = self.prepared()
        results: Dict[str, pd.DataFrame] = {}
        amount = "_amount" if "_amount" in df else None
        debit = get_col(self.mapping, "debit_account")
        debit_name = get_col(self.mapping, "debit_name")
        credit = get_col(self.mapping, "credit_account")
        credit_name = get_col(self.mapping, "credit_name")
        customer = get_col(self.mapping, "customer_id")
        doc = get_col(self.mapping, "document_number")
        branch = get_col(self.mapping, "branch")
        currency = get_col(self.mapping, "currency")
        note = get_col(self.mapping, "note")
        cash = get_col(self.mapping, "cash_flag")
        if amount:
            q99 = df[amount].quantile(.99)
            results["Large transactions"] = df[df[amount] >= q99].sort_values(amount, ascending=False).head(1000)
            results["Round amounts"] = df[(df[amount].abs() >= 100000) & (df[amount].abs() % 100000 == 0)].head(1000)
        if doc:
            dup_docs = df[df[doc].notna() & df.duplicated(doc, keep=False)].sort_values(doc)
            results["Repeated document numbers"] = dup_docs.head(1000)
        if customer and amount and "_date" in df:
            group_cols = [
                customer,
                *([doc] if doc else []),
                df["_date"].dt.date,
                amount,
            ]

            same_amount_day = (
                df.groupby(group_cols)
                .size()
                .reset_index(name="transaction_count")
            )
            same_amount_day = (
                same_amount_day[
                    same_amount_day["transaction_count"] >= 3
                    ]
                .sort_values("transaction_count", ascending=False)
            )
            results["Repeated same amount (same day)"] = same_amount_day.head(1000)
        if debit and credit:
            results["Top senders"] = df.groupby(debit_name).size().sort_values(ascending=False).head(50).reset_index(name="transaction_count")
            results["Top receivers"] = df.groupby(credit_name).size().sort_values(ascending=False).head(50).reset_index(name="transaction_count")
            cpty = df.groupby(debit)[credit].nunique().sort_values(ascending=False).head(50).reset_index(name="unique_counterparties")
            results["Frequent counterparty changes"] = cpty
            many_to_one = df.groupby(credit)[debit].nunique().sort_values(ascending=False).head(50).reset_index(name="unique_senders")
            results["Many-to-one transfers"] = many_to_one
            one_to_many = df.groupby(debit)[credit].nunique().sort_values(ascending=False).head(50).reset_index(name="unique_receivers")
            results["One-to-many transfers"] = one_to_many
        if customer and amount:
            results["High-risk customers by turnover"] = df.groupby(customer)[amount].agg(transaction_count="count", turnover=lambda s: s.abs().sum(), max_amount="max").sort_values("turnover", ascending=False).head(100).reset_index()
        if branch and amount:
            results["High-risk branches by turnover"] = df.groupby(branch)[amount].agg(transaction_count="count", turnover=lambda s: s.abs().sum()).sort_values("turnover", ascending=False).head(100).reset_index()
        if currency and customer:
            results["Cross-currency activity"] = df.groupby(customer)[currency].nunique().sort_values(ascending=False).head(100).reset_index(name="currency_count")
        if "_date" in df:
            results["Weekend transactions"] = df[df["_date"].dt.weekday >= 5].head(1000)
            results["Night transactions"] = df[df["_date"].dt.hour.between(0, 5)].head(10000)
        if note:
            results["Repeated notes"] = df[df[note].notna() & df.duplicated(note, keep=False)].head(1000)
        if cash and amount:
            cash_large = df[
                (df[cash].notna()) &
                (df[amount] >= df[amount].quantile(.99))
                ]
            results["Large cash transactions"] = cash_large.head(1000)
        if customer and branch:
            branch_use = (
                df.groupby(customer)[branch]
                .nunique()
                .reset_index(name="branch_count")
            )

            results["Customers using many branches"] = (
                branch_use.sort_values("branch_count", ascending=False)
                .head(1000)
            )
        return results

    def customer_analytics(self) -> pd.DataFrame:
        df = self.prepared()
        customer, amount = get_col(self.mapping, "customer_id"), "_amount"
        debit, credit = get_col(self.mapping, "debit_account"), get_col(self.mapping, "credit_account")
        if not customer or amount not in df:
            return pd.DataFrame()
        out = df.groupby(customer)[amount].agg(transaction_count="count", turnover=lambda s: s.abs().sum(), avg_amount="mean", median_amount="median", largest_transaction="max", net_flow="sum")
        if debit and credit:
            out["unique_accounts"] = df.groupby(customer).apply(lambda g: pd.concat([g[debit], g[credit]]).nunique())
            out["unique_counterparties"] = df.groupby(customer).apply(lambda g: pd.concat([g[debit], g[credit]]).nunique())
        return out.sort_values("turnover", ascending=False).reset_index()

    def account_analytics(self) -> pd.DataFrame:
        df = self.prepared()
        debit, credit, amount = get_col(self.mapping, "debit_account"), get_col(self.mapping, "credit_account"), "_amount"
        if not debit or not credit or amount not in df:
            return pd.DataFrame()
        outgoing = df.groupby(debit)[amount].agg(outgoing_count="count", outgoing="sum").rename_axis("account")
        incoming = df.groupby(credit)[amount].agg(incoming_count="count", incoming="sum").rename_axis("account")
        out = outgoing.join(incoming, how="outer").fillna(0)
        out["turnover"] = out["outgoing"].abs() + out["incoming"].abs()
        out["net_flow"] = out["incoming"] - out["outgoing"].abs()
        return out.sort_values("turnover", ascending=False).reset_index()
