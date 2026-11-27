"""Statistical, AML, customer and account analytics.

Every aggregation runs as a DuckDB query over the dataset's Parquet file;
only the small result tables are materialized as Polars DataFrames.
"""
from __future__ import annotations

from typing import Dict, Optional

import polars as pl
import streamlit as st

from storage import Dataset, date_expr, lit, num_expr, q, query, sql_source
from utils import get_col

LEGAL_ENTITY_PREFIXES = (
    "4310",
    "43123",
    "43124",
    "43125"
    "4313",
    "432",
)

_CACHE = dict(show_spinner=False, max_entries=32)


def _source(ds: Dataset, mapping: Dict[str, str], fill_missing: bool = True) -> str:
    amount = get_col(mapping, "amount")
    date = get_col(mapping, "transaction_date")

    cols = []

    for col in ds.columns:
        if col == amount:
            cols.append(f'{num_expr(col)} AS "{col}"')
        elif col == date:
            cols.append(f'{date_expr(col)} AS "{col}"')
        else:
            cols.append(f'"{col}"')

    if amount:
        cols.append(f'{num_expr(amount)} AS "_amount"')
    elif fill_missing:
        cols.append('NULL::DOUBLE AS "_amount"')

    if date:
        cols.append(f'{date_expr(date)} AS "_date"')
    elif fill_missing:
        cols.append('NULL::TIMESTAMP AS "_date"')

    return f"SELECT {', '.join(cols)} FROM {sql_source(ds)}"
    
def _group_keys(mapping: Dict[str, str]) -> list[str]:
    keys = []
    customer = get_col(mapping, "customer_id")
    doc = get_col(mapping, "document_number")
    if customer:
        keys.append(q(customer))
    if doc:
        keys.append(q(doc))
    return keys


@st.cache_data(**_CACHE)
def _prepared(ds: Dataset, mapping: Dict[str, str]) -> pl.DataFrame:
    return query(_source(ds, mapping, fill_missing=False))


@st.cache_data(**_CACHE)
def _dashboard(ds: Dataset, mapping: Dict[str, str]) -> Dict[str, object]:
    src = _source(ds, mapping)
    customer = get_col(mapping, "document_number")
    debit = get_col(mapping, "debit_account")
    credit = get_col(mapping, "credit_account")

    accounts_sql = "NULL"
    if debit and credit:
        accounts_sql = (
            "(SELECT count(DISTINCT acct) FROM ("
            f"SELECT CAST({q(debit)} AS VARCHAR) AS acct FROM p "
            f"UNION ALL SELECT CAST({q(credit)} AS VARCHAR) FROM p))"
        )
    customer_sql = f"count(DISTINCT {q(customer)})" if customer else "NULL"
    sql = f"""
        WITH p AS ({src})
        SELECT count(*) AS transactions,
               {customer_sql} AS unique_customers,
               {accounts_sql} AS accounts,
               sum(abs("_amount")) AS total_turnover,
               sum(CASE WHEN "_amount" > 0 THEN "_amount" END) AS total_incoming,
               abs(sum(CASE WHEN "_amount" < 0 THEN "_amount" END)) AS total_outgoing,
               avg("_amount") AS average,
               median("_amount") AS median,
               max("_amount") AS max,
               min("_amount") AS min,
               stddev_samp("_amount") AS std,
               var_samp("_amount") AS variance,
               strftime(min("_date"), '%Y-%m-%d %H:%M:%S') AS date_min,
               strftime(max("_date"), '%Y-%m-%d %H:%M:%S') AS date_max
        FROM p
    """
    row = query(sql).row(0, named=True)
    date_min, date_max = row["date_min"], row["date_max"]
    return {
        "transactions": int(row["transactions"] or 0),
        "unique_customers": int(row["unique_customers"]) if row["unique_customers"] is not None else None,
        "accounts": int(row["accounts"]) if row["accounts"] is not None else None,
        "total_turnover": float(row["total_turnover"]) if row["total_turnover"] is not None else None,
        "total_incoming": float(row["total_incoming"]) if row["total_incoming"] is not None else None,
        "total_outgoing": float(row["total_outgoing"]) if row["total_outgoing"] is not None else None,
        "average": float(row["average"]) if row["average"] is not None else None,
        "median": float(row["median"]) if row["median"] is not None else None,
        "max": float(row["max"]) if row["max"] is not None else None,
        "min": float(row["min"]) if row["min"] is not None else None,
        "std": float(row["std"]) if row["std"] is not None else None,
        "variance": float(row["variance"]) if row["variance"] is not None else None,
        "date_range": f"{date_min} – {date_max}" if date_min and date_max else None,
    }


@st.cache_data(**_CACHE)
def _amount_statistics(ds: Dataset, mapping: Dict[str, str]) -> pl.DataFrame:
    src = _source(ds, mapping)
    sql = f"""
        WITH s AS (SELECT CAST("_amount" AS DOUBLE) a FROM ({src}) WHERE "_amount" IS NOT NULL)
        SELECT count(*) AS count, avg(a) AS mean, stddev_samp(a) AS std,
               min(a) AS min, max(a) AS max,
               quantile_cont(a, 0.25) AS p25, quantile_cont(a, 0.50) AS p50,
               quantile_cont(a, 0.75) AS p75, skewness(a) AS skewness,
               kurtosis(a) AS kurtosis,
               quantile_cont(a, 0.01) AS q01, quantile_cont(a, 0.05) AS q05,
               quantile_cont(a, 0.10) AS q10, quantile_cont(a, 0.90) AS q90,
               quantile_cont(a, 0.95) AS q95, quantile_cont(a, 0.99) AS q99
        FROM s
    """
    row = query(sql).row(0, named=True)
    n = int(row["count"] or 0)
    if not n:
        return pl.DataFrame({"metric": ["skewness", "kurtosis"], "value": [float("nan"), float("nan")]})
    metrics = [
        ("count", float(n)), ("mean", row["mean"]), ("std", row["std"]),
        ("min", row["min"]), ("25%", row["p25"]), ("50%", row["p50"]),
        ("75%", row["p75"]), ("max", row["max"]),
        ("skewness", row["skewness"] if n > 2 else float("nan")),
        ("kurtosis", row["kurtosis"] if n > 3 else float("nan")),
    ]
    for label, key in [("p1", "q01"), ("p5", "q05"), ("p10", "q10"), ("p25", "p25"),
                       ("p50", "p50"), ("p75", "p75"), ("p90", "q90"),
                       ("p95", "q95"), ("p99", "q99")]:
        metrics.append((label, row[key]))
    return pl.DataFrame({"metric": [m for m, _ in metrics],
                         "value": [float(v) if v is not None else float("nan") for _, v in metrics]})


@st.cache_data(**_CACHE)
def _benford(ds: Dataset, mapping: Dict[str, str]) -> pl.DataFrame:
    src = _source(ds, mapping)
    if not get_col(mapping, "amount"):
        return pl.DataFrame()
    sql = f"""
        WITH p AS (SELECT abs("_amount") x FROM ({src}) WHERE "_amount" IS NOT NULL),
        d AS (SELECT CAST(FLOOR(x / POWER(10, FLOOR(LOG10(x)))) AS INTEGER) AS digit
              FROM p WHERE x > 0),
        c AS (SELECT digit, count(*) n FROM d GROUP BY digit),
        t AS (SELECT sum(n) total FROM c),
        digits AS (SELECT unnest([1,2,3,4,5,6,7,8,9]) AS digit)
        SELECT dg.digit AS digit,
               CAST(coalesce(c.n / t.total, 0) AS DOUBLE) AS observed_rate,
               log10(1 + 1.0 / dg.digit) AS expected_rate
        FROM digits dg LEFT JOIN c ON c.digit = dg.digit LEFT JOIN t ON true
        ORDER BY dg.digit
    """
    result = query(sql)
    if result.is_empty():
        return pl.DataFrame()
    return result.with_columns(
        (pl.col("observed_rate") - pl.col("expected_rate")).alias("difference")
    ).with_columns(pl.col("difference").abs().alias("absolute_difference"))


@st.cache_data(**_CACHE)
def _aml(ds: Dataset, mapping: Dict[str, str]) -> Dict[str, pl.DataFrame]:
    src = _source(ds, mapping)
    results: Dict[str, pl.DataFrame] = {}
    amount = bool(get_col(mapping, "amount"))
    debit = get_col(mapping, "debit_account")
    credit = get_col(mapping, "credit_account")
    doc = get_col(mapping, "document_number")
    customer = get_col(mapping, "customer_id")
    branch = get_col(mapping, "branch")
    currency = get_col(mapping, "currency")
    cash = get_col(mapping, "cash_flag")
    date = bool(get_col(mapping, "transaction_date"))

    q99 = None
    if amount:
        q99 = query(f'SELECT quantile_cont(CAST("_amount" AS DOUBLE), 0.99) v FROM ({src}) WHERE "_amount" IS NOT NULL').item()
        q99_sql = "NULL" if q99 is None else repr(float(q99))
        results["Large transactions"] = query(
            f'SELECT * FROM ({src}) WHERE "_amount" >= {q99_sql} ORDER BY "_amount" DESC LIMIT 1000'
        )
        results["Round amounts"] = query(
            f'SELECT * FROM ({src}) WHERE abs("_amount") >= 100000 AND fmod(abs("_amount"), 100000.0) = 0 LIMIT 1000'
        )

    if doc:
        results["High-frequency customers"] = query(f"""
            SELECT p.* FROM ({src}) p
            JOIN (SELECT {q(doc)} AS d, count(*) AS n FROM ({src}) WHERE {q(doc)} IS NOT NULL GROUP BY 1 HAVING count(*) > 1) x
              ON p.{q(doc)} = x.d
            ORDER BY x.n DESC, p.{q(doc)} ASC LIMIT 2000
        """)

    if customer and amount and date:
        keys = _group_keys(mapping)
        group = ", ".join(keys)
        where = " AND ".join([f"{k} IS NOT NULL" for k in keys] + ['"_date" IS NOT NULL', '"_amount" IS NOT NULL'])
        results["Repeated same amount (same day)"] = query(f"""
            SELECT {group}, CAST("_date" AS DATE) AS "_date", "_amount",
                   count(*) AS transaction_count
            FROM ({src}) WHERE {where}
            GROUP BY {group}, CAST("_date" AS DATE), "_amount"
            HAVING count(*) >= 3
            ORDER BY transaction_count DESC, {group}, "_amount" LIMIT 1000
        """)

    keys = _group_keys(mapping)
    if keys and amount:
        group = ", ".join(keys)
        where = " AND ".join(f"{k} IS NOT NULL" for k in keys)
        results["High-risk customers by turnover"] = query(f"""
            SELECT {group}, count("_amount") AS transaction_count,
                   coalesce(sum(abs("_amount")), 0) AS turnover, max("_amount") AS max_amount
            FROM ({src}) WHERE {where} GROUP BY {group}
            ORDER BY turnover DESC LIMIT 100
        """)

    if branch and amount:
        results["High-risk branches by turnover"] = query(f"""
            SELECT {q(branch)}, count(*) AS transaction_count, sum(abs("_amount")) AS turnover
            FROM ({src}) GROUP BY {q(branch)} ORDER BY turnover DESC LIMIT 100
        """)

    if currency and customer:
        group = ", ".join(keys)
        where = " AND ".join(f"{k} IS NOT NULL" for k in keys)
        results["Cross-currency activity"] = query(f"""
            SELECT {group}, count(DISTINCT {q(currency)}) AS currency_count
            FROM ({src}) WHERE {where} GROUP BY {group}
            ORDER BY currency_count DESC LIMIT 100
        """)

    if date:
        results["Weekend transactions"] = query(
            f'SELECT * FROM ({src}) WHERE dayofweek("_date") IN (0, 6) LIMIT 1000'
        )
        results["Night transactions"] = query(
            f'SELECT * FROM ({src}) WHERE hour("_date") BETWEEN 0 AND 5 LIMIT 10000'
        )

    if cash and amount and q99 is not None:
        results["Large cash transactions"] = query(
            f'SELECT * FROM ({src}) WHERE {q(cash)} IS NOT NULL AND "_amount" >= {q99_sql} LIMIT 1000'
        )

    if debit and credit:
        prefixes = " OR ".join(
            f"CAST(p.{q(debit)} AS VARCHAR) LIKE {lit(prefix + '%')}"
            for prefix in LEGAL_ENTITY_PREFIXES
        )
        results["Legal Entity → Individual Transfers"] = query(f"""
            SELECT * FROM ({src}) p
            WHERE ({prefixes}) AND CAST(p.{q(credit)} AS VARCHAR) LIKE '3022%'
            LIMIT 10000
        """)

    if customer and branch and amount:
        group = ", ".join(keys)
        where = " AND ".join(f"{k} IS NOT NULL" for k in keys + [q(branch)])
        results["Customers using many branches"] = query(f"""
            WITH g AS (
                SELECT {group}, {q(branch)} AS b, sum("_amount") AS amt
                FROM ({src}) WHERE {where}
                GROUP BY {group}, {q(branch)}
            )
            SELECT {group}, count(DISTINCT b) AS branch_count,
                   string_agg(printf('%s (%,.2f)', CAST(b AS VARCHAR), amt), ', ' ORDER BY b) AS branches
            FROM g GROUP BY {group} ORDER BY branch_count DESC LIMIT 1000
        """)

    return results


@st.cache_data(**_CACHE)
def _customer_analytics(ds: Dataset, mapping: Dict[str, str]) -> pl.DataFrame:
    src = _source(ds, mapping)
    customer = get_col(mapping, "customer_id")
    if not customer:
        return pl.DataFrame()
    debit = get_col(mapping, "debit_account")
    credit = get_col(mapping, "credit_account")
    uniq_sql = ""
    join_sql = ""
    if debit and credit:
        uniq_sql = f""", u AS (
            SELECT {q(customer)} AS c, CAST({q(debit)} AS VARCHAR) AS acct FROM ({src})
            UNION ALL
            SELECT {q(customer)}, CAST({q(credit)} AS VARCHAR) FROM ({src})
        ), uniq AS (SELECT c, count(DISTINCT acct) AS n FROM u WHERE c IS NOT NULL GROUP BY c)"""
        join_sql = "LEFT JOIN uniq ON agg.c = uniq.c"
        extra = ", coalesce(uniq.n, 0) AS unique_accounts, coalesce(uniq.n, 0) AS unique_counterparties"
    else:
        extra = ""
    return query(f"""
        WITH agg AS (
            SELECT {q(customer)} AS c, count("_amount") AS transaction_count,
                   coalesce(sum(abs("_amount")), 0) AS turnover, avg("_amount") AS avg_amount,
                   median("_amount") AS median_amount, max("_amount") AS largest_transaction,
                   coalesce(sum("_amount"), 0) AS net_flow
            FROM ({src}) WHERE {q(customer)} IS NOT NULL GROUP BY {q(customer)}
        ) {uniq_sql}
        SELECT agg.c AS {q(customer)}, agg.transaction_count, agg.turnover, agg.avg_amount,
               agg.median_amount, agg.largest_transaction, agg.net_flow {extra}
        FROM agg {join_sql} ORDER BY agg.turnover DESC
    """)


@st.cache_data(**_CACHE)
def _account_analytics(ds: Dataset, mapping: Dict[str, str]) -> pl.DataFrame:
    src = _source(ds, mapping)
    debit = get_col(mapping, "debit_account")
    credit = get_col(mapping, "credit_account")
    if not debit or not credit:
        return pl.DataFrame()
    return query(f"""
        WITH out AS (
            SELECT CAST({q(debit)} AS VARCHAR) AS account, count(*) AS outgoing_count, sum("_amount") AS outgoing
            FROM ({src}) WHERE {q(debit)} IS NOT NULL GROUP BY 1
        ), inc AS (
            SELECT CAST({q(credit)} AS VARCHAR) AS account, count(*) AS incoming_count, sum("_amount") AS incoming
            FROM ({src}) WHERE {q(credit)} IS NOT NULL GROUP BY 1
        )
        SELECT coalesce(out.account, inc.account) AS account,
               coalesce(outgoing_count, 0) AS outgoing_count, coalesce(outgoing, 0) AS outgoing,
               coalesce(incoming_count, 0) AS incoming_count, coalesce(incoming, 0) AS incoming,
               abs(coalesce(outgoing, 0)) + abs(coalesce(incoming, 0)) AS turnover,
               coalesce(incoming, 0) - abs(coalesce(outgoing, 0)) AS net_flow
        FROM out FULL OUTER JOIN inc ON out.account = inc.account
        ORDER BY turnover DESC
    """)


@st.cache_data(**_CACHE)
def _time_series(ds: Dataset, mapping: Dict[str, str], freq: str) -> pl.DataFrame:
    src = _source(ds, mapping)
    bucket = {"D": "day", "W": "week", "M": "month", "Y": "year", "H": "hour"}.get(freq, "day")
    return query(f"""
        SELECT date_trunc('{bucket}', "_date") AS "_date", count(*) AS count,
               sum("_amount") AS sum, avg("_amount") AS mean,
               median("_amount") AS median, max("_amount") AS max
        FROM ({src}) WHERE "_date" IS NOT NULL AND "_amount" IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)


class AnalyticsEngine:
    """Reusable banking analytics calculations."""

    def __init__(self, df: Dataset, mapping: Dict[str, str]):
        self.df = df
        self.mapping = mapping
        self.amount_col = get_col(mapping, "amount")
        self.date_col = get_col(mapping, "transaction_date")

    def prepared(self) -> pl.DataFrame:
        return _prepared(self.df, self.mapping)

    def benfords_law(self) -> pl.DataFrame:
        return _benford(self.df, self.mapping)

    def dashboard_metrics(self) -> Dict[str, object]:
        return _dashboard(self.df, self.mapping)

    def amount_statistics(self) -> pl.DataFrame:
        return _amount_statistics(self.df, self.mapping)

    def time_series(self, freq: str = "D") -> pl.DataFrame:
        return _time_series(self.df, self.mapping, freq)

    def aml_indicators(self) -> Dict[str, pl.DataFrame]:
        return _aml(self.df, self.mapping)

    def customer_analytics(self) -> pl.DataFrame:
        return _customer_analytics(self.df, self.mapping)

    def account_analytics(self) -> pl.DataFrame:
        return _account_analytics(self.df, self.mapping)
