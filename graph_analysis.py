"""Transaction graph construction, cycle detection and network analytics.

Cycle detection notes
---------------------
A customer transfer in this bank is recorded as TWO transactions:

    leg 1:  sender real account  ->  sender transit account (starts with "43")
    leg 2:  sender transit account  ->  real recipient account

The transit account is opened in the sender's name, so a logical transfer is
the *pair* of legs sharing the same transit account (and the same amount and
day).  ``effective_flows`` / ``build_effective_graph`` collapse the two legs of
every transit account into a single direct flow (sender -> receiver), so cycles
(A -> B -> A, A -> B -> C -> A, ...) are detected between real accounts.

Edges are aggregated with Polars before they reach NetworkX - no per-row
Python loop over the raw transaction table.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import polars as pl

from storage import Dataset, date_col_expr
from utils import get_col

_MAX_PAIR_WORK = 200_000


def _date_expr(column: str, dtype: pl.DataType | None = None) -> pl.Expr:
    return date_col_expr(column, dtype)


def _amount_expr(column: str) -> pl.Expr:
    return (
        pl.col(column).cast(pl.String)
        .str.replace_all(",", "").str.replace_all(" ", "")
        .cast(pl.Float64, strict=False)
        .abs().fill_null(1.0)
    )


def _account_expr(column: str) -> pl.Expr:
    cleaned = pl.col(column).cast(pl.String).fill_null("nan").str.strip_chars()
    return pl.when(cleaned.str.to_lowercase() == "nan").then(pl.lit("")).otherwise(cleaned)


class TransactionGraphAnalyzer:
    """Build and analyze transaction networks using NetworkX."""

    def __init__(self, df: Dataset, mapping: Dict[str, str]):
        self.mapping = mapping
        self.ds = df
        cash = get_col(mapping, "cash_flag")
        self._lazy = df.lazy().filter(pl.col(cash).is_null()) if cash and cash in df.columns else df.lazy()

    def build_graph(
        self,
        source_key: str = "debit_account",
        target_key: str = "credit_account",
        directed: bool = True,
        multigraph: bool = False,
        weighted: bool = True,
    ):
        src = get_col(self.mapping, source_key)
        dst = get_col(self.mapping, target_key)
        amount = get_col(self.mapping, "amount")

        if not src or not dst or src not in self.ds.columns or dst not in self.ds.columns:
            raise ValueError(f"Source ('{src}') and target ('{dst}') columns must exist in dataset.")

        weight_expr = _amount_expr(amount).alias("w") if weighted and amount and amount in self.ds.columns \
            else pl.lit(1.0).alias("w")
        edges = (
            self._lazy.select(
                _account_expr(src).alias("u"),
                _account_expr(dst).alias("v"),
                weight_expr,
            )
            .filter(
                pl.col("u").is_not_null() & pl.col("v").is_not_null()
                & (pl.col("u") != "") & (pl.col("v") != "")
                & (pl.col("u") != pl.col("v"))
            )
            .group_by("u", "v")
            .agg(pl.col("w").sum().alias("weight"), pl.len().alias("count"))
            .collect()
        )

        if multigraph:
            G = nx.MultiDiGraph() if directed else nx.MultiGraph()
        else:
            G = nx.DiGraph() if directed else nx.Graph()

        for u, v, weight, count in edges.iter_rows():
            if multigraph or not G.has_edge(u, v):
                G.add_edge(u, v, weight=float(weight), count=int(count))
            else:
                G[u][v]["weight"] += float(weight)
                G[u][v]["count"] += int(count)
        return G

    @staticmethod
    def _clean_account(value: Any) -> str:
        if value is None:
            return ""
        s = str(value).strip()
        return "" if s.lower() == "nan" else s

    def effective_flows(self, transit_prefix: str = "43") -> pl.DataFrame:
        """Collapse transit-account legs into direct sender -> receiver flows."""
        src_c = get_col(self.mapping, "debit_account")
        dst_c = get_col(self.mapping, "credit_account")
        if not src_c or not dst_c or src_c not in self.ds.columns or dst_c not in self.ds.columns:
            raise ValueError("Debit account and Credit account columns must be mapped to collapse transit flows.")

        amount_c = get_col(self.mapping, "amount")
        date_c = get_col(self.mapping, "transaction_date")
        cur_c = get_col(self.mapping, "currency")

        exprs = [_account_expr(src_c).alias("src"), _account_expr(dst_c).alias("dst")]
        exprs.append(_amount_expr(amount_c).alias("amount") if amount_c and amount_c in self.ds.columns
                     else pl.lit(1.0).alias("amount"))
        exprs.append(
            _date_expr(date_c, self._lazy.collect_schema().get(date_c)).dt.strftime("%Y-%m-%d").fill_null("").alias("date")
            if date_c and date_c in self.ds.columns else pl.lit("").alias("date")
        )
        if cur_c and cur_c in self.ds.columns:
            cur = pl.col(cur_c).cast(pl.String).fill_null("nan").str.strip_chars()
            exprs.append(
                pl.when(cur.str.to_lowercase().is_in(["nan", "none", ""])).then(pl.lit("ALL"))
                .otherwise(cur).alias("currency")
            )
        else:
            exprs.append(pl.lit("ALL").alias("currency"))

        d = self._lazy.select(exprs).collect()

        prefix = (transit_prefix or "").strip()
        d = d.with_columns(pl.lit("direct").alias("kind"))
        if prefix:
            src_tr, dst_tr = pl.col("src").str.starts_with(prefix), pl.col("dst").str.starts_with(prefix)
            d = d.with_columns(
                pl.when(dst_tr & ~src_tr).then(pl.lit("fund"))
                .when(src_tr & ~dst_tr).then(pl.lit("out"))
                .when(src_tr & dst_tr).then(pl.lit("transit2transit"))
                .otherwise(pl.lit("direct")).alias("kind")
            )

        edges: List[Dict[str, Any]] = []

        direct = d.filter(pl.col("kind") == "direct")
        for (u, v), grp in direct.group_by("src", "dst", maintain_order=True):
            if not u or not v or u == v:
                continue
            edges.append({
                "source": u, "target": v,
                "amount": float(grp["amount"].sum()), "count": int(grp.height),
                "pairing": "direct",
            })

        if prefix:
            legs = d.filter(pl.col("kind").is_in(["fund", "out"]))
            fund_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            out_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for i, row in enumerate(legs.iter_rows(named=True)):
                leg = {"date": row["date"], "amount": float(row["amount"]), "seq": i}
                if row["kind"] == "fund":
                    leg["src"] = row["src"]
                    fund_groups.setdefault((row["dst"], row["currency"]), []).append(leg)
                else:
                    leg["dst"] = row["dst"]
                    out_groups.setdefault((row["src"], row["currency"]), []).append(leg)

            for key in sorted(set(fund_groups) | set(out_groups)):
                transit = key[0]
                funds = sorted(fund_groups.get(key, []), key=lambda x: (x["date"], x["seq"]))
                outs = sorted(out_groups.get(key, []), key=lambda x: (x["date"], x["seq"]))

                if not funds or not outs or len(funds) * len(outs) > _MAX_PAIR_WORK:
                    for f in funds:
                        edges.append({"source": f["src"], "target": transit, "amount": f["amount"],
                                      "count": 1, "pairing": "unmatched-fund"})
                    for o in outs:
                        edges.append({"source": transit, "target": o["dst"], "amount": o["amount"],
                                      "count": 1, "pairing": "unmatched-out"})
                    continue

                used = [False] * len(funds)
                for o in outs:
                    match, flag = None, "paired"
                    for i in range(len(funds) - 1, -1, -1):
                        if not used[i] and funds[i]["amount"] == o["amount"] and funds[i]["date"] <= o["date"]:
                            match = i
                            break
                    if match is None:
                        for i in range(len(funds) - 1, -1, -1):
                            if not used[i] and funds[i]["amount"] == o["amount"]:
                                match, flag = i, "paired(out-of-order)"
                                break
                    if match is None:
                        edges.append({"source": transit, "target": o["dst"], "amount": o["amount"],
                                      "count": 1, "pairing": "unmatched-out"})
                        continue
                    used[match] = True
                    edges.append({"source": funds[match]["src"], "target": o["dst"],
                                  "amount": o["amount"], "count": 1, "pairing": flag})
                for i, f in enumerate(funds):
                    if not used[i]:
                        edges.append({"source": f["src"], "target": transit, "amount": f["amount"],
                                      "count": 1, "pairing": "unmatched-fund"})

        eff = pl.DataFrame(edges, schema=["source", "target", "amount", "count", "pairing"])
        if eff.is_empty():
            return eff
        return (
            eff.group_by("source", "target", maintain_order=True)
            .agg(
                pl.col("amount").sum().alias("amount"),
                pl.col("count").sum().alias("count"),
                pl.col("pairing").map_elements(
                    lambda s: ",".join(sorted(set(s))), return_dtype=pl.String
                ).alias("pairing"),
            )
        )

    @staticmethod
    def graph_from_flows(flows: pl.DataFrame):
        """Build a weighted DiGraph from an ``effective_flows`` dataframe."""
        G = nx.DiGraph()
        if flows is None or flows.is_empty():
            return G
        for u, v, amount, count in flows.select("source", "target", "amount", "count").iter_rows():
            u, v = str(u), str(v)
            if not u or not v or u == v or u.lower() == "nan" or v.lower() == "nan":
                continue
            if G.has_edge(u, v):
                G[u][v]["weight"] += float(amount)
                G[u][v]["count"] += int(count)
            else:
                G.add_edge(u, v, weight=float(amount), count=int(count))
        return G

    def build_effective_graph(self, transit_prefix: str = "43"):
        return self.graph_from_flows(self.effective_flows(transit_prefix))

    @staticmethod
    def metrics(G) -> Dict[str, Any]:
        UG = G.to_undirected() if hasattr(G, "to_undirected") else G
        metrics = {
            "nodes_count": G.number_of_nodes(),
            "edges_count": G.number_of_edges(),
            "graph_density": round(nx.density(G), 5) if G.number_of_nodes() > 1 else 0.0,
            "connected_components": nx.number_connected_components(UG) if G.number_of_nodes() else 0,
        }
        if G.is_directed() and G.number_of_nodes() > 0:
            metrics["strongly_connected_components"] = nx.number_strongly_connected_components(G)
            metrics["weakly_connected_components"] = nx.number_weakly_connected_components(G)
        return metrics

    @staticmethod
    def cycles_table(G, limit: int = 100, max_length: int = 6) -> pl.DataFrame:
        if G.number_of_nodes() == 0:
            return pl.DataFrame()

        if not G.is_directed():
            raw_cycles = nx.cycle_basis(G)
            cycles = [c for c in raw_cycles if len(c) >= 3][:limit]
        else:
            try:
                gen = nx.simple_cycles(G, length_bound=max_length)
            except TypeError:
                gen = nx.simple_cycles(G)
            cycles = []
            for c in gen:
                if 2 <= len(c) <= max_length:
                    cycles.append(c)
                if len(cycles) >= limit * 10:
                    break

        rows = []
        for c in cycles:
            bottleneck = 0.0
            if G.is_directed():
                weights = [
                    float(G[u][v].get("weight", 1.0))
                    for i, u in enumerate(c)
                    for v in [c[(i + 1) % len(c)]]
                    if G.has_edge(u, v)
                ]
                bottleneck = min(weights) if weights else 0.0
            if len(c) == 2:
                rating = "High (Round-Trip)"
            elif len(c) <= 4:
                rating = "High (Circular Flow)"
            elif len(c) <= max_length:
                rating = "Medium"
            else:
                rating = "Low"
            rows.append({
                "cycle_path": " ➔ ".join(map(str, c)) + f" ➔ {c[0]}",
                "cycle_length": len(c),
                "bottleneck_amount": round(bottleneck, 2),
                "risk_rating": rating,
            })
        out = pl.DataFrame(rows)
        if out.is_empty():
            return out
        return out.sort(["bottleneck_amount", "cycle_length"], descending=[True, False]).head(limit)

    @staticmethod
    def scc_table(G, min_size: int = 2, max_members: int = 10, limit: int = 100) -> pl.DataFrame:
        if G.number_of_nodes() == 0 or not G.is_directed():
            return pl.DataFrame()
        comps = []
        for comp in nx.strongly_connected_components(G):
            if len(comp) >= min_size:
                sub = G.subgraph(comp)
                internal = sum(float(d.get("weight", 0.0)) for _, _, d in sub.edges(data=True))
                comps.append((comp, internal, sub.number_of_edges()))
        comps.sort(key=lambda x: -x[1])
        rows = []
        for comp, internal, n_edges in comps[:limit]:
            members = sorted(comp)
            shown = ", ".join(map(str, members[:max_members]))
            if len(members) > max_members:
                shown += f" (+{len(members) - max_members} more)"
            rows.append({
                "members": shown,
                "members_count": len(members),
                "internal_edges": n_edges,
                "internal_amount": round(internal, 2),
            })
        return pl.DataFrame(rows)
