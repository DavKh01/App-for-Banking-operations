"""Transaction graph construction, cycle detection and network analytics."""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import networkx as nx

from utils import get_col


class TransactionGraphAnalyzer:
    """Build and analyze transaction networks using NetworkX."""

    def __init__(self, df: pd.DataFrame, mapping: Dict[str, str]):
        self.df = df
        self.mapping = mapping

    def build_graph(
        self,
        source_key: str = "debit_account",
        target_key: str = "credit_account",
        directed: bool = True,
        multigraph: bool = False,
        weighted: bool = True
    ):
        src = get_col(self.mapping, source_key)
        dst = get_col(self.mapping, target_key)
        amount = get_col(self.mapping, "amount")

        if not src or not dst or src not in self.df.columns or dst not in self.df.columns:
            raise ValueError(f"Source ('{src}') and target ('{dst}') columns must exist in dataset.")

        if multigraph:
            G = nx.MultiDiGraph() if directed else nx.MultiGraph()
        else:
            G = nx.DiGraph() if directed else nx.Graph()

        cols = [src, dst] + ([amount] if amount and amount in self.df.columns else [])
        valid_df = self.df[cols].dropna(subset=[src, dst])

        for _, row in valid_df.iterrows():
            u = str(row[src]).strip()
            v = str(row[dst]).strip()
            if not u or not v or u == v or u.lower() == "nan" or v.lower() == "nan":
                continue

            w = 1.0
            if weighted and amount and amount in row:
                try:
                    val = float(pd.to_numeric(str(row[amount]).replace(",", ""), errors="coerce"))
                    w = abs(val) if not pd.isna(val) else 1.0
                except Exception:
                    w = 1.0

            if G.has_edge(u, v) and not multigraph:
                G[u][v]["weight"] = G[u][v].get("weight", 0) + w
                G[u][v]["count"] = G[u][v].get("count", 0) + 1
            else:
                G.add_edge(u, v, weight=w, count=1)

        return G

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
    def cycles_table(G, limit: int = 100) -> pd.DataFrame:
        """Find circular transactions (A -> B -> C -> A), often used to obscure fund trails."""
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        if not G.is_directed():
            raw_cycles = nx.cycle_basis(G)
            cycles = [c for c in raw_cycles if len(c) >= 3][:limit]
        else:
            try:
                # Limit simple cycles to avoid exponential blowup
                cycles = []
                for c in nx.simple_cycles(G):
                    if len(c) >= 2:
                        cycles.append(c)
                    if len(cycles) >= limit:
                        break
            except Exception:
                cycles = []

        rows = []
        for c in cycles:
            path_str = " ➔ ".join(map(str, c)) + f" ➔ {c[0]}"
            rows.append({
                "cycle_path": path_str,
                "cycle_length": len(c),
                "risk_rating": "High (Circular Flow)" if len(c) <= 4 else "Medium",
            })
        return pd.DataFrame(rows)
