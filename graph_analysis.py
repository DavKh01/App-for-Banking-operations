
"""Transaction graph construction and network analytics."""
from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import networkx as nx

from utils import get_col, safe_numeric


class TransactionGraphAnalyzer:
    """Build and analyze transaction networks using NetworkX."""

    def __init__(self, df: pd.DataFrame, mapping: Dict[str, str]):
        self.df = df
        self.mapping = mapping

    def build_graph(self, source_key: str = "debit_account", target_key: str = "credit_account", directed: bool = True, multigraph: bool = False, weighted: bool = True):
        src, dst = get_col(self.mapping, source_key), get_col(self.mapping, target_key)
        amount = get_col(self.mapping, "amount")
        if not src or not dst:
            raise ValueError("Source and target columns must be mapped.")
        if multigraph:
            G = nx.MultiDiGraph() if directed else nx.MultiGraph()
        else:
            G = nx.DiGraph() if directed else nx.Graph()
        for _, row in self.df[[src, dst] + ([amount] if amount else [])].dropna(subset=[src, dst]).iterrows():
            w = float(pd.to_numeric(row[amount], errors="coerce")) if weighted and amount else 1.0
            if pd.isna(w):
                w = 1.0
            if G.has_edge(row[src], row[dst]) and not multigraph:
                G[row[src]][row[dst]]["weight"] += abs(w)
                G[row[src]][row[dst]]["count"] += 1
            else:
                G.add_edge(row[src], row[dst], weight=abs(w), count=1)
        return G

    @staticmethod
    def metrics(G) -> Dict[str, Any]:
        UG = G.to_undirected() if hasattr(G, "to_undirected") else G
        metrics = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "connected_components": nx.number_connected_components(UG) if G.number_of_nodes() else 0,
        }
        if G.is_directed():
            metrics["strongly_connected_components"] = nx.number_strongly_connected_components(G)
            metrics["weakly_connected_components"] = nx.number_weakly_connected_components(G)
        return metrics

    @staticmethod
    def centrality_table(G, top_n: int = 100) -> pd.DataFrame:
        if G.number_of_nodes() == 0:
            return pd.DataFrame()
        H = nx.Graph(G) if G.is_multigraph() else G
        pr = nx.pagerank(H, weight="weight") if H.number_of_nodes() else {}
        deg = nx.degree_centrality(H)
        between = nx.betweenness_centrality(H, k=min(200, H.number_of_nodes()), weight="weight") if H.number_of_nodes() > 2 else {}
        close = nx.closeness_centrality(H)
        rows = []
        for n in H.nodes():
            rows.append({"node": n, "degree_centrality": deg.get(n), "betweenness": between.get(n), "closeness": close.get(n), "pagerank": pr.get(n)})
        return pd.DataFrame(rows).sort_values("pagerank", ascending=False).head(top_n)

    @staticmethod
    def cycles_table(G, limit: int = 100) -> pd.DataFrame:
        if not G.is_directed():
            cycles = nx.cycle_basis(G)
        else:
            cycles = list(nx.simple_cycles(G))[:limit]
        return pd.DataFrame({"cycle": [" → ".join(map(str, c)) for c in cycles[:limit]], "cycle_length": [len(c) for c in cycles[:limit]]})
