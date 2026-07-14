
"""Optional machine-learning analytics for anomaly detection and clustering."""
from __future__ import annotations

from typing import List, Optional
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MLAnalyzer:
    """Small, unit-testable wrapper around common unsupervised models."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _matrix(self, columns: List[str]):
        X = self.df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        return StandardScaler().fit_transform(X)

    def isolation_forest(self, columns: List[str], contamination: float = 0.02) -> pd.DataFrame:
        X = self._matrix(columns)
        model = IsolationForest(contamination=contamination, random_state=42)
        out = self.df.copy()
        out["ml_anomaly"] = model.fit_predict(X)
        out["ml_anomaly_score"] = -model.score_samples(X)
        return out.sort_values("ml_anomaly_score", ascending=False)

    def kmeans(self, columns: List[str], clusters: int = 5) -> pd.DataFrame:
        X = self._matrix(columns)
        out = self.df.copy()
        out["cluster"] = KMeans(n_clusters=clusters, random_state=42, n_init="auto").fit_predict(X)
        return out

    def dbscan(self, columns: List[str], eps: float = 0.8, min_samples: int = 10) -> pd.DataFrame:
        X = self._matrix(columns)
        out = self.df.copy()
        out["cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        return out

    def pca_projection(self, columns: List[str], components: int = 2) -> pd.DataFrame:
        X = self._matrix(columns)
        arr = PCA(n_components=components, random_state=42).fit_transform(X)
        out = self.df.copy()
        for i in range(components):
            out[f"PC{i+1}"] = arr[:, i]
        return out
