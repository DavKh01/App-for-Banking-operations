
"""Plotly visualization helpers."""
from __future__ import annotations

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


def histogram(df: pd.DataFrame, x: str):
    return px.histogram(df, x=x, marginal="box", title=f"Distribution of {x}")


def boxplot(df: pd.DataFrame, y: str, x: Optional[str] = None):
    return px.box(df, x=x, y=y, points="outliers", title=f"Boxplot of {y}")


def line_chart(df: pd.DataFrame, x: str, y: str, title: str = "Time series"):
    return px.line(df, x=x, y=y, title=title)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str = "Bar chart"):
    return px.bar(df, x=x, y=y, title=title)


def scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None):
    return px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")


def heatmap_corr(df: pd.DataFrame):
    corr = df.select_dtypes("number").corr()
    return px.imshow(corr, text_auto=True, title="Correlation heatmap")



def sankey_from_edges(
    edges: pd.DataFrame,
    source: str,
    target: str,
    value: str,
    title: str = "Money Flow Sankey",
    min_link_value: float | None = None,
    top_n: int = 50,
    height: int = 850,
):
    """
    Create a readable Plotly Sankey diagram for money-flow visualization.

    Parameters
    ----------
    edges : pd.DataFrame
        Aggregated edge dataframe. Must contain source, target and value columns.

    source : str
        Source column name.

    target : str
        Target column name.

    value : str
        Numeric flow value column name.

    title : str
        Chart title.

    min_link_value : float | None
        Optional minimum flow value. Links below this value are excluded.

    top_n : int
        Number of top flows to display.

    height : int
        Plot height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Sankey diagram figure.
    """
    import pandas as pd
    import plotly.graph_objects as go

    required_columns = [source, target, value]

    missing_columns = [
        col for col in required_columns
        if col not in edges.columns
    ]

    if missing_columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"Missing columns for Sankey: {', '.join(missing_columns)}",
            height=height,
        )
        return fig

    df = edges[[source, target, value]].copy()

    df[source] = df[source].astype(str).str.strip()
    df[target] = df[target].astype(str).str.strip()
    df[value] = pd.to_numeric(df[value], errors="coerce")

    df = df.dropna(subset=[source, target, value])

    df = df[
        (df[source] != "")
        & (df[target] != "")
        & (df[source].str.lower() != "nan")
        & (df[target].str.lower() != "nan")
        & (df[value] > 0)
    ]

    if min_link_value is not None:
        df = df[df[value] >= min_link_value]

    df = (
        df.sort_values(value, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No valid Sankey data available",
            height=height,
        )
        return fig

    labels = pd.Index(
        pd.concat([df[source], df[target]])
        .astype(str)
        .unique()
    )

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    source_ids = df[source].astype(str).map(label_to_id)
    target_ids = df[target].astype(str).map(label_to_id)

    max_value = df[value].max()

    if max_value and max_value > 0:
        normalized_opacity = (
            0.25 + 0.55 * (df[value] / max_value)
        ).clip(0.25, 0.80)
    else:
        normalized_opacity = pd.Series(
            [0.45] * len(df),
            index=df.index
        )

    link_colors = [
        f"rgba(34, 197, 94, {opacity:.2f})"
        for opacity in normalized_opacity
    ]

    node_colors = []

    source_label_set = set(df[source].astype(str))
    target_label_set = set(df[target].astype(str))

    for label in labels:
        if label in source_label_set and label in target_label_set:
            node_colors.append("rgba(168, 85, 247, 0.90)")  # intermediate/both
        elif label in source_label_set:
            node_colors.append("rgba(59, 130, 246, 0.90)")  # source
        else:
            node_colors.append("rgba(249, 115, 22, 0.90)")  # target

    customdata = df[[source, target, value]].copy()
    customdata[value] = customdata[value].map(lambda x: f"{x:,.2f}")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=28,
                    thickness=24,
                    line=dict(
                        color="rgba(70, 70, 70, 0.65)",
                        width=0.8,
                    ),
                    label=labels.tolist(),
                    color=node_colors,
                    hovertemplate=(
                        "<b>Node:</b> %{label}<extra></extra>"
                    ),
                ),
                link=dict(
                    source=source_ids,
                    target=target_ids,
                    value=df[value],
                    color=link_colors,
                    customdata=customdata.astype(str),
                    hovertemplate=(
                        "<b>Source:</b> %{customdata[0]}<br>"
                        "<b>Target:</b> %{customdata[1]}<br>"
                        "<b>Value:</b> %{customdata[2]}"
                        "<extra></extra>"
                    ),
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
        ),
        height=height,
        font=dict(
            size=14,
            color="#111827",
        ),
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20,
        ),
    )

    return fig



def network_figure(G, max_nodes: int = 300):
    H = G.copy()
    if H.number_of_nodes() > max_nodes:
        top_nodes = sorted(dict(H.degree()).items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        H = H.subgraph([n for n, _ in top_nodes]).copy()
    pos = nx.spring_layout(H, seed=42, k=0.35)
    edge_x, edge_y = [], []
    for e in H.edges():
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5), hoverinfo="none", mode="lines")
    node_x, node_y, text, size = [], [], [], []
    deg = dict(H.degree())
    for n in H.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y); text.append(str(n)); size.append(8 + deg.get(n, 1) * 1.5)
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center", hoverinfo="text", marker=dict(size=size, color=size, showscale=True))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Transaction Network", showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def benford_chart(benford_df: pd.DataFrame):

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=benford_df["digit"],
        y=benford_df["expected_rate"],
        name="Benford expected",
        marker_color="#4ade80"
    ))

    fig.add_trace(go.Bar(
        x=benford_df["digit"],
        y=benford_df["observed_rate"],
        name="Observed",
        marker_color="#f87171"
    ))

    fig.update_layout(
        title="Benford's Law — First Digit Distribution",
        xaxis_title="First digit",
        yaxis_title="Frequency",
        barmode="group",
        template="plotly_white",
        height=450
    )

    fig.update_xaxes(dtick=1)

    return fig
