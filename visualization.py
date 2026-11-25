"""Plotly visualization helpers (Polars dataframes stay in memory only briefly)."""
from __future__ import annotations

from typing import Optional

import networkx as nx
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

from storage import to_polars

_PLOT_ROWS = 200_000  # keep browser-side payload bounded for very large files


def _plot_frame(df, limit: int = _PLOT_ROWS):
    frame = to_polars(df)
    if frame.height > limit:
        return frame.sample(n=limit, shuffle=True, seed=42)
    return frame


def histogram(df, x: str):
    return px.histogram(_plot_frame(df), x=x, marginal="box", title=f"Distribution of {x}")


def boxplot(df, y: str, x: Optional[str] = None):
    return px.box(_plot_frame(df), x=x, y=y, points="outliers", title=f"Boxplot of {y}")


def line_chart(df, x: str, y: str, title: str = "Time series"):
    return px.line(_plot_frame(df), x=x, y=y, title=title)


def bar_chart(df, x: str, y: str, title: str = "Bar chart"):
    return px.bar(_plot_frame(df), x=x, y=y, title=title)


def scatter(df, x: str, y: str, color: Optional[str] = None):
    return px.scatter(_plot_frame(df), x=x, y=y, color=color, title=f"{y} vs {x}")


def heatmap_corr(df):
    frame = to_polars(df)
    numeric = frame.select(pl.selectors.numeric())
    corr = numeric.corr().to_pandas() if numeric.width else None
    return px.imshow(corr, text_auto=True, title="Correlation heatmap")


def sankey_from_edges(
    edges,
    source: str,
    target: str,
    value: str,
    title: str = "Money Flow Sankey",
    min_link_value: float | None = None,
    top_n: int = 50,
    height: int = 850,
):
    """Create a readable Plotly Sankey diagram for money-flow visualization."""
    required_columns = [source, target, value]
    missing_columns = [col for col in required_columns if col not in to_polars(edges).columns]
    if missing_columns:
        fig = go.Figure()
        fig.update_layout(title=f"Missing columns for Sankey: {', '.join(missing_columns)}", height=height)
        return fig

    df = to_polars(edges).select(source, target, value).with_columns(
        pl.col(source).cast(pl.String).str.strip_chars(),
        pl.col(target).cast(pl.String).str.strip_chars(),
        pl.col(value).cast(pl.Float64, strict=False),
    ).filter(
        pl.col(source).is_not_null() & pl.col(target).is_not_null() & pl.col(value).is_not_null()
        & (pl.col(source) != "") & (pl.col(target) != "")
        & (pl.col(source).str.to_lowercase() != "nan") & (pl.col(target).str.to_lowercase() != "nan")
        & (pl.col(value) > 0)
    )
    if min_link_value is not None:
        df = df.filter(pl.col(value) >= min_link_value)
    df = df.sort(value, descending=True).head(top_n)

    if df.is_empty():
        fig = go.Figure()
        fig.update_layout(title="No valid Sankey data available", height=height)
        return fig

    labels = pl.concat([df[source], df[target]]).unique(maintain_order=True).to_list()
    label_to_id = {label: i for i, label in enumerate(labels)}
    source_ids = [label_to_id[v] for v in df[source].to_list()]
    target_ids = [label_to_id[v] for v in df[target].to_list()]
    values = df[value].to_list()
    max_value = max(values) if values else 0.0

    if max_value and max_value > 0:
        normalized_opacity = [max(0.25, min(0.80, 0.25 + 0.55 * (v / max_value))) for v in values]
    else:
        normalized_opacity = [0.45] * len(df)

    link_colors = [f"rgba(34, 197, 94, {opacity:.2f})" for opacity in normalized_opacity]

    source_labels = set(df[source].to_list())
    target_labels = set(df[target].to_list())
    node_colors = []
    for label in labels:
        if label in source_labels and label in target_labels:
            node_colors.append("rgba(168, 85, 247, 0.90)")
        elif label in source_labels:
            node_colors.append("rgba(59, 130, 246, 0.90)")
        else:
            node_colors.append("rgba(249, 115, 22, 0.90)")

    src_vals, dst_vals, val_vals = (df[c].to_list() for c in (source, target, value))
    customdata = [
        [str(s), str(t), f"{v:,.2f}"] for s, t, v in zip(src_vals, dst_vals, val_vals)
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=28,
                    thickness=24,
                    line=dict(color="rgba(70, 70, 70, 0.65)", width=0.8),
                    label=[str(x) for x in labels],
                    color=node_colors,
                    hovertemplate="<b>Node:</b> %{label}<extra></extra>",
                ),
                link=dict(
                    source=source_ids,
                    target=target_ids,
                    value=values,
                    color=link_colors,
                    customdata=customdata,
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
        title=dict(text=title, x=0.01, xanchor="left"),
        height=height,
        font=dict(size=14, color="#111827"),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def network_figure(G, max_nodes: int = 250):
    H = G.copy()
    if H.number_of_nodes() > max_nodes:
        top_nodes = sorted(dict(H.degree()).items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        H = H.subgraph([n for n, _ in top_nodes]).copy()

    pos = nx.spring_layout(H, seed=42, k=0.4)
    edge_x, edge_y = [], []
    for e in H.edges():
        if e[0] in pos and e[1] in pos:
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.7, color="#94a3b8"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, text, size = [], [], [], []
    deg = dict(H.degree())
    for n in H.nodes():
        if n in pos:
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            d = deg.get(n, 1)
            text.append(f"Node: {n}<br>Connections: {d}")
            size.append(min(32, 10 + d * 2))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        text=text,
        hoverinfo="text",
        marker=dict(size=size, color=size, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Degree")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Transaction Network Graph (Գործարքների ցանցային գրաֆ)",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=650,
    )
    return fig


def benford_chart(benford_df):
    frame = to_polars(benford_df)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=frame["digit"].to_list(),
        y=frame["expected_rate"].to_list(),
        name="Benford expected",
        marker_color="#4ade80",
    ))
    fig.add_trace(go.Bar(
        x=frame["digit"].to_list(),
        y=frame["observed_rate"].to_list(),
        name="Observed",
        marker_color="#f87171",
    ))
    fig.update_layout(
        title="Benford's Law — First Digit Distribution",
        xaxis_title="First digit",
        yaxis_title="Frequency",
        barmode="group",
        template="plotly_white",
        height=450,
    )
    fig.update_xaxes(dtick=1)
    return fig
