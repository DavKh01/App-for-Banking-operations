
"""Streamlit entrypoint for banking AML analytics."""
from __future__ import annotations

import json

from typing import Dict

import pandas as pd
import streamlit as st
import plotly.express as px

from config import APP_TITLE, APP_ICON, CANONICAL_FIELDS, DEFAULT_RULES, CANONICAL_FIELDS_2
from data_loader import DataLoader, LoadOptions
from preprocessing import DataPreprocessor
from analysis import AnalyticsEngine
from risk_engine import RiskEngine
from graph_analysis import TransactionGraphAnalyzer
from visualization import histogram, boxplot, line_chart, heatmap_corr, network_figure, sankey_from_edges,benford_chart
from ml_analysis import MLAnalyzer
from utils import dataframe_info, memory_usage_mb, normalize_mapping, download_dataframe,\
    dataframe_pdf_bytes, dataframe_pptx_bytes

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


def init_state():
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("mapping", {})
    st.session_state.setdefault("rules", json.loads(json.dumps(DEFAULT_RULES)))


def sidebar_mapping(df: pd.DataFrame) -> Dict[str, str]:
    st.sidebar.subheader("Column Mapping")
    options = ["<Not available>"] + list(df.columns)
    mapping = {}
    canon = CANONICAL_FIELDS if df.columns[0] == "Doc Type" else CANONICAL_FIELDS_2

    column_map = {c.strip().lower(): c for c in df.columns}

    for key, label in canon.items():
        guessed = column_map.get(label.strip().lower(), "<Not available>")

        idx = options.index(guessed) if guessed in options else 0

        mapping[key] = st.sidebar.selectbox(
            label,
            options,
            index=idx,
            key=f"map_{key}",
        )

    mapping = normalize_mapping(mapping)
    st.session_state["mapping"] = mapping
    return mapping


def load_page():
    st.header("1. Load Data")
    uploaded = st.file_uploader("Upload Excel, CSV or TSV", type=["xlsx", "xls", "csv", "tsv"])
    if not uploaded:
        st.info("Upload a banking transaction dataset to begin.")
        return
    with st.expander("Loading options", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        delimiter = c1.text_input("Delimiter", value=",")
        encoding = c2.text_input("Encoding", value="utf-8")
        decimal = c3.text_input("Decimal separator", value=".")
        thousands = c4.text_input("Thousands separator", value="")
        c5, c6, c7, c8 = st.columns(4)
        header = c5.number_input("Header row (0-based, -1 for no header)", min_value=-1, value=0)
        skiprows = c6.number_input("Skip rows", min_value=0, value=0)
        nrows = c7.number_input("N rows (0 = all)", min_value=0, value=0)
        load_as_text = c8.checkbox("Load everything as text", value=False)
        sheet_name = 0
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = DataLoader.excel_sheets(uploaded)
                uploaded.seek(0)
                sheet_name = st.selectbox("Excel sheet", sheets)
            except Exception as exc:
                st.warning(f"Could not read sheet names: {exc}")
        opts = LoadOptions(delimiter=delimiter, encoding=encoding, decimal=decimal, thousands=thousands or None, sheet_name=sheet_name, header=None if header < 0 else header, skiprows=skiprows, nrows=None if nrows == 0 else nrows, load_as_text=load_as_text)
    if st.button("Load dataset", type="primary"):
        try:
            st.session_state["df"] = DataLoader.load(uploaded.getvalue(), uploaded.name, opts)
            st.success("Dataset loaded successfully.")
        except Exception as exc:
            st.error(f"Loading failed: {exc}")
    df = pd.read_excel(uploaded,skiprows=skiprows, nrows=100 if nrows == 0 else nrows)
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(100).astype(str), width='content')

def cleaning_page(df):
    st.header("2. Data Cleaning")
    prep = DataPreprocessor(df)
    with st.form("clean_form"):
        remove_dups = st.checkbox("Remove duplicates")
        remove_fees = st.checkbox("Remove fees")
        remove_loans = st.checkbox("Remove loans")
        dup_subset = st.multiselect("Duplicate subset columns", df.columns)
        trim_cols = st.multiselect("Trim spaces in columns", df.columns)
        case_cols = st.multiselect("Case conversion columns", df.columns)
        case_mode = st.radio("Case mode", ["upper", "lower"], horizontal=True)
        drop_missing_cols = st.multiselect("Remove rows missing selected columns", df.columns)
        fill_cols = st.multiselect("Fill missing values in columns", df.columns)
        fill_value = st.text_input("Fill value", "")
        convert_col = st.selectbox("Convert datatype column", ["<None>"] + list(df.columns))
        convert_type = st.selectbox("Target datatype", ["numeric", "datetime", "string", "category"])
        remove_rows_by_text = st.selectbox("Delete if contains", ["<None>"] + list(df.columns))
        target_text = st.text_input("Target text", "").lower()
        submitted = st.form_submit_button("Apply cleaning")
    if submitted:
        if remove_dups: prep.remove_duplicates(dup_subset or None)
        if remove_fees: prep.remove_rows("միջնորդավճար")
        if remove_loans: prep.remove_rows("վարկի մարում")
        if trim_cols: prep.trim_spaces(trim_cols)
        if case_cols: prep.change_case(case_cols, case_mode)
        if drop_missing_cols: prep.remove_missing(drop_missing_cols)
        if fill_cols: prep.fill_missing(fill_cols, fill_value)
        if convert_col != "<None>": prep.convert_dtype(convert_col, convert_type)
        if remove_rows_by_text != "<None>": prep = prep.delete_rows(remove_rows_by_text, target_text)
        st.session_state["df"] = prep.result()
        st.success("Cleaning applied.")
    st.dataframe(st.session_state["df"].head(200), width='content')


def dashboard_page(df, mapping):
    st.header("3. Dashboard")
    engine = AnalyticsEngine(df, mapping)
    metrics = engine.dashboard_metrics()
    cols = st.columns(4)
    for i, (k, v) in enumerate(metrics.items()):
        cols[i % 4].metric(k.replace("_", " ").title(), "-" if v is None else f"{v:,.2f}" if isinstance(v, float) else str(v))
    if "amount" in mapping:
        st.plotly_chart(histogram(engine.prepared(), "_amount"), width='content')
    if "transaction_date" in mapping and "amount" in mapping:
        ts = engine.time_series(st.selectbox("Frequency", ["D", "W", "ME", "YE"]))
        if not ts.empty:
            st.plotly_chart(line_chart(ts, "_date", "sum", "Turnover over time"), width='content')
    df = st.session_state.get("df")
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(100), width='content')
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Memory", f"{memory_usage_mb(df):,.2f} MB")
        st.subheader("DataFrame information")
        st.dataframe(dataframe_info(df), width='content')
        st.subheader("Missing values")
        st.bar_chart(df.isna().sum())


def statistics_page(df, mapping):
    st.header("4. Statistical Analysis")
    engine = AnalyticsEngine(df, mapping)
    dfp = engine.prepared()
    st.dataframe(engine.amount_statistics(), width='content')
    if "_amount" in dfp:
        c1, c2 = st.columns(2)
        c1.plotly_chart(histogram(dfp, "_amount"), width='content')
        c2.plotly_chart(boxplot(dfp, "_amount"), width='content')
    st.plotly_chart(heatmap_corr(dfp), width='content')
    numeric_cols = list(dfp.select_dtypes("number").columns)
    if len(numeric_cols) >= 2:
        x = st.selectbox("Scatter X", numeric_cols)
        y = st.selectbox("Scatter Y", numeric_cols, index=1)
        st.plotly_chart(px.scatter(dfp, x=x, y=y), width='content')

    st.subheader("Benford's Law Analysis")

    benford_df = engine.benfords_law()

    if benford_df.empty:
        st.info("Map and prepare the Amount column first to run Benford analysis.")
    else:
        st.plotly_chart(benford_chart(benford_df), width='content')
        st.dataframe(benford_df, width='content')

        mad = benford_df["absolute_difference"].mean()
        st.metric("Mean Absolute Deviation", f"{mad:.4f}")

        if mad < 0.006:
            st.success("Benford conformity: Close conformity")
        elif mad < 0.012:
            st.info("Benford conformity: Acceptable conformity")
        elif mad < 0.015:
            st.warning("Benford conformity: Marginal conformity")
        else:
            st.error("Benford conformity: Non-conformity detected")


def aml_page(df, mapping):
    st.header("5. AML Indicators")
    results = AnalyticsEngine(df, mapping).aml_indicators()
    if not results:
        st.warning("Map relevant fields such as amount, debit/credit accounts, customer, date and document number.")
        return
    selected = st.selectbox("AML analysis", list(results.keys()))
    st.dataframe(results[selected], width='content')
    if not results[selected].empty:
        download_dataframe(results[selected], selected.lower().replace(" ", "_"))


def risk_page(df, mapping):
    st.header("6. Risk Scoring Engine")
    st.write("Enable/disable rules and tune thresholds/weights. Scores are capped at 100 and include explanations.")
    rules = st.session_state["rules"]
    with st.expander("Rule configuration", expanded=True):
        for name, rule in rules.items():
            c1, c2, c3 = st.columns([2, 1, 1])
            rule["enabled"] = c1.checkbox(name, value=rule.get("enabled", True), key=f"rule_enabled_{name}")
            rule["weight"] = c2.number_input("Weight", value=float(rule.get("weight", 0)), key=f"rule_weight_{name}")
            if rule.get("threshold") is not None:
                rule["threshold"] = c3.number_input("Threshold", value=float(rule.get("threshold", 0)), key=f"rule_threshold_{name}")
    scored = RiskEngine(rules).score(df, mapping)
    st.dataframe(scored.head(1000), width='content')
    st.plotly_chart(px.histogram(scored, x="risk_score", color="risk_level", title="Risk score distribution"), width='content')
    download_dataframe(scored, "risk_scoring")


def graph_page(df, mapping):
    st.header("7. Graph Analysis")
    source_key = st.selectbox("Source node", ["debit_account", "customer_id", "contract_customer", "document_number","debit_name"])
    target_key = st.selectbox("Target node", ["credit_account", "customer_id", "contract_customer", "document_number","credit_name"])
    directed = st.checkbox("Directed graph", True)
    multigraph = st.checkbox("MultiGraph", False)
    try:
        analyzer = TransactionGraphAnalyzer(df, mapping)
        g = analyzer.build_graph(source_key, target_key, directed=directed, multigraph=multigraph)
        st.json(analyzer.metrics(g))
        st.plotly_chart(network_figure(g), width='content')
        tabs = st.tabs(["Centrality", "Cycles"])
        tabs[0].dataframe(analyzer.centrality_table(g), width='content')
        tabs[1].dataframe(analyzer.cycles_table(g), width='content')
    except Exception as exc:
        st.warning(str(exc))


def money_flow_page(df, mapping):

    st.header("Money Flow Analysis")

    st.caption(
        "Visualize aggregated money flows between accounts, customers, branches, "
        "currencies or documents. Best used for Debit Account → Credit Account flows."
    )

    possible_nodes = [
        "bank_account",
        "debit_account",
        "credit_account",
        "debit_name",
        "credit_name",
        "customer_id",
        "contract_customer",
        "document_number",
        "branch",
        "currency",
    ]

    available_nodes = [
        node for node in possible_nodes
        if mapping.get(node) is not None
    ]

    if len(available_nodes) < 2:
        st.warning(
            "Please map at least two node columns in the sidebar "
            "for Money Flow Analysis."
        )
        return

    if not mapping.get("amount"):
        st.warning("Please map the Amount column in the sidebar.")
        return

    st.subheader("Flow Configuration")

    c1, c2, c3 = st.columns(3)

    default_source_index = (
        available_nodes.index("debit_account")
        if "debit_account" in available_nodes
        else 0
    )

    default_target_index = (
        available_nodes.index("credit_account")
        if "credit_account" in available_nodes
        else min(1, len(available_nodes) - 1)
    )

    with c1:
        source_key = st.selectbox(
            "Source node",
            available_nodes,
            index=default_source_index,
            help="Usually Debit Account, Sender, Customer or Branch."
        )

    with c2:
        target_key = st.selectbox(
            "Target node",
            available_nodes,
            index=default_target_index,
            help="Usually Credit Account, Receiver, Contract Customer or Branch."
        )

    with c3:
        aggregation_method = st.selectbox(
            "Aggregation",
            ["sum", "count", "mean", "max"],
            help="How to aggregate transactions between source and target."
        )

    source_col = mapping.get(source_key)
    target_col = mapping.get(target_key)
    amount_col = mapping.get("amount")

    if source_col == target_col:
        st.warning(
            "Source and Target are mapped to the same column. "
            "The Sankey may be difficult to interpret."
        )

    st.subheader("Display Settings")

    c1, c2, c3 = st.columns(3)

    with c1:
        top_n = st.slider(
            "Top flows to display",
            min_value=10,
            max_value=300,
            value=30,
            step=10,
            help="Sankey diagrams are clearer with fewer top flows."
        )

    with c2:
        chart_height = st.slider(
            "Chart height",
            min_value=500,
            max_value=1400,
            value=850,
            step=50
        )

    with c3:
        min_amount = st.number_input(
            "Minimum flow amount",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Exclude small flows from the visualization."
        )

    try:
        edges = df[[source_col, target_col, amount_col]].copy()

        edges[source_col] = edges[source_col].astype(str).str.strip()
        edges[target_col] = edges[target_col].astype(str).str.strip()

        edges[amount_col] = pd.to_numeric(
            edges[amount_col],
            errors="coerce"
        ).abs()

        edges = edges.dropna(
            subset=[source_col, target_col, amount_col]
        )

        edges = edges[
            (edges[source_col] != "")
            & (edges[target_col] != "")
            & (edges[source_col].str.lower() != "nan")
            & (edges[target_col].str.lower() != "nan")
            & (edges[amount_col] > 0)
            ]

        if min_amount > 0:
            edges = edges[edges[amount_col] >= min_amount]

        if edges.empty:
            st.info("No valid money flow data found after filtering.")
            return

        if aggregation_method == "sum":
            flow_df = (
                edges
                .groupby([source_col, target_col], as_index=False)
                .agg(
                    total_amount=(amount_col, "sum"),
                    transaction_count=(amount_col, "count"),
                    average_amount=(amount_col, "mean"),
                    max_amount=(amount_col, "max"),
                )
            )
            value_col = "total_amount"

        elif aggregation_method == "count":
            flow_df = (
                edges
                .groupby([source_col, target_col], as_index=False)
                .agg(
                    total_amount=(amount_col, "sum"),
                    transaction_count=(amount_col, "count"),
                    average_amount=(amount_col, "mean"),
                    max_amount=(amount_col, "max"),
                )
            )
            value_col = "transaction_count"

        elif aggregation_method == "mean":
            flow_df = (
                edges
                .groupby([source_col, target_col], as_index=False)
                .agg(
                    total_amount=(amount_col, "sum"),
                    transaction_count=(amount_col, "count"),
                    average_amount=(amount_col, "mean"),
                    max_amount=(amount_col, "max"),
                )
            )
            value_col = "average_amount"

        else:
            flow_df = (
                edges
                .groupby([source_col, target_col], as_index=False)
                .agg(
                    total_amount=(amount_col, "sum"),
                    transaction_count=(amount_col, "count"),
                    average_amount=(amount_col, "mean"),
                    max_amount=(amount_col, "max"),
                )
            )
            value_col = "max_amount"

        flow_df = flow_df.sort_values(
            value_col,
            ascending=False
        ).reset_index(drop=True)

        st.subheader("Optional Focus Filter")

        focus_options = ["All"] + sorted(
            flow_df[source_col].dropna().astype(str).unique().tolist()
        )

        selected_source = st.selectbox(
            "Focus on one source",
            focus_options,
            help="Use this if the Sankey is too crowded."
        )

        if selected_source != "All":
            flow_df = flow_df[
                flow_df[source_col].astype(str) == selected_source
                ]

        if flow_df.empty:
            st.info("No flows available for the selected source.")
            return

        sankey_df = flow_df.head(top_n).copy()

        st.subheader("Summary Metrics")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Unique Sources", f"{flow_df[source_col].nunique():,}")
        m2.metric("Unique Targets", f"{flow_df[target_col].nunique():,}")
        m3.metric("Flow Links", f"{len(flow_df):,}")
        m4.metric("Total Amount", f"{flow_df['total_amount'].sum():,.2f}")

        st.subheader("Top Aggregated Money Flows")

        display_df = flow_df.copy()
        for col in ["total_amount", "average_amount", "max_amount"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)

        st.dataframe(
            display_df.head(500),
            width='content',
            height=500
        )

        st.subheader("Money Flow Sankey Diagram")

        st.caption(
            f"Showing top {min(top_n, len(sankey_df))} flows by "
            f"{value_col.replace('_', ' ')}."
        )

        fig = sankey_from_edges(
            sankey_df,
            source=source_col,
            target=target_col,
            value=value_col,
            title=f"Money Flow: {source_col} → {target_col}",
            top_n=top_n,
            height=chart_height,
            min_link_value=None,
        )

        st.plotly_chart(
            fig,
            width='content'
        )

        st.subheader("AML Interpretation Hints")

        st.info(
            """
            Look for:

            - One source sending funds to many targets
            - Many sources sending funds to one target
            - Very large flows between unusual counterparties
            - Repeated flows through the same intermediate accounts
            - High transaction count but relatively similar amounts
            - Flows involving high-risk branches, currencies or customers
            """
        )

        download_dataframe(
            flow_df,
            "money_flow_analysis"
        )

    except Exception as exc:
        st.error(f"Money Flow Analysis failed: {exc}")


def customer_account_page(df, mapping):
    st.header("8. Customer & Account Analytics")
    engine = AnalyticsEngine(df, mapping)
    tab1, tab2 = st.tabs(["Customers", "Accounts"])
    cust = engine.customer_analytics()
    acc = engine.account_analytics()
    tab1.dataframe(cust, width='content')
    tab2.dataframe(acc, width='content')
    if not cust.empty: download_dataframe(cust, "customer_analytics")
    if not acc.empty: download_dataframe(acc, "account_analytics")


def filters_pivot_page(df):
    st.header("9. Filters, Duplicates & Pivot Tables")
    filtered = df.copy()
    with st.expander("Advanced filters", expanded=True):
        col = st.selectbox("Filter column", ["<None>"] + list(df.columns))
        if col != "<None>":
            mode = st.selectbox("Filter mode", ["contains", "starts with", "ends with", "regex", "null", "not null","doesn't contain"])
            val = st.text_input("Filter value")
            s = filtered[col].astype(str)
            if st.button("Apply filter"):
                if mode == "contains": filtered = filtered[s.str.contains(val, case=False, na=False)]
                elif mode == "starts with": filtered = filtered[s.str.startswith(val, na=False)]
                elif mode == "ends with": filtered = filtered[s.str.endswith(val, na=False)]
                elif mode == "regex": filtered = filtered[s.str.contains(val, regex=True, na=False)]
                elif mode == "null": filtered = filtered[filtered[col].isna()]
                elif mode == "not null": filtered = filtered[filtered[col].notna()]
                elif mode == "doesn't contain" : filtered = filtered[~s.str.contains(val, case=False, na=False)]
    st.dataframe(filtered.head(1000), width='content')
    with st.expander("Duplicate detection"):
        dup_cols = st.multiselect("Duplicate columns", df.columns, key="dup_cols")
        if dup_cols:
            dups = df[df.duplicated(dup_cols, keep=False)].sort_values(dup_cols)
            st.dataframe(dups, width='content')
    with st.expander("Pivot builder"):
        rows = st.multiselect("Rows", df.columns, key="pivot_rows")
        cols = st.multiselect("Columns", df.columns, key="pivot_cols")
        vals = st.selectbox("Values", ["<None>"] + list(df.columns))
        agg = st.selectbox("Aggregation", ["count", "sum", "mean", "median", "min", "max", "nunique"])
        if rows and vals != "<None>":
            try:
                pivot = pd.pivot_table(df, index=rows, columns=cols or None, values=vals, aggfunc=agg, fill_value=0)
                st.dataframe(pivot, width='content')
            except Exception as exc:
                st.error(exc)



def ml_page(df):
    st.header("10. Machine Learning")
    numeric_cols = list(df.select_dtypes("number").columns)
    if not numeric_cols:
        st.warning("Convert/select numeric columns first.")
        return
    cols = st.multiselect("Numeric features", numeric_cols, default=numeric_cols[: min(5, len(numeric_cols))])
    if not cols:
        return
    method = st.selectbox("Method", ["Isolation Forest", "KMeans", "DBSCAN", "PCA"])
    analyzer = MLAnalyzer(df)
    if method == "Isolation Forest":
        contamination = st.slider("Contamination", 0.001, 0.20, 0.02)
        result = analyzer.isolation_forest(cols, contamination)
    elif method == "KMeans":
        k = st.slider("Clusters", 2, 20, 5)
        result = analyzer.kmeans(cols, k)
    elif method == "DBSCAN":
        eps = st.slider("EPS", 0.1, 5.0, 0.8)
        min_samples = st.slider("Min samples", 2, 100, 10)
        result = analyzer.dbscan(cols, eps, min_samples)
    else:
        result = analyzer.pca_projection(cols)
        st.plotly_chart(px.scatter(result, x="PC1", y="PC2" if "PC2" in result else "PC1"), width='content')
    st.dataframe(result.head(1000), width='content')
    download_dataframe(result, "ml_results")


def export_page(df):
    st.header("11. Export")
    download_dataframe(df, "cleaned_transactions")
    try:
        st.download_button("Download Parquet", df.to_parquet(index=False), "cleaned_transactions.parquet")
    except Exception as exc:
        st.info(f"Parquet export unavailable: {exc}")
    html = df.head(1000).to_html(index=False)
    st.download_button("Download HTML report", html.encode("utf-8"), "report.html", "text/html")
    st.download_button("Download PDF report", dataframe_pdf_bytes(df), "aml_report.pdf", "application/pdf")
    st.download_button("Download PowerPoint report", dataframe_pptx_bytes(df), "aml_report.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")


def main():
    init_state()
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Configurable AML, fraud detection, customer behavior, network analysis and risk scoring for banking transactions.")
    df = st.session_state.get("df")
    page = st.sidebar.radio("Navigation", ["Load Data", "Cleaning", "Dashboard", "Statistics", "AML", "Risk Scoring", "Graph Analysis","Money Flow", "Customer/Account", "Filters/Pivot", "Machine Learning", "Export"])
    if page == "Load Data" or df is None:
        load_page()
        return
    mapping = sidebar_mapping(df)
    if page == "Cleaning": cleaning_page(df)
    elif page == "Dashboard": dashboard_page(df, mapping)
    elif page == "Statistics": statistics_page(df, mapping)
    elif page == "AML": aml_page(df, mapping)
    elif page == "Risk Scoring": risk_page(df, mapping)
    elif page == "Graph Analysis": graph_page(df, mapping)
    elif page == "Customer/Account": customer_account_page(df, mapping)
    elif page == "Filters/Pivot": filters_pivot_page(df)
    elif page == "Machine Learning": ml_page(df)
    elif page == "Export": export_page(df)
    elif page == "Money Flow": money_flow_page(df, mapping)


if __name__ == "__main__":
    main()
