
"""Streamlit entrypoint for banking AML analytics."""
from __future__ import annotations

import json
from typing import Dict

import pandas as pd
import polars as pl
import plotly.express as px
import streamlit as st

from config import APP_TITLE, APP_ICON, CANONICAL_FIELDS, DEFAULT_RULES, CANONICAL_FIELDS_2, CANONICAL_FIELDS_3, \
    LEGAL_ENTITY_PREFIXES, DEFAULT_RULES_LEGAL
from data_loader import DataLoader, LoadOptions
from preprocessing import DataPreprocessor
from analysis import AnalyticsEngine
from risk_engine import RiskEngine
from graph_analysis import TransactionGraphAnalyzer
from visualization import histogram, boxplot, line_chart, heatmap_corr, network_figure, sankey_from_edges, benford_chart

from storage import Dataset, digest, make_dataset
from utils import dataframe_info, memory_usage_mb, normalize_mapping, download_dataframe, download_csv

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


def init_state():
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("mapping", {})
    st.session_state.setdefault("rules_physical", json.loads(json.dumps(DEFAULT_RULES)))
    st.session_state.setdefault("rules_legal", json.loads(json.dumps(DEFAULT_RULES_LEGAL)))


def show_table(data, *, height=None):
    """Render a table with readable widths and left-to-right scrolling.

    Keeping every column at a fixed minimum width makes wide datasets wider
    than the page, so Streamlit shows a horizontal scrollbar instead of
    squeezing (and truncating) the values.  No column is pinned.
    """
    columns = list(getattr(data, "columns", []))
    config = {
        str(col): st.column_config.Column(width=180)
        for col in columns
    }
    kwargs = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    if config:
        kwargs["column_config"] = config
    st.dataframe(data, **kwargs)


def sidebar_mapping(df: Dataset) -> Dict[str, str]:
    st.sidebar.subheader("Column Mapping")
    options = ["<Not available>"] + list(df.columns)
    mapping = {}
    if df.columns[0] == "Doc Type":
        canon = CANONICAL_FIELDS
    elif df.columns[0] == "DocType":
        canon = CANONICAL_FIELDS_3
    else:
        canon = CANONICAL_FIELDS_2

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

    if "debit_name" not in mapping and "debit_account" in mapping:
        mapping["debit_name"] = mapping["debit_account"]

    if "credit_name" not in mapping and "credit_account" in mapping:
        mapping["credit_name"] = mapping["credit_account"]
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
    limit_rows = 100 if nrows == 0 else nrows
    preview_opts = LoadOptions(delimiter=delimiter, encoding=encoding, decimal=decimal,
                               thousands=thousands or None, sheet_name=sheet_name,
                               header=None if header < 0 else header, skiprows=skiprows,
                               nrows=limit_rows, load_as_text=load_as_text)
    try:
        preview_ds = DataLoader.load(uploaded.getvalue(), uploaded.name, preview_opts)
        st.subheader("Preview")
        show_table(preview_ds.pandas_head(100).astype(str))
    except Exception as exc:
        st.info(f"Preview unavailable: {exc}")


def cleaning_page(df: Dataset):
    st.header("2. Data Cleaning")
    prep = DataPreprocessor(df)
    with st.form("clean_form"):
        #remove_dups = st.checkbox("Remove duplicates")
        remove_fees = st.checkbox("Remove fees")
        remove_loans = st.checkbox("Remove loans")
        remove_deposits = st.checkbox("Remove deposits")
        #dup_subset = st.multiselect("Duplicate subset columns", df.columns)
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
        #if remove_dups: prep.remove_duplicates(dup_subset or None)
        if remove_fees: prep.remove_rows("միջնորդավճար")
        if remove_loans: prep.remove_rows("վարկ")
        if remove_deposits: prep.remove_rows("ավանդ")
        if trim_cols: prep.trim_spaces(trim_cols)
        if case_cols: prep.change_case(case_cols, case_mode)
        if drop_missing_cols: prep.remove_missing(drop_missing_cols)
        if fill_cols: prep.fill_missing(fill_cols, fill_value)
        if convert_col != "<None>": prep.convert_dtype(convert_col, convert_type)
        if remove_rows_by_text != "<None>": prep = prep.delete_rows(remove_rows_by_text, target_text)
        st.session_state["df"] = prep.result()
        st.success("Cleaning applied.")
    show_table(st.session_state["df"].pandas_head(200))


def dashboard_page(df: Dataset, mapping):
    st.header("3. Dashboard")
    engine = AnalyticsEngine(df, mapping)
    metrics = engine.dashboard_metrics()
    cols = st.columns(4)
    for i, (k, v) in enumerate(metrics.items()):
        cols[i % 4].metric(k.replace("_", " ").title(), "-" if v is None else f"{v:,.2f}" if isinstance(v, float) else str(v))

    df = st.session_state.get("df")
    if df is not None:
        st.subheader("Preview")
        show_table(df.pandas_head(100))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Memory", f"{memory_usage_mb(df):,.2f} MB")
        st.subheader("DataFrame information")
        show_table(dataframe_info(df))
        st.subheader("Missing values")
        info = dataframe_info(df).to_pandas().set_index("column")["nulls"]
        st.bar_chart(info)


def aml_page(df: Dataset, mapping):
    st.header("5. AML Indicators")
    results = AnalyticsEngine(df, mapping).aml_indicators()
    if not results:
        st.warning("Map relevant fields such as amount, debit/credit accounts, customer, date and document number.")
        return
    selected = st.selectbox("AML analysis", list(results.keys()))
    show_table(results[selected])
    if not results[selected].is_empty():
        download_dataframe(results[selected], selected.lower().replace(" ", "_"))


def risk_page(df: Dataset, mapping, entity_type: str = "all"):
    st.header("6. Risk Scoring Engine")
    is_legal = (entity_type == "legal")
    state_key = "rules_legal" if is_legal else "rules_physical"

    if state_key not in st.session_state:
        default_rules = DEFAULT_RULES_LEGAL if is_legal else DEFAULT_RULES
        st.session_state[state_key] = json.loads(json.dumps(default_rules))

    rules = st.session_state[state_key]

    with st.expander("Rule configuration", expanded=True):
        col_res, _ = st.columns([1, 3])

        for name, rule in rules.items():
            c1, c2, c3 = st.columns([2, 1, 1])
            rule["enabled"] = c1.checkbox(
                name,
                value=rule.get("enabled", True),
                key=f"rule_{state_key}_enabled_{name}"
            )
            rule["weight"] = c2.number_input(
                "Weight",
                value=float(rule.get("weight", 0)),
                key=f"rule_{state_key}_weight_{name}"
            )
            if rule.get("threshold") is not None:
                rule["threshold"] = c3.number_input(
                    "Threshold",
                    value=float(rule.get("threshold", 0)),
                    key=f"rule_{state_key}_threshold_{name}"
                )

    scored = RiskEngine(rules).score(df, mapping)
    show_table(scored.head(1000))
    st.plotly_chart(
        px.histogram(scored.head(200_000), x="risk_score", color="risk_level", title="Risk score distribution"),
        width='stretch')
    download_dataframe(scored, f"risk_scoring_{state_key}")


def graph_page(df: Dataset, mapping, lang="am"):
    st.header("7. Գրաֆային Վերլուծություն (Graph Network)" if lang == "am" else "7. Graph Analysis")
    c1, c2 = st.columns(2)
    src_key = c1.selectbox("Աղբյուր (Source)", ["debit_account", "debit_name", "customer_id"], index=0)
    dst_key = c2.selectbox("Նպատակ (Target)", ["credit_account", "credit_name", "customer_id"], index=0)

    try:
        analyzer = TransactionGraphAnalyzer(df, mapping)
        G = analyzer.build_graph(src_key, dst_key, directed=True)
        st.json(analyzer.metrics(G))
        st.plotly_chart(network_figure(G), width='stretch')
        st.header("Cycles")
        show_table(analyzer.cycles_table(G))
    except Exception as exc:
        st.error(f"Գրաֆի կառուցման սխալ: {exc}")



def filters_pivot_page(df: Dataset):
    st.header("9. Filters, Duplicates & Pivot Tables")
    filtered = df.lazy()
    with st.expander("Advanced filters", expanded=True):
        col = st.selectbox("Filter column", ["<None>"] + list(df.columns))
        if col != "<None>":
            mode = st.selectbox("Filter mode", ["contains", "starts with", "ends with", "regex", "null", "not null","doesn't contain"])
            val = st.text_input("Filter value")
            s = pl.col(col).cast(pl.String).fill_null("nan")
            if st.button("Apply filter"):
                if mode == "contains": filtered = filtered.filter(s.str.contains(val, case_insensitive=True))
                elif mode == "starts with": filtered = filtered.filter(s.str.starts_with(val))
                elif mode == "ends with": filtered = filtered.filter(s.str.ends_with(val))
                elif mode == "regex": filtered = filtered.filter(s.str.contains(val))
                elif mode == "null": filtered = filtered.filter(pl.col(col).is_null())
                elif mode == "not null": filtered = filtered.filter(pl.col(col).is_not_null())
                elif mode == "doesn't contain" : filtered = filtered.filter(~s.str.contains(val, case_insensitive=True))
    show_table(filtered.head(1000).collect())
    with st.expander("Duplicate detection"):
        dup_cols = st.multiselect("Duplicate columns", df.columns, key="dup_cols")
        if dup_cols:
            dups = (
                df.lazy()
                .filter(pl.struct(dup_cols).is_duplicated())
                .sort(dup_cols)
                .collect()
            )
            show_table(dups)
    with st.expander("Pivot builder"):
        rows = st.multiselect("Rows", df.columns, key="pivot_rows")
        cols = st.multiselect("Columns", df.columns, key="pivot_cols")
        vals = st.selectbox("Values", ["<None>"] + list(df.columns))
        agg = st.selectbox("Aggregation", ["count", "sum", "mean", "median", "min", "max", "nunique"])
        if rows and vals != "<None>":
            try:
                needed = list(dict.fromkeys(rows + cols + [vals]))
                data = df.lazy().select(needed).collect().to_pandas()
                pivot = pd.pivot_table(
                    data, index=rows, columns=cols or None, values=vals,
                    aggfunc=agg, fill_value=0,
                )
                show_table(pivot.reset_index())
            except Exception as exc:
                st.error(exc)


def export_page(df: Dataset):
    st.header("11. Export")
    download_dataframe(df, "cleaned_transactions")
    download_csv(df, "cleaned_transactions")


@st.cache_data(show_spinner=False, max_entries=16)
def _split_by_entity(ds: Dataset, mapping: Dict[str, str]):
    type_col = mapping.get("doc_type")
    debit_col = mapping.get("debit_account")
    credit_col = mapping.get("credit_account")

    if (
        not type_col or type_col not in ds.columns
        or not debit_col or debit_col not in ds.columns
        or not credit_col or credit_col not in ds.columns
    ):
        return ds, ds, ds

    doc_type = pl.col(type_col).cast(pl.String).fill_null("nan").str.strip_chars()
    debit_str = pl.col(debit_col).cast(pl.String).fill_null("nan").str.strip_chars()
    credit_str = pl.col(credit_col).cast(pl.String).fill_null("nan").str.strip_chars()

    is_outgoing = doc_type.eq("Բանկային հաշվից ելք")
    is_replenishment = doc_type.eq("Բանկային հաշվի համալրում")

    is_legal = (
        (is_outgoing & pl.any_horizontal([debit_str.str.starts_with(p) for p in LEGAL_ENTITY_PREFIXES]))
        | (is_replenishment & pl.any_horizontal([credit_str.str.starts_with(p) for p in LEGAL_ENTITY_PREFIXES]))
    )

    phys_sig = digest(ds.signature, "split", "phys", mapping)
    legal_sig = digest(ds.signature, "split", "legal", mapping)
    df_legal = make_dataset(ds.lazy().filter(is_legal).collect(), legal_sig, "legal")
    df_phys = make_dataset(ds.lazy().filter(~is_legal).collect(), phys_sig, "phys")

    return ds, df_phys, df_legal


def split_by_entity_type(df: Dataset, mapping: Dict[str, str]):
    return _split_by_entity(df, mapping)


def show_choice(df_all, df_phys, df_legal):
    entity_choice = st.radio(
        "Active Dataset for Analysis:",
        [
            f"👤 Ֆիզիկական անձինք ({len(df_phys):,} գործարք)",
            f"🏢 Իրավաբանական անձինք ({len(df_legal):,} գործարք)",
            f"🌐 Բոլոր գործարքները ({len(df_all):,} գործարք)",
        ],
        index=0,
        horizontal=True,
    )

    if "Ֆիզիկական" in entity_choice:
        active_df = df_phys
        entity_type = "phys"
        st.info(f"Ցուցադրվում են **ֆիզիկական անձանց** գործարքները ({len(df_phys):,} գործարք):")
    elif "Իրավաբանական" in entity_choice:
        active_df = df_legal
        entity_type = "legal"
        st.info(f"Ցուցադրվում են **իրավաբանական անձանց** գործարքները ({len(df_legal):,} գործարք):")
    else:
        active_df = df_all
        entity_type = "all"
        st.info(f"Ցուցադրվում են **բոլոր** գործարքները ({len(df_all):,} գործարք):")
    return active_df, entity_type


def main():
    init_state()
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Configurable AML, fraud detection, customer behavior, network analysis and risk scoring for banking transactions.")
    active_df = st.session_state.get("df")
    if active_df is not None and not isinstance(active_df, Dataset):
        st.session_state["df"] = None
        active_df = None
    page = st.sidebar.radio("Navigation", ["Load Data", "Cleaning", "Dashboard", "Statistics", "AML", "Risk Scoring", "Graph Analysis","Money Flow", "Customer/Account", "Filters/Pivot", "Export"])
    if page == "Load Data" or active_df is None:
        load_page()
        return
    mapping = sidebar_mapping(active_df)

    df_all, df_phys, df_legal = split_by_entity_type(active_df, mapping)

    st.markdown("---")

    if page == "Cleaning": cleaning_page(active_df)
    elif page == "Dashboard": dashboard_page(active_df, mapping)
    elif page == "AML": active_df,entity_type = show_choice(df_all, df_phys, df_legal); aml_page(active_df, mapping)
    elif page == "Risk Scoring": active_df,entity_type = show_choice(df_all, df_phys, df_legal); risk_page(active_df, mapping,entity_type)
    elif page == "Graph Analysis": active_df,entity_type = show_choice(df_all, df_phys, df_legal); graph_page(active_df, mapping)
    elif page == "Filters/Pivot": active_df,entity_type= show_choice(df_all, df_phys, df_legal); filters_pivot_page(active_df)
    elif page == "Export": export_page(active_df)
 


if __name__ == "__main__":
    main()
