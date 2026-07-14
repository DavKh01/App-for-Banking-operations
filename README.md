
# Banking AML, Fraud & Risk Analytics Streamlit App

A professional, modular Streamlit application for AML analysis, fraud detection, transaction network analysis, customer/account analytics and configurable risk scoring.

## Features

- Load Excel, CSV and TSV files with delimiter, encoding, decimal/thousands separator, sheet, header, skip rows, row limit and text-loading options.
- Interactive column mapping: the app does **not** assume fixed banking column names.
- Data profiling: preview, dataframe info, memory usage and missing values.
- Cleaning: duplicates, missing values, trimming, case conversion, replacement, renaming, datatype conversion and currency normalization hooks.
- Dashboard metrics: transactions, customers, accounts, turnover, incoming/outgoing, average, median, min/max, standard deviation, variance and date range.
- Statistical analysis: distributions, boxplots, time series, correlation heatmaps and summary statistics.
- AML indicators: large transactions, round amounts, repeated documents, high-risk customers/branches, one-to-many, many-to-one, counterparty changes, weekend/night activity, cross-currency activity and repeated notes.
- Configurable risk scoring engine with enable/disable rules, thresholds, weights, 0–100 scores, risk levels and explanations.
- NetworkX graph analytics: directed/undirected graphs, multigraphs, centrality, PageRank, components, cycles and Plotly graph visualization.
- Customer and account analytics.
- Duplicate detection, filters and pivot-table builder.
- Optional ML: Isolation Forest, KMeans, DBSCAN and PCA.
- Export to CSV, Excel, JSON, Parquet, HTML, PDF and PowerPoint.

## Project Structure

```text
aml_streamlit_app/
├── app.py
├── config.py
├── data_loader.py
├── preprocessing.py
├── analysis.py
├── risk_engine.py
├── graph_analysis.py
├── visualization.py
├── utils.py
├── requirements.txt
├── README.md
├── sample_config.json
├── ml_analysis.py
└── example_transactions.csv
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Usage Guide

1. Open **Load Data** and upload an Excel/CSV/TSV file.
2. Configure loading parameters and click **Load dataset**.
3. Navigate to any analysis page.
4. In the sidebar, map your real dataset columns to canonical concepts such as Transaction Date, Amount, Debit Account, Credit Account, Customer ID and Document Number.
5. Run AML indicators, graph analysis or risk scoring.
6. Export results from the relevant page.

## Risk Engine Notes

Default rules are defined in `config.py`. Each rule has:

- `enabled`
- `weight`
- `threshold`
- `description`

The score is capped at 100. The output contains `risk_score`, `risk_level` and `risk_reasons`.

## Performance Tips

- Use `nrows` for initial exploration.
- Use text loading when source formats are inconsistent.
- Convert only required columns to numeric/date types.
- For very large files, pre-filter or chunk upstream if necessary.
- The code is structured so Polars/lazy loading can be added in `data_loader.py` for production-scale deployments.

## Example Data

The included `example_transactions.csv` is synthetic and intended only for smoke testing.
