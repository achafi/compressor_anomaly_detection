"""Streamlit app for MetroPT-3 air-compressor anomaly detection.

Ports the pipeline from notebooks/metropt3_tp2_regression.ipynb:
5-minute aggregation -> LightGBM regression on TP2 -> residual-based
anomaly scoring with a calibrated threshold.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMRegressor
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "MetroPT3(AirCompressor).csv"
ZIP_PATH = RAW_DIR / "metropt3_uci.zip"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"

TARGET = "TP2"
FEATURES = [
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
]
STATE_COLS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

COLUMN_DOCS = {
    "timestamp": ("datetime", "Recording time (10 s nominal sampling)"),
    "TP2": ("bar", "Pressure at the compressor — model target"),
    "TP3": ("bar", "Pressure at the pneumatic panel"),
    "H1": ("bar", "Pressure at cyclonic separator filter discharge"),
    "DV_pressure": ("bar", "Pressure drop at air-dryer tower discharge (0 ⇒ under load)"),
    "Reservoirs": ("bar", "Downstream reservoir pressure"),
    "Oil_temperature": ("°C", "Compressor oil temperature"),
    "Motor_current": ("A", "Motor phase current (≈0 off, ≈4 offloaded, ≈7 loaded)"),
    "COMP": ("0/1", "Air-intake valve signal (active ⇒ off/offloaded)"),
    "DV_eletric": ("0/1", "Outlet valve signal (active ⇒ under load)"),
    "Towers": ("0/1", "Tower selector (0 ⇒ tower 1, 1 ⇒ tower 2)"),
    "MPG": ("0/1", "Start-under-load signal (<8.2 bar), follows COMP"),
    "LPS": ("0/1", "Low-pressure signal (<7 bar)"),
    "Pressure_switch": ("0/1", "Tower discharge indication"),
    "Oil_level": ("0/1", "Low oil level signal"),
    "Caudal_impulses": ("0/1", "Air-flow pulse count, APU→reservoirs"),
}

PLOT_TEMPLATE = "plotly_white"


def ensure_dataset() -> Path:
    """Download the UCI dataset and extract the CSV if missing."""
    if CSV_PATH.exists():
        return CSV_PATH
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        urllib.request.urlretrieve(UCI_ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(member) as src, open(CSV_PATH, "wb") as dst:
            dst.write(src.read())
    return CSV_PATH


@st.cache_data(show_spinner=False)


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    timestamp_col = next(c for c in df.columns if c.lower() == "timestamp")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)


def aggregate(df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
    """Resample to fixed bins; state signals are binarized at >= 0.5."""
    timestamp_col = next(c for c in df.columns if c.lower() == "timestamp")
    signal_cols = [c for c in df.select_dtypes(include="number").columns if c != "Unnamed: 0"]
    binned = df.set_index(timestamp_col)[signal_cols].resample(resample_rule).mean()
    for col in STATE_COLS:
        if col in binned.columns:
            binned[col] = np.where(binned[col].notna(), (binned[col] >= 0.5).astype(float), np.nan)
    binned = binned.dropna(how="all").reset_index().sort_values(timestamp_col)
    return binned.reset_index(drop=True)


def split_periods(
    binned: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    use_normal_mode: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = binned.columns[0]
    normal_mode = (
        binned["COMP"].eq(1)
        & binned["DV_eletric"].eq(0)
        & binned["Motor_current"].between(3.0, 5.5)
    )
    train_mask = binned[ts].ge(train_start) & binned[ts].lt(train_end)
    test_mask = binned[ts].ge(train_end) & binned[ts].lt(test_end)
    if use_normal_mode:
        train_mask &= normal_mode
        test_mask &= normal_mode
    return binned.loc[train_mask].copy(), binned.loc[test_mask].copy()


@st.cache_data(show_spinner=False)


def train_and_score(
    binned: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    use_normal_mode: bool,
    threshold_quantile: float,
    model_params: dict,
    resample_rule: str = "5min",
) -> dict:
    ts = binned.columns[0]
    train_df, test_df = split_periods(binned, train_start, train_end, test_end, use_normal_mode)
    if len(train_df) < 100:
        raise ValueError(
            f"Training window has only {len(train_df)} bins — widen the window or relax the filters."
        )

    X_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET].copy()
    X_test = test_df[FEATURES].copy() if len(test_df) else pd.DataFrame(columns=FEATURES)
    y_test = test_df[TARGET].copy() if len(test_df) else pd.Series(dtype=float)

    cv = TimeSeriesSplit(n_splits=5)
    oof_pred = np.full(len(X_train), np.nan)
    oof_fold = np.zeros(len(X_train), dtype=int)
    fold_rows = []
    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train), start=1):
        X_fit, X_val = X_train.iloc[fit_idx].copy(), X_train.iloc[val_idx].copy()
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]
        medians = X_fit.median(numeric_only=True)
        fold_model = LGBMRegressor(**model_params)
        fold_model.fit(X_fit.fillna(medians), y_fit)
        pred = fold_model.predict(X_val.fillna(medians))
        oof_pred[val_idx] = pred
        oof_fold[val_idx] = fold
        fold_rows.append(
            {
                "Fold": fold,
                "Bins": len(val_idx),
                "MAE": mean_absolute_error(y_val, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
                "R²": r2_score(y_val, pred),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)

    covered = np.isfinite(oof_pred)
    oof_residual = y_train.to_numpy()[covered] - oof_pred[covered]
    oof_abs = np.abs(oof_residual)
    oof_results = pd.DataFrame(
        {
            "Timestamp": train_df[ts].to_numpy()[covered],
            "Actual TP2": y_train.to_numpy()[covered],
            "Predicted TP2": oof_pred[covered],
            "Residual": oof_residual,
            "Absolute residual": oof_abs,
            "Fold": oof_fold[covered],
        }
    )

    train_medians = X_train.median(numeric_only=True)
    model = LGBMRegressor(**model_params)
    model.fit(X_train.fillna(train_medians), y_train)
    train_pred = model.predict(X_train.fillna(train_medians))

    def _metrics(y, p) -> dict:
        if len(y) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "R²": np.nan}
        return {
            "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R²": r2_score(y, p),
        }

    test_pred = model.predict(X_test.fillna(train_medians)) if len(X_test) else np.array([])
    performance = pd.DataFrame(
        {
            "Train (in-sample)": _metrics(y_train, train_pred),
            "OOF (pooled CV)": _metrics(y_train.to_numpy()[covered], oof_pred[covered]),
            "Test": _metrics(y_test, test_pred),
        }
    ).T

    threshold = float(np.quantile(oof_abs, threshold_quantile))
    threshold_alt = float(oof_abs.mean() + 3 * oof_abs.std(ddof=1))

    X_all = binned[FEATURES].fillna(train_medians)
    y_all_pred = model.predict(X_all)
    residual_all = binned[TARGET].to_numpy() - y_all_pred
    abs_residual = np.abs(residual_all)
    raw_score = pd.Series(abs_residual / threshold, index=binned.index)
    results = pd.DataFrame(
        {
            "Timestamp": binned[ts].to_numpy(),
            "Actual TP2": binned[TARGET].to_numpy(),
            "Predicted TP2": y_all_pred,
            "Residual": residual_all,
            "Absolute residual": abs_residual,
            "raw_anomaly_score": raw_score.to_numpy(),
            "anomaly_score": raw_score.clip(upper=1.0).to_numpy(),
            "is_anomaly": (raw_score >= 1).to_numpy(),
        }
    )

    gap = results["Timestamp"].diff().gt(pd.Timedelta(resample_rule))
    run_id = (results["is_anomaly"].ne(results["is_anomaly"].shift()) | gap).cumsum()
    runs = results.groupby(run_id, sort=False).agg(
        is_anomaly=("is_anomaly", "first"),
        start=("Timestamp", "first"),
        end=("Timestamp", "last"),
        bins=("Timestamp", "size"),
        max_score=("raw_anomaly_score", "max"),
    )
    anomaly_runs = runs.loc[runs["is_anomaly"]].drop(columns="is_anomaly").reset_index(drop=True)
    anomaly_runs.index += 1
    anomaly_runs.index.name = "run"

    feature_importance = pd.DataFrame(
        {"Feature": FEATURES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    return {
        "model": model,
        "train_medians": train_medians,
        "threshold": threshold,
        "threshold_alt": threshold_alt,
        "threshold_quantile": threshold_quantile,
        "fold_metrics": fold_metrics,
        "performance": performance,
        "oof_results": oof_results,
        "results": results,
        "anomaly_runs": anomaly_runs,
        "feature_importance": feature_importance,
        "residual_q": {
            "q95": float(np.quantile(oof_abs, 0.95)),
            "q97.5": float(np.quantile(oof_abs, 0.975)),
            "q99": float(np.quantile(oof_abs, 0.99)),
            "q99.5": float(np.quantile(oof_abs, 0.995)),
        },
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_span": (train_df[ts].min(), train_df[ts].max()),
        "test_span": (test_df[ts].min(), test_df[ts].max()) if len(test_df) else (None, None),
    }


def _score_timeline(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=results["Timestamp"],
            y=results["anomaly_score"],
            mode="lines",
            name="Anomaly score",
            line=dict(color="#1f77b4", width=1),
        )
    )
    flagged = results.loc[results["is_anomaly"]]
    fig.add_trace(
        go.Scattergl(
            x=flagged["Timestamp"],
            y=flagged["anomaly_score"],
            mode="markers",
            name="Above threshold",
            marker=dict(color="red", size=5),
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Threshold = 1.0")
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Anomaly score over time (clipped at 1.0)",
        height=400,
        yaxis_range=[0, 1.05],
        dragmode="pan",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def _actual_vs_pred_timeline(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=results["Timestamp"], y=results["Actual TP2"], mode="lines", name="Actual TP2")
    )
    fig.add_trace(
        go.Scattergl(
            x=results["Timestamp"], y=results["Predicted TP2"], mode="lines", name="Predicted TP2"
        )
    )
    flagged = results.loc[results["is_anomaly"]]
    fig.add_trace(
        go.Scattergl(
            x=flagged["Timestamp"],
            y=flagged["Actual TP2"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=6, symbol="x"),
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Actual vs predicted TP2 with flagged bins",
        height=400,
        dragmode="pan",
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="TP2 (bar)")
    return fig

st.set_page_config(page_title="Compressor Anomaly Detection", layout="wide")
st.title("MetroPT-3 Compressor · Anomaly Detection")
st.caption(
    "Residual-based anomaly detection on the UCI MetroPT+3 air-compressor dataset — "
    "5-minute aggregation, LightGBM regression of TP2, threshold calibrated on OOF residuals."
)

with st.sidebar:
    st.header("Configuration")
    try:
        csv_path = ensure_dataset()
        st.success(f"Dataset ready · {csv_path.name}")
    except Exception as exc:
        st.error(f"Dataset missing and download failed: {exc}")
        st.stop()

    resample_rule = st.selectbox("Aggregation window", ["1min", "5min", "10min", "15min"], index=1)

    st.subheader("Period split")
    d1, d2 = st.columns(2)
    train_start_date = d1.date_input("Train start", pd.Timestamp("2020-02-01"))
    train_end_date = d2.date_input("Train end", pd.Timestamp("2020-02-21"))
    test_end_date = st.date_input("Test end", pd.Timestamp("2020-03-01"))
    use_normal_mode = st.toggle(
        "Restrict to active-offloaded mode (COMP=1, DV_eletric=0, 3.0≤Motor_current≤5.5)",
        value=True,
    )

    st.subheader("Model")
    n_estimators = st.slider("n_estimators", 50, 1000, 300, 50)
    learning_rate = st.select_slider("learning_rate", [0.01, 0.02, 0.05, 0.1, 0.2], 0.05)
    num_leaves = st.select_slider("num_leaves", [15, 31, 63, 127], 31)
    threshold_quantile = st.slider("Threshold quantile (OOF |residual|)", 0.90, 0.999, 0.99, 0.001)

    model_params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }

    st.divider()
    st.caption(
        "Above-threshold scores are **review signals, not confirmed failures**. Scores outside "
        "the active-offloaded regime may reflect distribution shift rather than faults."
    )

with st.spinner("Loading raw telemetry (1.5M rows)…"):
    raw_df = load_raw(str(csv_path))
with st.spinner("Aggregating to bins…"):
    binned = aggregate(raw_df, resample_rule)

train_start = pd.Timestamp(train_start_date)
train_end = pd.Timestamp(train_end_date)
test_end = pd.Timestamp(test_end_date)
if not train_start < train_end < test_end:
    st.error("Require: train start < train end < test end.")
    st.stop()

try:
    with st.spinner("Training LightGBM + calibrating threshold…"):
        bundle = train_and_score(
            binned,
            train_start,
            train_end,
            test_end,
            use_normal_mode,
            float(threshold_quantile),
            model_params,
            resample_rule,
        )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

results: pd.DataFrame = bundle["results"]

tab_overview, tab_eda, tab_model, tab_anomaly = st.tabs(
    ["Overview", "Exploration", "Model", "Anomalies"]
)

with tab_overview:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Raw rows", f"{len(raw_df):,}")
    c2.metric(f"Bins ({resample_rule})", f"{len(binned):,}")
    c3.metric("Train bins", f"{bundle['n_train']:,}")
    c4.metric("Test bins", f"{bundle['n_test']:,}")
    c5.metric("Threshold", f"{bundle['threshold']:.4f} bar")
    c6.metric(
        "Flagged bins",
        f"{int(results['is_anomaly'].sum()):,} ({results['is_anomaly'].mean():.2%})",
    )

    st.subheader("Column dictionary")
    st.dataframe(
        pd.DataFrame(
            [(k, v[0], v[1]) for k, v in COLUMN_DOCS.items()],
            columns=["Column", "Unit / type", "Description"],
        ),
        hide_index=True,
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Aggregated sample")
        st.dataframe(binned.head(20), hide_index=True, width="stretch")
    with right:
        st.subheader("Data coverage")
        span = pd.DataFrame(
            {
                "Split": ["Raw", "Train", "Test"],
                "Start": [raw_df["timestamp"].min(), *bundle["train_span"][:1], *bundle["test_span"][:1]],
                "End": [raw_df["timestamp"].max(), *bundle["train_span"][1:], *bundle["test_span"][1:]],
            }
        )
        st.dataframe(span, hide_index=True, width="stretch")
        st.markdown(
            f"""
- **Target**: `{TARGET}` (compressor pressure, bar)
- **Features**: {", ".join(f"`{f}`" for f in FEATURES)}
- **CV**: chronological `TimeSeriesSplit(n_splits=5)`
- **Threshold**: {threshold_quantile:.3f} quantile of OOF |residual| = **{bundle['threshold']:.4f} bar**
  (alternative μ+3σ = {bundle['threshold_alt']:.4f} bar)
- **Score**: `clip(|residual| / threshold, upper=1)`; anomaly when raw score ≥ 1
"""
        )

with tab_eda:
    numeric_cols = binned.select_dtypes(include="number").columns.tolist()
    continuous_cols = [c for c in numeric_cols if c not in STATE_COLS]

    sel = st.multiselect("Signals to plot", continuous_cols, default=["TP2", "TP3", "Motor_current", "Oil_temperature"])
    if sel:
        fig = make_subplots(
            rows=len(sel), cols=1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=sel,
        )
        for i, col in enumerate(sel, start=1):
            fig.add_trace(go.Scattergl(x=binned[binned.columns[0]], y=binned[col], mode="lines", name=col), row=i, col=1)
            fig.update_yaxes(title_text=col, row=i, col=1)
        fig.update_layout(
            template=PLOT_TEMPLATE,
            height=max(350, 220 * len(sel)),
            showlegend=False,
            dragmode="pan",
        )
        st.plotly_chart(fig, config={"scrollZoom": True}, theme=None)

    st.subheader("Pearson correlation (continuous signals)")
    corr = binned[continuous_cols].corr(method="pearson")
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(template=PLOT_TEMPLATE, height=650)
    st.plotly_chart(fig, config={"scrollZoom": True}, theme=None)

with tab_model:
    st.subheader("Cross-validation folds (chronological)")
    st.dataframe(
        bundle["fold_metrics"].style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Performance")
    st.dataframe(
        bundle["performance"].style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        oof = bundle["oof_results"]
        fig = px.scatter(
            oof, x="Predicted TP2", y="Actual TP2", opacity=0.5,
            title="OOF: actual vs predicted TP2",
        )
        lo = min(oof["Actual TP2"].min(), oof["Predicted TP2"].min())
        hi = max(oof["Actual TP2"].max(), oof["Predicted TP2"].max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", line=dict(dash="dash", color="red")))
        fig.update_layout(template=PLOT_TEMPLATE, height=400)
        st.plotly_chart(fig, config={"scrollZoom": True}, theme=None)

    with right:
        fig = px.histogram(oof, x="Residual", nbins=60, title="OOF residual distribution")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(template=PLOT_TEMPLATE, height=400)
        st.plotly_chart(fig, config={"scrollZoom": True}, theme=None)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            oof, x="Predicted TP2", y="Residual", opacity=0.5,
            title="OOF residuals vs predicted",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(template=PLOT_TEMPLATE, height=400)
        st.plotly_chart(fig, config={"scrollZoom": True}, theme=None)

    with right:
        st.subheader("Feature importance (final model)")
        fi = bundle["feature_importance"]
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     category_orders={"Feature": fi["Feature"].tolist()})
        fig.update_layout(template=PLOT_TEMPLATE, height=400, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, config={"scrollZoom": True}, theme=None)

    st.subheader("Residual diagnostics (OOF)")
    rq = bundle["residual_q"]
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("q95", f"{rq['q95']:.4f}")
    d2.metric("q97.5", f"{rq['q97.5']:.4f}")
    d3.metric("q99 (threshold)", f"{rq['q99']:.4f}")
    d4.metric("q99.5", f"{rq['q99.5']:.4f}")
    d5.metric("μ + 3σ", f"{bundle['threshold_alt']:.4f}")

with tab_anomaly:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scored bins", f"{len(results):,}")
    m2.metric("Flagged", f"{int(results['is_anomaly'].sum()):,}")
    m3.metric("Anomaly runs", f"{len(bundle['anomaly_runs']):,}")
    m4.metric("Max raw score", f"{results['raw_anomaly_score'].max():.2f}")

    st.plotly_chart(_score_timeline(results), config={"scrollZoom": True}, theme=None)
    st.plotly_chart(_actual_vs_pred_timeline(results), config={"scrollZoom": True}, theme=None)

    left, right = st.columns(2)
    with left:
        st.subheader("Top 20 anomalies")
        top = results.nlargest(20, "raw_anomaly_score")[
            ["Timestamp", "Actual TP2", "Predicted TP2", "Residual",
             "Absolute residual", "raw_anomaly_score", "anomaly_score"]
        ]
        st.dataframe(
            top.style.format(
                {
                    "Actual TP2": "{:.3f}",
                    "Predicted TP2": "{:.3f}",
                    "Residual": "{:.3f}",
                    "Absolute residual": "{:.3f}",
                    "raw_anomaly_score": "{:.2f}",
                    "anomaly_score": "{:.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Anomaly runs")
        runs = bundle["anomaly_runs"]
        if len(runs):
            st.dataframe(
                runs.style.format({"max_score": "{:.2f}"}),
                hide_index=False,
                width="stretch",
            )
            st.caption(
                "Runs are contiguous flagged bins, broken when the time gap exceeds the "
                f"aggregation window ({resample_rule})."
            )
        else:
            st.info("No anomaly runs detected with the current threshold.")

    st.subheader("Score distribution")
    fig = px.histogram(
        results, x="raw_anomaly_score", nbins=80, log_y=True,
        title="Raw anomaly score distribution (log count)",
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="red")
    fig.update_layout(template=PLOT_TEMPLATE, height=380)
    st.plotly_chart(fig, config={"scrollZoom": True}, theme=None)

    st.download_button(
        "Download full results (CSV)",
        results.to_csv(index=False).encode(),
        file_name="anomaly_results.csv",
        mime="text/csv",
    )
