"""
CHRONO-Fair Monitoring Console.

A Streamlit dashboard for time-resolved counterfactual fairness monitoring
of clinical machine-learning models. The console reads a patient stream
(synthetic, calibrated to Texas-100X, or a user-supplied CSV) and exposes
the four CHRONO-Fair estimators plus an Inspector report.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Research prototype. Not a medical device. Streaming examples are controlled
replay or simulation unless connected to a validated deployment stream.
"""
from __future__ import annotations
import os
import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.flip_hazard import (kaplan_meier_curve, logrank_two_groups,
                                       restricted_mean_flip_time)
from chrono_fair.e_process import EProcessMonitor, IntersectionalMonitor
from chrono_fair.decomposition import ensemble_decompose, aggregate_by_group
from chrono_fair.rcap import rank_positions, rcap_w1_ci
from chrono_fair.inspector_agent import build_report, LLMNarrator

# ----------------------------------------------------------------------------
# Page configuration and dark clinical theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="CHRONO-Fair Monitoring Console",
                    page_icon="aplus", layout="wide",
                    initial_sidebar_state="expanded")

_CSS = """
<style>
  .stApp { background-color: #0e1117; color: #e6e6e6; }
  .kpi-card {
    background: #1a1f2e; border: 1px solid #2a3142; border-radius: 10px;
    padding: 14px 16px; margin: 4px;
  }
  .kpi-label { font-size: 0.78rem; color: #8b94a8; text-transform: uppercase;
               letter-spacing: 0.05em; }
  .kpi-value { font-size: 1.7rem; font-weight: 700; color: #f0f3f8; }
  .risk-green  { color: #2ecc71; } .risk-amber { color: #f39c12; }
  .risk-red    { color: #e74c3c; }
  .disclaimer  { background: #2a1f1f; border-left: 4px solid #e74c3c;
                 padding: 8px 14px; border-radius: 4px; font-size: 0.82rem;
                 color: #d8b8b8; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#0e1117",
                    plot_bgcolor="#161b26",
                    font=dict(color="#e6e6e6"), margin=dict(t=40, l=40, r=20, b=40))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def kpi(col, label, value, risk="green"):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value risk-{risk}">{value}</div></div>',
        unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_stream(n, drift_at, drift_mag, aleatoric, seed):
    cfg = StreamConfig(n=n, drift_at=drift_at, drift_magnitude=drift_mag,
                        aleatoric_bias=aleatoric, seed=seed)
    return generate_stream(cfg)


def synthetic_predictions(df, seed=0):
    """Risk-scored predictions and counterfactual predictions on the stream."""
    rng = np.random.default_rng(seed)
    risk = (df[['x0', 'x1', 'x2']].to_numpy().sum(axis=1)
             + (df['race'].isin(['Black', 'Hispanic']).to_numpy()) * 0.6)
    score = risk + 0.4 * rng.standard_normal(len(df))
    thr = np.median(score)
    y_hat = (score > thr).astype(int)
    risk_cf = risk - 0.6 * df['race'].isin(['Black', 'Hispanic']).to_numpy()
    y_hat_cf = (risk_cf + 0.4 * rng.standard_normal(len(df)) > thr).astype(int)
    los = 4.0 + risk + 0.3 * rng.standard_normal(len(df))
    return y_hat, y_hat_cf, los


# ----------------------------------------------------------------------------
# Sidebar: configuration
# ----------------------------------------------------------------------------
st.sidebar.title("CHRONO-Fair")
st.sidebar.caption("Monitoring console configuration")

alpha = st.sidebar.slider("Anytime-valid alpha", 0.01, 0.20, 0.05, 0.01)
rho0 = st.sidebar.slider("Baseline flip rate rho_0", 0.01, 0.20, 0.05, 0.01)
fdr_q = st.sidebar.slider("Step-wise BH level q", 0.01, 0.25, 0.10, 0.01)
n_stream = st.sidebar.select_slider("Stream length",
                                      [3000, 6000, 12000, 20000], 12000)
drift_on = st.sidebar.checkbox("Inject drift", True)
drift_at = st.sidebar.slider("Drift onset index", 1000, n_stream - 1000,
                               min(6000, n_stream - 1000), 500) if drift_on else None
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

uploaded = st.sidebar.file_uploader("Or load a prediction CSV", type="csv")

st.sidebar.markdown(
    '<div class="disclaimer">Research prototype. Not a medical device. '
    'Streaming examples are controlled replay or simulation unless connected '
    'to a validated deployment stream.</div>', unsafe_allow_html=True)

# Build the stream
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {len(df)} rows from CSV")
else:
    df = load_stream(n_stream, drift_at if drift_on else None,
                      0.6, 0.05, seed)
y_hat, y_hat_cf, los = synthetic_predictions(df, seed=seed)
df = df.copy()
df['y_hat'] = y_hat
df['y_hat_cf'] = y_hat_cf
df['y_los_hat'] = los
df['flip'] = (df['y_hat'] != df['y_hat_cf']).astype(int)

RACES = [r for r in ['White', 'Black', 'Hispanic', 'Asian/PI', 'Other']
          if r in set(df['race'])]

st.title("CHRONO-Fair Monitoring Console")
st.caption("Time-resolved counterfactual fairness monitoring for clinical "
            "machine learning")

tabs = st.tabs(["Overview", "Live Stream", "Flip Hazard", "Anytime E-Process",
                 "RCAP Regression", "Decomposition", "Texas-100X Verdicts",
                 "Robustness", "Inspector Report", "Export"])

# ----------------------------------------------------------------------------
# Tab 1: Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Monitoring overview")
    overall_flip = df['flip'].mean()
    # quick e-process pass per race
    cell_e, flagged = {}, []
    for r in RACES:
        sub = df[df['race'] == r]
        m = EProcessMonitor(rho0=rho0, alpha=alpha)
        for z in sub['flip']:
            m.update(int(z))
        cell_e[r] = float(np.exp(min(m.log_E, 700)))
        if m.alarm_at is not None:
            flagged.append(r)
    top_e = max(cell_e.values()) if cell_e else 1.0
    top_e_race = max(cell_e, key=cell_e.get) if cell_e else "none"

    c = st.columns(4)
    kpi(c[0], "Monitored patients", f"{len(df):,}")
    kpi(c[1], "Protected groups", f"{len(RACES)}")
    kpi(c[2], "Flagged cells", f"{len(flagged)}",
        "red" if flagged else "green")
    kpi(c[3], "Highest e-value", f"{top_e:.1f} ({top_e_race})",
        "red" if top_e > 1 / alpha else "green")
    c = st.columns(4)
    kpi(c[0], "Current flip rate", f"{overall_flip:.1%}",
        "amber" if overall_flip > rho0 else "green")
    kpi(c[1], "Baseline rho_0", f"{rho0:.3f}")
    # highest RCAP group
    ref = df.loc[df['race'] == RACES[0], 'y_los_hat'].to_numpy()
    rcap_by = {}
    for r in RACES:
        sub = df[df['race'] == r]
        u_a = rank_positions(sub['y_los_hat'].to_numpy(),
                              sub['y_los_hat'].to_numpy())
        u_ap = rank_positions(sub['y_los_hat'].to_numpy()
                               - 0.6 * sub['race'].isin(['Black', 'Hispanic']).to_numpy(),
                               ref)
        w1, _, _ = rcap_w1_ci(u_a, u_ap, n_boot=200, seed=0)
        rcap_by[r] = w1
    top_rcap = max(rcap_by, key=rcap_by.get) if rcap_by else "none"
    kpi(c[2], "Highest RCAP group", f"{top_rcap}")
    status = "ALARM" if flagged else "NOMINAL"
    kpi(c[3], "Monitoring status", status, "red" if flagged else "green")

    st.markdown("---")
    fr = (df.assign(bucket=(df['patient_id'] // 500))
            .groupby('bucket')['flip'].mean().reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr['bucket'] * 500, y=fr['flip'],
                              mode='lines+markers', line=dict(color='#3498db')))
    fig.add_hline(y=rho0, line_dash="dash", line_color="#e74c3c",
                   annotation_text="baseline rho_0")
    fig.update_layout(title="Flip rate over arrival order (500-patient buckets)",
                       xaxis_title="Patient index", yaxis_title="Flip rate",
                       **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2: Live Stream Monitor
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Live stream monitor")
    st.caption("Replays the stream patient by patient. Controlled replay, "
                "not a live clinical feed.")
    cc = st.columns(3)
    if cc[0].button("Start / step 500"):
        st.session_state['cursor'] = st.session_state.get('cursor', 0) + 500
    if cc[1].button("Reset"):
        st.session_state['cursor'] = 0
    cursor = min(st.session_state.get('cursor', 1000), len(df))
    cc[2].metric("Current patient index", f"{cursor:,}")

    seen = df.iloc[:cursor]
    if len(seen) > 0:
        m = EProcessMonitor(rho0=rho0, alpha=alpha)
        trace = []
        for z in seen['flip']:
            r = m.update(int(z))
            trace.append(r['log_E'])
        k = st.columns(3)
        kpi(k[0], "Flip rate so far", f"{seen['flip'].mean():.1%}")
        kpi(k[1], "log E_t", f"{m.log_E:.2f}")
        kpi(k[2], "Alarm", "RAISED" if m.alarm_at else "none",
            "red" if m.alarm_at else "green")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=trace, mode='lines',
                                  line=dict(color='#f39c12')))
        fig.add_hline(y=np.log(1 / alpha), line_dash="dash",
                       line_color="#e74c3c",
                       annotation_text="log(1/alpha) alarm boundary")
        fig.update_layout(title="E-process trajectory (replay so far)",
                           xaxis_title="Patient", yaxis_title="log E_t",
                           **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3: Flip Hazard
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Flip Hazard survival")
    fig = go.Figure()
    rmft_rows = []
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for r, col in zip(RACES, palette):
        sub = df[df['race'] == r].reset_index(drop=True)
        if len(sub) < 30:
            continue
        km = kaplan_meier_curve(np.arange(len(sub), dtype=float),
                                 sub['flip'].values)
        fig.add_trace(go.Scatter(x=km['t'], y=km['S'], mode='lines',
                                  name=f"{r} (n={len(sub)})",
                                  line=dict(color=col)))
        rmft_rows.append({'group': r, 'n': len(sub),
                           'RMFT_5000': round(restricted_mean_flip_time(km, 5000), 1),
                           'final_survival': round(float(km['S'].iloc[-1]), 4)})
    fig.update_layout(title="No-flip survival by protected group",
                       xaxis_title="Patient index t",
                       yaxis_title="P(no flip up to t)", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Restricted Mean Flip Time (tau\\* = 5000)**")
    st.dataframe(pd.DataFrame(rmft_rows), use_container_width=True)
    # log-rank vs first group
    lr_rows = []
    ref = df[df['race'] == RACES[0]].reset_index(drop=True)
    et_r = np.arange(len(ref), dtype=float)
    for r in RACES[1:]:
        sub = df[df['race'] == r].reset_index(drop=True)
        lr = logrank_two_groups(et_r, ref['flip'].values,
                                 np.arange(len(sub), dtype=float),
                                 sub['flip'].values)
        lr_rows.append({'reference': RACES[0], 'group': r,
                         'chi2': round(lr['chi2'], 2),
                         'pvalue': lr['pvalue']})
    st.markdown("**Log-rank tests**")
    st.dataframe(pd.DataFrame(lr_rows), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 4: Anytime E-Process
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Anytime-valid e-process")
    st.info("The step-wise Benjamini-Hochberg layer is an inspection-time "
             "adjustment across cells, not a full time-uniform online "
             "false-discovery guarantee.")
    fig = go.Figure()
    e_table = []
    for r, col in zip(RACES, palette):
        sub = df[df['race'] == r].reset_index(drop=True)
        m = EProcessMonitor(rho0=rho0, alpha=alpha)
        tr = []
        for z in sub['flip']:
            m.update(int(z))
            tr.append(m.log_E)
        fig.add_trace(go.Scatter(y=tr, mode='lines', name=r,
                                  line=dict(color=col)))
        e_table.append({'cell': r, 'n': len(sub),
                         'e_value': round(float(np.exp(min(m.log_E, 700))), 2),
                         'alarm_at': m.alarm_at})
    fig.add_hline(y=np.log(1 / alpha), line_dash="dash", line_color="#e74c3c",
                   annotation_text="log(1/alpha)")
    fig.update_layout(title="log E_t per cell", xaxis_title="Patient",
                       yaxis_title="log E_t", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    et = pd.DataFrame(e_table)
    p_vals = np.clip(1.0 / np.maximum(et['e_value'].values, 1e-9), 0, 1)
    order = np.argsort(p_vals)
    m = len(p_vals)
    thr = fdr_q * np.arange(1, m + 1) / m
    below = np.where(p_vals[order] <= thr)[0]
    kk = below.max() + 1 if len(below) else 0
    et['bh_flagged'] = False
    if kk:
        et.loc[order[:kk], 'bh_flagged'] = True
    st.markdown("**Per-cell e-values and inspection-time BH flags**")
    st.dataframe(et, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 5: RCAP
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("RCAP regression fairness")
    st.info("RCAP is the Wasserstein-1 distance between factual and "
             "counterfactual predicted-LOS rank-position distributions. It "
             "measures allocation rank shift, not raw regression error.")
    ref = df.loc[df['race'] == RACES[0], 'y_los_hat'].to_numpy()
    rows, fig = [], go.Figure()
    for r, col in zip(RACES, palette):
        sub = df[df['race'] == r]
        if len(sub) < 30:
            continue
        u_a = rank_positions(sub['y_los_hat'].to_numpy(),
                              sub['y_los_hat'].to_numpy())
        u_ap = rank_positions(
            sub['y_los_hat'].to_numpy()
            - 0.6 * sub['race'].isin(['Black', 'Hispanic']).to_numpy(), ref)
        w1, lo, hi = rcap_w1_ci(u_a, u_ap, n_boot=400, seed=1)
        rows.append({'group': r, 'n': len(sub),
                      'RCAP_W1': round(w1, 4),
                      'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})
        fig.add_trace(go.Histogram(x=(u_a - u_ap), name=r, opacity=0.6,
                                    marker_color=col, nbinsx=40))
    fig.update_layout(title="Per-patient rank shift by group",
                       barmode='overlay', xaxis_title="rank shift",
                       **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 6: Decomposition
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Aleatoric / epistemic decomposition")
    rng = np.random.default_rng(seed)
    K = 11
    risk = (df[['x0', 'x1', 'x2']].to_numpy().sum(axis=1)
             + df['race'].isin(['Black', 'Hispanic']).to_numpy() * 0.6)
    noise = 0.4 * rng.standard_normal((K, len(df)))
    pa = 1 / (1 + np.exp(-(risk[None, :] + noise)))
    pc = 1 / (1 + np.exp(-(risk[None, :]
              - 0.6 * df['race'].isin(['Black', 'Hispanic']).to_numpy()
              + noise)))
    dec = ensemble_decompose(pa, pc)
    agg = aggregate_by_group(dec, df['race'])
    agg = agg[agg['group'].isin(RACES)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg['group'], y=agg['aleatoric_flip'],
                          name='Aleatoric', marker_color='#3498db'))
    fig.add_trace(go.Bar(x=agg['group'], y=agg['epistemic_flip'],
                          name='Epistemic', marker_color='#f39c12'))
    fig.update_layout(title="Flip mass split by cause", barmode='stack',
                       yaxis_title="Flip mass", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agg[['group', 'flip_rate', 'aleatoric_share',
                       'epistemic_share']].round(3), use_container_width=True)
    st.caption("Aleatoric-dominated: investigate the label and feature "
                "pipeline. Epistemic-dominated: collect more samples in the "
                "stratum. Mixed: run both in parallel.")

# ----------------------------------------------------------------------------
# Tab 7: Texas-100X verdict stability
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Texas-100X verdict stability (real audit data)")
    art = os.path.join(os.path.dirname(__file__), 'data',
                        'real_texas100x_audit_artifacts.json')
    if os.path.exists(art):
        with open(art) as fh:
            audit = json.load(fh)
        fluct = audit.get('fluctuation', {})
        rows = []
        for attr in fluct:
            for met in ['DI', 'WTPR', 'PPV_Ratio']:
                if met not in fluct[attr]:
                    continue
                arr = np.asarray(fluct[attr][met], float)
                point = float(arr.mean())
                verds = ['fair' if v >= 0.8 else 'unfair' for v in arr]
                vfr = min(verds.count('fair'), verds.count('unfair')) / len(arr)
                rows.append({'attribute': attr, 'metric': met,
                              'point': round(point, 4),
                              'point_verdict': 'fair' if point >= 0.8 else 'unfair',
                              'VFR': round(vfr, 3)})
        vdf = pd.DataFrame(rows)
        st.dataframe(vdf, use_container_width=True)
        unstable = vdf[vdf['VFR'] > 0]
        if len(unstable):
            st.error(f"Unstable verdict(s): "
                      + ", ".join(f"{r.attribute} {r.metric} (VFR={r.VFR})"
                                   for r in unstable.itertuples()))
        else:
            st.success("All audited verdicts stable on this artefact.")
    else:
        st.warning("Real audit artefact not found. Run "
                    "`python -m chrono_fair.data.build_dataset` first.")

# ----------------------------------------------------------------------------
# Tab 8: Robustness
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Robustness stress tests")
    st.caption("Pre-computed summary from exp9_robustness. Underestimating "
                "rho_0 inflates false alarms; a cell needs roughly 1000 "
                "patients for a detection rate above 0.95.")
    fig = os.path.join(os.path.dirname(__file__), '..',
                        'chrono_fair_figures', 'fig10_robustness.png')
    if os.path.exists(fig):
        st.image(fig, use_container_width=True)
    else:
        st.info("Run `python -m chrono_fair.experiments.exp9_robustness` "
                 "to generate the robustness figure.")

# ----------------------------------------------------------------------------
# Tab 9: Inspector report
# ----------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Inspector report")
    narrator = LLMNarrator()
    any_flag = False
    for r in RACES:
        sub = df[df['race'] == r]
        m = EProcessMonitor(rho0=rho0, alpha=alpha)
        for z in sub['flip']:
            m.update(int(z))
        if m.alarm_at is None:
            continue
        any_flag = True
        agg_row = pd.Series({'flip_rate': sub['flip'].mean(),
                              'epistemic_share': 0.5, 'aleatoric_share': 0.5,
                              'n': len(sub)})
        rep = build_report(r, {'n': len(sub),
                                 'E': float(np.exp(min(m.log_E, 700)))},
                            agg_row, baseline_rate=rho0)
        with st.expander(f"Flagged cell: {r}", expanded=True):
            st.code(narrator.narrate(rep), language=None)
    if not any_flag:
        st.success("No cell currently flagged. Inspector report is empty.")

# ----------------------------------------------------------------------------
# Tab 10: Export
# ----------------------------------------------------------------------------
with tabs[9]:
    st.subheader("Export monitoring report")
    summary = (df.groupby('race')['flip']
                 .agg(['mean', 'size']).reset_index()
                 .rename(columns={'mean': 'flip_rate', 'size': 'n'}))
    st.dataframe(summary, use_container_width=True)
    csv = summary.to_csv(index=False).encode()
    st.download_button("Download cell summary CSV", csv,
                        "chrono_fair_cells.csv", "text/csv")
    md = io.StringIO()
    md.write("# CHRONO-Fair monitoring report\n\n")
    md.write(f"- Patients monitored: {len(df):,}\n")
    md.write(f"- Baseline rho_0: {rho0}\n- Alpha: {alpha}\n\n")
    md.write(summary.to_markdown(index=False))
    md.write("\n\nResearch prototype. Not a medical device.\n")
    st.download_button("Download Markdown report", md.getvalue(),
                        "chrono_fair_report.md", "text/markdown")
