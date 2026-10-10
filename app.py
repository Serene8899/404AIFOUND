"""
Maintenance Triage Copilot — Streamlit dashboard.

    streamlit run app.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

import core

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent
MODELS = ROOT / "models"

st.set_page_config(page_title="Triage Copilot", page_icon="🛠️", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
h1 {font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0 !important;}
.sub {color: #6b7280; font-size: 0.98rem; margin-top: 0.15rem; margin-bottom: 1.1rem;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px;}
.kpi .lab {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #6b7280;}
.kpi .val {font-size: 1.75rem; font-weight: 800; line-height: 1.2;}
.kpi .hint {font-size: .8rem; color: #6b7280;}
.card {border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
       border: 1px solid rgba(128,128,128,.25);}
.card.red {border-left: 6px solid #dc2626;}
.card.amber {border-left: 6px solid #f59e0b;}
.card.green {border-left: 6px solid #16a34a;}
.card.blue {border-left: 6px solid #2563eb;}
.card h4 {margin: 0 0 6px 0; font-size: 1.15rem !important; line-height: 1.4;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem;
       font-weight: 700; color: white;}
</style>
""", unsafe_allow_html=True)

STATUS_ICON = {"Fix today": "🔴 Fix today", "Next shift": "🟠 Next shift",
               "Watch": "🟡 Watch", "Check sensor": "🔵 Check sensor", "Healthy": "🟢 Healthy"}
STATUS_COLOR = {"Fix today": "#dc2626", "Next shift": "#f97316", "Watch": "#eab308",
                "Check sensor": "#2563eb", "Healthy": "#16a34a"}
STATUS_CARD = {"Fix today": "red", "Next shift": "amber", "Watch": "amber",
               "Check sensor": "blue", "Healthy": "green"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    if not (MODELS / "bundle.joblib").exists():
        return None
    b = joblib.load(MODELS / "bundle.joblib")
    b["explainer"] = shap.TreeExplainer(b["detector"])
    return b


@st.cache_data
def load_reports():
    return (pd.read_csv(MODELS / "test_set.csv"),
            pd.read_csv(MODELS / "val_predictions.csv"),
            pd.read_csv(MODELS / "robustness.csv"),
            json.load(open(MODELS / "metrics.json")))


bundle = load_bundle()
if bundle is None:
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()
test_set, val_preds, robustness, meta = load_reports()


@st.cache_data
def tuned_threshold(fail, prevent, false_alarm, efficacy):
    costs = dict(fail=fail, prevent=prevent, false_alarm=false_alarm, efficacy=efficacy)
    return core.best_threshold(val_preds["y"].values, val_preds["p"].values, costs)


def shap_values(X):
    sv = bundle["explainer"].shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def kpi(col, label, value, hint=""):
    col.markdown(f'<div class="kpi"><div class="lab">{label}</div>'
                 f'<div class="val">{value}</div><div class="hint">{hint}</div></div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — shift setup and business assumptions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏭 Shift setup")
    source = st.radio("Machine readings", ["Demo shift (held-out data)", "Upload CSV"],
                      help="The demo uses machines the model never saw during training.")
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("AI4I-format CSV", type="csv")
        st.caption("Needs columns: Type, Air temperature [K], Process temperature [K], "
                   "Rotational speed [rpm], Torque [Nm], Tool wear [min].")
    else:
        n_machines = st.slider("Machines on this shift", 50, 400, 150, 10)
        shift_no = st.number_input("Shift #", 1, 999, 7, help="Change to load a different shift.")
    capacity = st.slider("Technician capacity (jobs this shift)", 1, 30, 6)
    noisy = st.toggle("Simulate messy sensors", value=False,
                      help="Adds 10% sensor noise and drops 10% of readings — "
                           "shows how the copilot behaves on real-world data.")

    st.header("💰 Cost assumptions (RM)")
    c_fail = st.number_input("Unplanned failure", 1_000, 500_000,
                             int(core.DEFAULT_COSTS["fail"]), 1_000,
                             help="Downtime + emergency repair + scrapped product.")
    c_prev = st.number_input("Planned preventive fix", 100, 100_000,
                             int(core.DEFAULT_COSTS["prevent"]), 100)
    c_fa = st.number_input("False alarm (needless stop)", 0, 50_000,
                           int(core.DEFAULT_COSTS["false_alarm"]), 100)
    eff = st.slider("Fix success rate", 0.5, 1.0, core.DEFAULT_COSTS["efficacy"], 0.05,
                    help="Share of failures a planned fix actually prevents.")
    costs = dict(fail=float(c_fail), prevent=float(c_prev), false_alarm=float(c_fa), efficacy=eff)
    threshold, thr_curve = tuned_threshold(costs["fail"], costs["prevent"],
                                           costs["false_alarm"], costs["efficacy"])
    st.caption(f"Alert threshold auto-tuned to these costs: **{threshold:.0%}** risk")

# ---------------------------------------------------------------------------
# Build the shift snapshot
# ---------------------------------------------------------------------------
if source == "Upload CSV":
    if uploaded is None:
        st.info("Upload a CSV in the sidebar, or switch back to the demo shift.")
        st.stop()
    raw = pd.read_csv(uploaded)
    missing_cols = [c for c in core.SENSORS + ["Type"] if c not in raw.columns]
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}")
        st.stop()
    raw["Machine"] = (raw["Product ID"].astype(str) if "Product ID" in raw.columns
                      else [f"M-{i + 1:03d}" for i in range(len(raw))])
else:
    raw = test_set.sample(n=min(n_machines, len(test_set)),
                          random_state=int(shift_no)).reset_index(drop=True)
    raw["Machine"] = [f"M-{i + 1:03d}" for i in range(len(raw))]

if noisy:
    raw = core.corrupt(raw, 0.10, 0.10, np.random.default_rng(123))

ranked, feats = core.triage(raw, bundle, costs, threshold, capacity)
sv = shap_values(feats)

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.markdown("# 🛠️ Maintenance Triage Copilot")
st.markdown('<div class="sub">For the maintenance supervisor: <b>which machines to fix this shift, '
            'what is wrong with them, and what it saves</b> — before they break.</div>',
            unsafe_allow_html=True)

fix = ranked[ranked.status == "Fix today"]
nxt = ranked[ranked.status == "Next shift"]
k = st.columns(5)
kpi(k[0], "Machines monitored", f"{len(ranked)}", "this shift")
kpi(k[1], "Fix today", f"{len(fix)}", f"of {capacity} technician slots")
kpi(k[2], "Queued next shift", f"{len(nxt)}", "over capacity")
kpi(k[3], "Loss avoided today", f"RM {fix.net_benefit.sum():,.0f}", "expected, net of fix costs")
kpi(k[4], "Sensor checks", f"{int(ranked.sensor_issue.sum())}", "readings missing/implausible")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Today's action list", "🔍 Machine drill-down",
                                  "📈 Impact & proof", "ℹ️ Model & assumptions"])

# ---------------------------------------------------------------------------
# Tab 1 — action list
# ---------------------------------------------------------------------------
with tab1:
    left, right = st.columns([3, 1])
    with left:
        only = st.checkbox("Show only machines that need attention", value=True)
        view = ranked if not only else ranked[ranked.status != "Healthy"]
        table = pd.DataFrame({
            "#": view["priority"],
            "Machine": view["Machine"],
            "Grade": view["Type"],
            "Status": view["status"].map(STATUS_ICON),
            "Failure risk": view["risk"],
            "Likely issue": view["failure_type"],
            "Diagnosis confidence": view["type_confidence"],
            "Expected loss (RM)": view["expected_loss"].round(0),
            "Net saving if fixed (RM)": view["net_benefit"].clip(lower=0).round(0),
            "Recommended action": view["action"],
            "Owner": view["owner"],
        })
        st.dataframe(
            table, hide_index=True, width="stretch", height=min(460, 38 + 35 * len(table)),
            column_config={
                "Failure risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Expected loss (RM)": st.column_config.NumberColumn(format="%,d"),
                "Net saving if fixed (RM)": st.column_config.NumberColumn(format="%,d"),
                "Recommended action": st.column_config.TextColumn(width="large"),
            })
        st.download_button("⬇️ Download work orders (CSV)",
                           table[table.Status.str.contains("Fix today|Next shift|Check sensor")]
                           .to_csv(index=False).encode(),
                           file_name=f"work_orders_shift.csv", mime="text/csv")
    with right:
        st.markdown("**Why these machines first?**")
        st.caption("Ranked by *expected RM saved by acting now*: failure probability × "
                   "cost of failure × fix success rate, minus the cost of the fix or a needless stop. "
                   "Only machines above the cost-tuned alert threshold get a work order, "
                   "and only as many as your technicians can handle.")
        attn = ranked[ranked.status.isin(["Fix today", "Next shift", "Check sensor"])]
        if len(attn):
            mix = attn["failure_type"].value_counts().reset_index()
            mix.columns = ["Issue", "Machines"]
            fig = px.bar(mix, x="Machines", y="Issue", orientation="h", height=260,
                         color_discrete_sequence=["#dc2626"])
            fig.update_xaxes(tickformat="d", nticks=6)
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title="Issue mix",
                              yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.success("No machines need intervention this shift.")

# ---------------------------------------------------------------------------
# Tab 2 — drill-down
# ---------------------------------------------------------------------------
with tab2:
    labels = [f"#{r.priority} · {r.Machine} · {r.status} · {r.risk:.0%} risk"
              for r in ranked.itertuples()]
    pick = st.selectbox("Machine", range(len(ranked)), format_func=lambda i: labels[i])
    row = ranked.iloc[pick]
    exp = core.explain(row, feats.iloc[pick], sv[pick], bundle["normal"], costs)

    a, b = st.columns([1.25, 1])
    with a:
        color = STATUS_COLOR[row.status]
        reasons = "".join(f"<li>{r}</li>" for r in exp["reasons"]) or "<li>No strong risk drivers.</li>"
        headline = exp["headline"].replace("**", "")
        st.markdown(
            f'<div class="card {STATUS_CARD[row.status]}">'
            f'<span class="pill" style="background:{color}">{row.status}</span>'
            f'&nbsp;&nbsp;<b>{row.Machine}</b> · grade {row.Type}'
            f'<h4 style="margin-top:10px">{headline}</h4>'
            f'<b>Why the model thinks so</b><ul>{reasons}</ul></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="card blue"><b>Recommended action</b><br>{exp["action"]}<br>'
            f'<span style="color:#6b7280">Owner: {exp["owner"]} · ~{exp["minutes"]} min</span></div>',
            unsafe_allow_html=True)
        st.markdown(exp["value"])

        st.markdown("**Live readings vs. healthy average**")
        cols = st.columns(3)
        show = [("air_k", "Air temp", "K"), ("proc_k", "Process temp", "K"), ("rpm", "Speed", "rpm"),
                ("torque_nm", "Torque", "Nm"), ("wear_min", "Tool wear", "min"), ("power_kw", "Power", "kW")]
        for i, (f, name, unit) in enumerate(show):
            v = feats.iloc[pick][f]
            n = bundle["normal"][f]
            cols[i % 3].metric(f"{name} ({unit})", "missing" if pd.isna(v) else f"{v:,.1f}",
                               None if pd.isna(v) else f"{v - n:+,.1f} vs normal",
                               delta_color="off")
    with b:
        contrib = pd.DataFrame({
            "feature": [core.FEATURE_TEXT[f][0] for f in core.FEATURES],
            "impact": sv[pick],
        }).sort_values("impact", key=abs, ascending=True).tail(8)
        fig = go.Figure(go.Bar(
            x=contrib["impact"], y=contrib["feature"], orientation="h",
            marker_color=np.where(contrib["impact"] > 0, "#dc2626", "#2563eb")))
        fig.update_layout(title="What drives this machine's risk", height=330,
                          margin=dict(l=0, r=0, t=40, b=0),
                          xaxis_title="← lowers risk   ·   raises risk →")
        st.plotly_chart(fig, width="stretch")

        mp = pd.DataFrame({"Issue": [core.MODE_INFO[m]["short"] for m in core.MODES],
                           "Probability": [row[f"p_{m}"] for m in core.MODES]})
        fig2 = px.bar(mp, x="Probability", y="Issue", orientation="h", height=230,
                      range_x=[0, 1], color_discrete_sequence=["#f97316"])
        fig2.update_layout(title="Which failure type?", margin=dict(l=0, r=0, t=40, b=0),
                           yaxis_title=None, xaxis_tickformat=".0%")
        st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3 — impact & proof
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### Does it help the supervisor decide better? Backtest on unseen data")
    st.caption(f"{len(test_set):,} machine-process records the model never saw in training, "
               f"replayed shift by shift (100 machines per shift, {capacity} technician jobs each). "
               "Costs follow your sidebar assumptions — change them and everything re-computes.")
    tt = pd.DataFrame({"y": test_set[core.LABEL], "p": test_set["p"],
                       "wear_min": test_set[core.RAW["wear"]],
                       "TWF": test_set["TWF"], "OSF": test_set["OSF"]})
    bt = core.backtest(tt, costs, threshold, shift_size=100, capacity=capacity)
    rtf = bt.set_index("policy").loc["Run-to-failure"]
    sch = bt.set_index("policy").loc["Fixed schedule"]
    cop = bt.set_index("policy").loc["AI Triage Copilot"]
    best = bt.set_index("policy").loc["Perfect foresight"]
    pct = cop.saving_vs_rtf / rtf.total_cost if rtf.total_cost else 0
    capture = cop.saving_vs_rtf / best.saving_vs_rtf if best.saving_vs_rtf else 0

    k = st.columns(4)
    kpi(k[0], "Cost cut vs run-to-failure", f"{pct:.0%}", f"RM {cop.saving_vs_rtf:,.0f} saved")
    kpi(k[1], "Failures prevented", f"{int(cop.failures_caught)} / {int(rtf.failures_missed)}",
        f"schedule-based: {int(sch.failures_caught)}")
    kpi(k[2], "False alarms", f"{int(cop.false_alarms)}",
        f"schedule-based: {int(sch.false_alarms)}")
    kpi(k[3], "Of max possible saving", f"{capture:.0%}", "vs. perfect foresight")
    st.write("")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        order = ["Run-to-failure", "Fixed schedule", "AI Triage Copilot", "Perfect foresight"]
        fig = px.bar(bt, x="policy", y="total_cost", text="total_cost", height=360,
                     category_orders={"policy": order}, color="policy",
                     color_discrete_map={"Run-to-failure": "#9ca3af", "Fixed schedule": "#9ca3af",
                                         "AI Triage Copilot": "#dc2626", "Perfect foresight": "#d1d5db"})
        fig.update_traces(texttemplate="RM %{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, title="Total maintenance cost on held-out data",
                          yaxis_title="RM", xaxis_title=None, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        disp = bt[["policy", "interventions", "failures_caught", "failures_missed",
                   "false_alarms", "total_cost", "note"]].copy()
        disp.columns = ["Policy", "Jobs", "Caught", "Missed", "False alarms", "Total RM", "How it works"]
        st.dataframe(disp, hide_index=True, width="stretch",
                     column_config={"Total RM": st.column_config.NumberColumn(format="%,d")})
        procs = st.number_input("Assumed machine-processes per year (for annual projection)",
                                1_000, 1_000_000, 10_000, 1_000,
                                help="Assumption — set this to your plant's real volume.")
        st.markdown(f"**Annual projection:** ≈ **RM {cop.saving_vs_rtf / len(test_set) * procs:,.0f}** "
                    f"saved per year vs run-to-failure, "
                    f"RM {(sch.total_cost - cop.total_cost) / len(test_set) * procs:,.0f} vs a fixed schedule.")

    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(thr_curve, x="threshold", y="total_cost", height=320)
        fig.add_vline(x=threshold, line_dash="dash", line_color="#dc2626",
                      annotation_text=f"chosen {threshold:.0%}")
        fig.update_layout(title="Why this alert threshold? (fewest RM lost)",
                          xaxis_tickformat=".0%", xaxis_title="Alert when risk ≥",
                          yaxis_title="Total RM (validation)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Too low → technicians chase false alarms. Too high → failures slip through. "
                   "The threshold is picked on validation data to minimise total cost.")
    with c4:
        rb = robustness.melt(id_vars="scenario", value_vars=["precision", "recall"])
        fig = px.bar(rb, x="scenario", y="value", color="variable", barmode="group", height=320,
                     color_discrete_map={"precision": "#2563eb", "recall": "#dc2626"})
        fig.update_layout(title="Robustness to noisy & missing sensor data",
                          yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None,
                          legend_title=None, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("The model was trained on a deliberately messy copy of the data, "
                   "so performance degrades gradually instead of collapsing.")

# ---------------------------------------------------------------------------
# Tab 4 — model card
# ---------------------------------------------------------------------------
with tab4:
    tm = meta["test_metrics"]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Model card")
        st.markdown(f"""
- **Data:** {meta['source']} · {meta['n_rows']:,} records · {meta['failure_rate']:.1%} failures
- **Split:** {meta['split']['train']:,} train / {meta['split']['validation']:,} validation (threshold tuning) / {meta['split']['test']:,} test (all results)
- **Detect:** LightGBM classifier on 6 sensor readings + 4 physics-based features
- **Categorize:** one classifier per failure type (tool wear, heat dissipation, power, overstrain); low-confidence → *unclassified*; missing/implausible readings → *sensor issue*
- **Prioritize:** expected RM saved by acting now, capped by technician capacity
- **Explain:** SHAP values → plain-language reasons + action playbook
""")
        st.markdown("### Held-out test performance")
        m = st.columns(4)
        m[0].metric("Recall", f"{tm['recall']:.0%}", help="Share of real failures flagged")
        m[1].metric("Precision", f"{tm['precision']:.0%}", help="Share of alerts that were real")
        m[2].metric("PR-AUC", f"{tm['pr_auc']:.2f}")
        m[3].metric("Failure-type accuracy", f"{meta['failure_type_accuracy']:.0%}")
        st.markdown("### Honest limitations")
        st.markdown("""
- Records are **operating snapshots**, not time series: the copilot flags machines *currently in a high-risk state*. With live sensor streams, a time-to-failure forecaster would be added on top.
- **Random failures** (~0.1%) are unpredictable by design and excluded from categorisation.
- Tool-wear failures happen at a random point in the 200–240 min window, so they are the hardest type to catch early.
- Cost figures are **assumptions** — the sidebar lets a plant plug in its own numbers.
""")
    with c2:
        fi = pd.Series(meta["feature_importance"]).rename(lambda f: core.FEATURE_TEXT[f][0])
        fi = fi.sort_values().reset_index()
        fi.columns = ["Feature", "Share of model decisions"]
        fig = px.bar(fi, x="Share of model decisions", y="Feature", orientation="h", height=380,
                     color_discrete_sequence=["#2563eb"])
        fig.update_layout(title="What the model relies on overall", xaxis_tickformat=".0%",
                          yaxis_title=None, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Physics-based features** (temperature gap, power, strain vs limit) "
                    "carry most of the signal — so explanations map to things a technician can check.")
