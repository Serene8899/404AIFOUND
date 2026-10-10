"""
Core logic shared by training (train.py) and the dashboard (app.py):
feature engineering, failure categorisation, cost-based prioritisation,
plain-language explanations and policy backtesting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
RAW = {
    "air": "Air temperature [K]",
    "proc": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "wear": "Tool wear [min]",
}
SENSORS = list(RAW.values())
MODES = ["TWF", "HDF", "PWF", "OSF"]  # RNF is random by design -> not modelled
LABEL = "Machine failure"

FEATURES = [
    "type_code", "air_k", "proc_k", "rpm", "torque_nm", "wear_min",
    "temp_diff_k", "power_kw", "strain", "strain_ratio",
]

OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}  # tool wear x torque limit per grade

# ---------------------------------------------------------------------------
# Default business assumptions (editable live in the dashboard)
# ---------------------------------------------------------------------------
DEFAULT_COSTS = {
    "fail": 20_000.0,        # RM: unplanned failure (downtime + emergency repair + scrap)
    "prevent": 2_000.0,      # RM: planned preventive fix when the machine really was at risk
    "false_alarm": 1_500.0,  # RM: unnecessary stop + inspection when the machine was fine
    "efficacy": 0.90,        # share of failures a planned fix actually prevents
}

# ---------------------------------------------------------------------------
# Human-readable knowledge base
# ---------------------------------------------------------------------------
MODE_INFO = {
    "TWF": {
        "name": "Tool wear failure",
        "short": "Tool wear",
        "why": "the cutting tool is near the end of its life",
        "action": "Replace the cutting tool at the next changeover (before it breaks mid-process).",
        "owner": "Machine operator",
        "minutes": 15,
    },
    "HDF": {
        "name": "Heat dissipation failure",
        "short": "Overheating",
        "why": "the machine is not shedding heat — the process-to-air temperature gap is small while the spindle is slow",
        "action": "Check coolant flow and ventilation; raise spindle speed above ~1,380 rpm if the process allows.",
        "owner": "Maintenance technician",
        "minutes": 30,
    },
    "PWF": {
        "name": "Power failure",
        "short": "Power out of band",
        "why": "mechanical power (torque x speed) is outside the safe 3.5–9 kW operating band",
        "action": "Inspect motor and drive; adjust the torque/speed set-point to bring power back into 3.5–9 kW.",
        "owner": "Electrical technician",
        "minutes": 45,
    },
    "OSF": {
        "name": "Overstrain failure",
        "short": "Overstrain",
        "why": "the worn tool is under too much torque for this product grade",
        "action": "Reduce cutting load now, and replace the tool before the next batch.",
        "owner": "Machine operator",
        "minutes": 20,
    },
    "UNK": {
        "name": "Unclassified anomaly",
        "short": "Unclassified",
        "why": "readings look risky but do not match a known failure signature",
        "action": "Run a general inspection and log findings so the model can learn this pattern.",
        "owner": "Maintenance technician",
        "minutes": 30,
    },
    "SENSOR": {
        "name": "Sensor data issue",
        "short": "Sensor issue",
        "why": "one or more sensor readings are missing or physically implausible",
        "action": "Verify the sensors before trusting the risk score; treat the machine as 'unknown'.",
        "owner": "Instrumentation technician",
        "minutes": 20,
    },
}

FEATURE_TEXT = {
    "type_code": ("Product grade", ""),
    "air_k": ("Air temperature", " K"),
    "proc_k": ("Process temperature", " K"),
    "rpm": ("Rotational speed", " rpm"),
    "torque_nm": ("Torque", " Nm"),
    "wear_min": ("Tool wear", " min"),
    "temp_diff_k": ("Process–air temperature gap", " K"),
    "power_kw": ("Mechanical power", " kW"),
    "strain": ("Tool strain (wear x torque)", " min·Nm"),
    "strain_ratio": ("Strain vs. overstrain limit", ""),
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw AI4I columns -> model features. Missing sensor values stay NaN
    (LightGBM handles them natively; they are also flagged as sensor issues)."""
    t = df["Type"].astype(str).str.upper().str[0]
    X = pd.DataFrame(index=df.index)
    X["type_code"] = t.map({"L": 0, "M": 1, "H": 2}).astype(float)
    X["air_k"] = pd.to_numeric(df[RAW["air"]], errors="coerce")
    X["proc_k"] = pd.to_numeric(df[RAW["proc"]], errors="coerce")
    X["rpm"] = pd.to_numeric(df[RAW["rpm"]], errors="coerce")
    X["torque_nm"] = pd.to_numeric(df[RAW["torque"]], errors="coerce")
    X["wear_min"] = pd.to_numeric(df[RAW["wear"]], errors="coerce")
    X["temp_diff_k"] = X["proc_k"] - X["air_k"]
    X["power_kw"] = X["torque_nm"] * X["rpm"] * 2 * np.pi / 60 / 1000
    X["strain"] = X["wear_min"] * X["torque_nm"]
    limit = t.map(OSF_LIMIT).astype(float)
    X["strain_ratio"] = X["strain"] / limit
    return X[FEATURES]


def sensor_issues(df: pd.DataFrame, ranges: dict) -> pd.Series:
    """True where a reading is missing or far outside the range seen in training."""
    bad = pd.Series(False, index=df.index)
    for col in SENSORS:
        v = pd.to_numeric(df[col], errors="coerce")
        lo, hi = ranges[col]
        span = hi - lo
        bad |= v.isna() | (v < lo - 0.25 * span) | (v > hi + 0.25 * span)
    return bad


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------
def net_benefit(p, costs):
    """Expected RM saved by intervening now vs. waiting (can be negative)."""
    p = np.asarray(p, dtype=float)
    saved = p * costs["efficacy"] * costs["fail"]
    spend = p * costs["prevent"] + (1 - p) * costs["false_alarm"]
    return saved - spend


def decision_cost(y, intervene, costs, preventable=None):
    """Realised RM cost of a set of decisions given what actually happened."""
    y = np.asarray(y).astype(bool)
    a = np.asarray(intervene).astype(bool)
    prev = np.ones_like(y, dtype=bool) if preventable is None else np.asarray(preventable).astype(bool)
    eff = costs["efficacy"]
    c = np.zeros(len(y))
    hit = a & y & prev
    c[hit] = costs["prevent"] + (1 - eff) * costs["fail"]
    c[a & y & ~prev] = costs["prevent"] + costs["fail"]       # fixed the wrong thing
    c[a & ~y] = costs["false_alarm"]
    c[~a & y] = costs["fail"]
    return c


def best_threshold(y, p, costs, grid=None):
    """Alert threshold that minimises total RM cost on validation data."""
    grid = np.linspace(0.005, 0.95, 190) if grid is None else grid
    totals = [decision_cost(y, p >= t, costs).sum() for t in grid]
    i = int(np.argmin(totals))
    return float(grid[i]), pd.DataFrame({"threshold": grid, "total_cost": totals})


def categorize(mode_probs: pd.DataFrame, issue: pd.Series) -> pd.Series:
    """Most likely failure type. A type is assigned when it clearly dominates the
    others (>=60% of the mode probability mass); otherwise 'unclassified'."""
    best = mode_probs[MODES].idxmax(axis=1)
    top = mode_probs[MODES].max(axis=1)
    share = top / mode_probs[MODES].sum(axis=1).clip(lower=1e-9)
    cat = best.where((top >= 0.5) | ((top >= 0.02) & (share >= 0.6)), "UNK")
    return cat.where(~issue, "SENSOR")


def type_confidence(mode_probs: pd.DataFrame) -> pd.Series:
    top = mode_probs[MODES].max(axis=1)
    return pd.Series(np.where(top >= 0.5, "High", np.where(top >= 0.15, "Medium", "Low")),
                     index=mode_probs.index)


def triage(df_raw, bundle, costs, threshold, capacity):
    """Score a set of machines and return a ranked action list."""
    X = build_features(df_raw)
    p = bundle["detector"].predict_proba(X)[:, 1]
    mp = pd.DataFrame({m: bundle["modes"][m].predict_proba(X)[:, 1] for m in MODES}, index=X.index)
    issue = sensor_issues(df_raw, bundle["ranges"])
    out = df_raw.copy()
    out["risk"] = p
    for m in MODES:
        out[f"p_{m}"] = mp[m]
    out["category"] = categorize(mp, issue)
    out["type_confidence"] = type_confidence(mp)
    out["failure_type"] = out["category"].map(lambda c: MODE_INFO[c]["name"])
    out["expected_loss"] = p * costs["fail"]
    out["net_benefit"] = net_benefit(p, costs)
    out["sensor_issue"] = issue
    out = out.sort_values(["net_benefit", "risk"], ascending=False).reset_index(drop=True)

    above = (out["risk"] >= threshold) & (out["net_benefit"] > 0)
    rank_above = above.cumsum()
    status = np.where(above & (rank_above <= capacity), "Fix today",
             np.where(above, "Next shift",
             np.where(out["risk"] >= threshold / 3, "Watch", "Healthy")))
    status = np.where(out["sensor_issue"] & (status == "Healthy"), "Check sensor", status)
    out["status"] = status
    out["action"] = out["category"].map(lambda c: MODE_INFO[c]["action"])
    out["owner"] = out["category"].map(lambda c: MODE_INFO[c]["owner"])
    out.loc[out["status"] == "Healthy", ["action", "owner"]] = ["No action — keep monitoring", "—"]
    out.insert(0, "priority", np.arange(1, len(out) + 1))
    return out, build_features(out)


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def _fmt(feat, v):
    if pd.isna(v):
        return "missing"
    if feat == "type_code":
        return {0: "L (low)", 1: "M (medium)", 2: "H (high)"}.get(int(v), "?")
    if feat == "strain_ratio":
        return f"{v:.0%}"
    if feat in ("power_kw",):
        return f"{v:.2f}"
    if feat in ("strain", "rpm", "wear_min"):
        return f"{v:,.0f}"
    return f"{v:.1f}"


def explain(row: pd.Series, feats: pd.Series, shap_row: np.ndarray, normal: dict,
            costs: dict, top: int = 3) -> dict:
    """Turn model output + SHAP values into a plain-language explanation."""
    cat = row["category"]
    info = MODE_INFO[cat]
    order = np.argsort(-shap_row)
    reasons = []
    for i in order[:top]:
        if shap_row[i] <= 0:
            break
        f = FEATURES[i]
        name, unit = FEATURE_TEXT[f]
        val = _fmt(f, feats[f])
        norm = _fmt(f, normal.get(f, np.nan))
        if f == "type_code":
            reasons.append(f"Product grade is {val}, which has a lower overstrain tolerance")
        elif f == "strain_ratio":
            reasons.append(f"{name} is {val} (healthy machines average {norm})")
        else:
            reasons.append(f"{name} is {val}{unit} (healthy machines average {norm}{unit})")
    risk = row["risk"]
    hedge = " (low-confidence diagnosis — confirm on inspection)" if (
        cat in MODES and row.get("type_confidence") == "Low") else ""
    headline = (f"{risk:.0%} failure risk — likely **{info['name'].lower()}**{hedge}: {info['why']}."
                if row["status"] not in ("Healthy",) else
                f"{risk:.1%} failure risk — operating normally.")
    # Secondary failure signature, if the machine shows two at once
    if row["status"] != "Healthy" and cat in MODES:
        others = sorted(((row[f"p_{m}"], m) for m in MODES if m != cat), reverse=True)
        if others and others[0][0] >= 0.5:
            headline += f" Also showing signs of **{MODE_INFO[others[0][1]]['name'].lower()}**."
    avoided = max(row["net_benefit"], 0)
    return {
        "headline": headline,
        "reasons": reasons,
        "action": info["action"] if row["status"] != "Healthy" else "No action — keep monitoring.",
        "owner": info["owner"],
        "minutes": info["minutes"],
        "value": (f"Acting now avoids an expected **RM {avoided:,.0f}** "
                  f"(expected loss RM {row['expected_loss']:,.0f} if ignored)."
                  if avoided > 0 else
                  "Intervening now would cost more than it is expected to save."),
    }


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(test: pd.DataFrame, costs: dict, threshold: float,
             shift_size: int = 100, capacity: int = 10) -> pd.DataFrame:
    """Compare maintenance policies on held-out data.
    test needs: y, p, wear_min, TWF, OSF."""
    y = test["y"].values
    p = test["p"].values
    rows = []

    def add(name, intervene, preventable=None, note=""):
        c = decision_cost(y, intervene, costs, preventable)
        a = np.asarray(intervene).astype(bool)
        caught = (a & y.astype(bool) & (np.ones_like(a) if preventable is None else preventable)).sum()
        rows.append({
            "policy": name,
            "interventions": int(a.sum()),
            "failures_caught": int(caught),
            "failures_missed": int(y.sum() - caught),
            "false_alarms": int((a & ~y.astype(bool)).sum()),
            "total_cost": float(c.sum()),
            "note": note,
        })

    add("Run-to-failure", np.zeros_like(y), note="Fix only after breakdown")
    tool_pre = (test["TWF"].values == 1) | (test["OSF"].values == 1)
    add("Fixed schedule", test["wear_min"].values >= 200, preventable=tool_pre,
        note="Replace tool at ≥200 min wear (only prevents tool-related failures)")

    # Copilot under a realistic capacity limit: shifts of N machines, K jobs each
    nb = net_benefit(p, costs)
    a_cap = np.zeros(len(y), dtype=bool)
    order_idx = np.arange(len(y))
    for s in range(0, len(y), shift_size):
        idx = order_idx[s:s + shift_size]
        cand = idx[(p[idx] >= threshold) & (nb[idx] > 0)]
        cand = cand[np.argsort(-nb[cand])][:capacity]
        a_cap[cand] = True
    add("AI Triage Copilot", a_cap,
        note=f"Model-ranked, max {capacity} jobs per {shift_size}-machine shift")
    add("Perfect foresight", y.astype(bool), note="Theoretical best (upper bound)")

    bt = pd.DataFrame(rows)
    base = bt.loc[bt.policy == "Run-to-failure", "total_cost"].iloc[0]
    bt["saving_vs_rtf"] = base - bt["total_cost"]
    return bt


def corrupt(df, noise_frac, missing_frac, rng):
    """Add Gaussian sensor noise (as a fraction of each sensor's spread)
    and knock out a share of readings, to mimic messy plant data."""
    d = df.copy()
    for col in SENSORS:
        v = pd.to_numeric(d[col], errors="coerce").astype(float)
        sd = v.std()
        v = v + rng.normal(0, noise_frac * sd, len(d))
        if missing_frac > 0:
            v[rng.random(len(d)) < missing_frac / len(SENSORS)] = np.nan
        d[col] = v
    return d
