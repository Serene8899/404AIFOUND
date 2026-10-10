# 🛠️ Maintenance Triage Copilot
**Track 1 — Predict Early, Decide Better**

For the **maintenance supervisor** who has more machines than technicians:
*which machines should we fix this shift, what is wrong with them, and how much does it save?*

| Requirement | How the copilot does it |
|---|---|
| **Detect** | LightGBM model predicts failure risk from 6 sensor readings + 4 physics features |
| **Categorize** | One model per failure type: tool wear, heat dissipation, power, overstrain. Missing/implausible readings are routed to "sensor issue" instead of a false alarm |
| **Prioritize** | Ranks by expected RM saved by acting now: P(failure) × failure cost × fix success rate − fix/false-alarm cost. Only as many work orders as the technicians can handle |
| **Explain** | SHAP drivers → plain-language reasons + a concrete action, owner and time estimate |
| **Minimize false alarms** | Alert threshold is tuned on validation data to minimise total RM cost, and re-tunes live when costs change |
| **Robust to noisy data** | Trained on clean data plus a deliberately messy copy (sensor noise + dropped readings) |
| **Prove it** | Backtest on unseen data vs run-to-failure, fixed schedule and perfect foresight |

---

## Run it

```bash
pip install -r requirements.txt
python train.py          # ~30 s — trains models, writes models/
streamlit run app.py     # opens the dashboard
```

The `models/` folder is already trained, so `streamlit run app.py` works straight away.

### Using the real AI4I 2020 dataset
The included data is a **replica generated from the dataset's published rules** (see `make_data.py`), because the original couldn't be downloaded in the build environment.
Before the hackathon:

1. Download `ai4i2020.csv` from UCI ("AI4I 2020 Predictive Maintenance Dataset") or Kaggle.
2. Save it as `data/ai4i2020.csv`.
3. Run `python train.py` again.

The pipeline picks up the real file automatically. The column names are identical, and the dashboard's model card shows which data source was used.

---

## Files

| File | What it does |
|---|---|
| `core.py` | Feature engineering, categorisation, cost model, explanations, backtest |
| `train.py` | Trains detection + 4 failure-type models, tunes the threshold, runs robustness + backtest, saves to `models/` |
| `app.py` | Streamlit dashboard |
| `make_data.py` | Generates the replica dataset (only used when the real file is absent) |
| `models/` | Trained models, held-out predictions, metrics |

## Dashboard tour (demo order)

1. **📋 Today's action list.** A ranked work-order list with status, risk, likely issue, RM at stake, action and owner. Download as CSV.
2. **🔍 Machine drill-down.** Why this machine is flagged (SHAP chart + plain English), which failure type, and what to do.
3. **📈 Impact & proof.** The backtest on unseen data, the threshold-vs-cost curve and the robustness chart.
4. **ℹ️ Model & assumptions.** The model card, metrics and honest limitations.

**Live demo moments**
- Turn on **"Simulate messy sensors"** in the sidebar. The copilot keeps working and routes bad readings to "Check sensor" instead of raising false alarms.
- Change the **failure cost** or **technician capacity**. The threshold, ranking and backtest all re-compute instantly.

---

## Results (replica data, held-out test set of 2,000 records)

| | |
|---|---|
| Failures caught (recall) | ~90% |
| Alerts that were real (precision) | ~69% |
| Failure-type accuracy on caught failures | 100%* |
| Cost vs run-to-failure | **−65%** with only 6 technician jobs per 100-machine shift |
| False alarms vs fixed tool-change schedule | **20 vs 145** |
| Share of the maximum possible saving captured | ~81% |

\*Failure types in AI4I follow physical rules, so they are highly separable. Expect slightly different numbers on the real file. Re-run `train.py` and quote those.
