"""
==============================================================================
COLD STORAGE SENSOR ANOMALY DETECTION
==============================================================================
Student project script: detects the 10 injected anomaly scenarios in the
cold storage dataset using a mix of:
  (A) Statistical / rule-based thresholds  -> good for "obvious" anomalies
      (point outliers, flatlines, missing data, duplicates)
  (B) Machine Learning (Isolation Forest)   -> good for multivariate /
      contextual anomalies that no single sensor threshold would catch
      (e.g. a door-open event where 4 sensors move together)

We then compare our flags against the ground-truth labels (from the
"_labeled" file) and build a CONFUSION MATRIX to see how many anomalies we
correctly caught (True Positives), missed (False Negatives), and how many
normal readings we incorrectly flagged (False Positives).

WHY combine statistics AND ML?
  - Pure thresholding is easy to explain but misses anomalies that only
    look wrong when you consider several sensors together.
  - Pure ML (Isolation Forest) is good at multivariate patterns but can
    struggle with rare very-short spikes or exact-duplicate/missing-data
    issues, which are really data-quality problems, not statistical
    outliers.
  - Using both and taking the union gives broader coverage - a realistic
    "intermediate data scientist" approach.
==============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 140)

# ------------------------------------------------------------------------
# 0. LOAD DATA
# ------------------------------------------------------------------------
# NOTE: In a real project you would only have the "unlabeled" file.
# We load the labeled file too, but ONLY to score our results at the end -
# never to help the detection logic itself.
DATA_PATH = "cold_storage_sensor_data.csv"
LABELED_PATH = "cold_storage_sensor_data_labeled.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
truth = pd.read_csv(LABELED_PATH, parse_dates=["timestamp"])[
    ["timestamp", "is_anomaly", "anomaly_type"]
]

print(f"Loaded {len(df)} rows, {df['timestamp'].duplicated().sum()} duplicate timestamp(s) present.")

# ------------------------------------------------------------------------
# 1. DATA-QUALITY CHECKS (Scenarios 7, 8/10-style issues)
#    These are not "statistical outliers" in the ML sense - they are
#    structural problems in the data itself, so we check for them directly.
# ------------------------------------------------------------------------

df["flag_duplicate_timestamp"] = df.duplicated(subset="timestamp", keep=False).astype(int)

# Missing-value flags (per-sensor). Any NaN in a sensor reading = anomaly.
sensor_cols = [
    "temperature_C",
    "humidity_pct",
    "power_consumption_kWh",
    "co2_level_ppm",
    "compressor_vibration_mms",
    "airflow_ms",
    "refrigerant_pressure_psi",
]
df["flag_missing_data"] = df[sensor_cols].isna().any(axis=1).astype(int)

# ------------------------------------------------------------------------
# 2. STATISTICAL THRESHOLD DETECTION (per-sensor, univariate)
#    Method: rolling z-score.
#      z = (x - rolling_mean) / rolling_std
#    A z-score beyond +/-3 is a classic "3-sigma rule" threshold: under a
#    normal distribution, ~99.7% of points fall within 3 standard
#    deviations, so anything beyond that is statistically rare/unusual.
#
#    We use a ROLLING window (not the global mean/std) so the threshold
#    adapts to each sensor's natural cycle (e.g. compressor duty cycling)
#    instead of flagging every normal oscillation as an anomaly.
# ------------------------------------------------------------------------

ROLL_WINDOW = 8    # 8 x 15min = 2 hours of context
Z_THRESHOLD = 3.0  # 3-sigma rule


def rolling_zscore_flags(series, window=ROLL_WINDOW, z_thresh=Z_THRESHOLD):
    """Return (z-scores, boolean flag) using a rolling mean/std."""
    roll_mean = series.rolling(window, center=True, min_periods=3).mean()
    roll_std = series.rolling(window, center=True, min_periods=3).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    flag = (z.abs() > z_thresh).fillna(False)
    return z, flag


for col in sensor_cols:
    z, flag = rolling_zscore_flags(df[col])
    df[f"z_{col}"] = z
    df[f"flag_{col}_zscore"] = flag.astype(int)

# ------------------------------------------------------------------------
# 3. FLATLINE / STUCK-SENSOR DETECTION (Scenario 2: stuck temperature)
#    Method: rolling standard deviation == 0 (or extremely close to it)
#    A real sensor almost never reports the EXACT same value repeatedly;
#    if std over a window collapses to ~0, the sensor is likely frozen/stuck.
# ------------------------------------------------------------------------

FLATLINE_WINDOW = 4  # 1 hour
FLATLINE_STD_THRESH = 1e-6

df["flag_temperature_flatline"] = (
    df["temperature_C"].rolling(FLATLINE_WINDOW).std() < FLATLINE_STD_THRESH
).fillna(False).astype(int)

# ------------------------------------------------------------------------
# 4. TREND / DRIFT DETECTION (Scenarios 4 & 9: refrigerant leak, vibration
#    drift). These are SLOW, gradual changes - a single-point z-score often
#    won't catch them because each individual step is small. Instead we
#    fit a rolling linear regression slope and flag periods where the
#    trend (rate of change) is unusually steep for a sustained period.
# ------------------------------------------------------------------------

TREND_WINDOW = 12  # 3 hours


def rolling_slope(series, window=TREND_WINDOW):
    """Rolling linear-regression slope (units per 15-min step)."""
    def slope_of(y):
        x = np.arange(len(y))
        if np.isnan(y).any():
            return np.nan
        return np.polyfit(x, y, 1)[0]

    return series.rolling(window).apply(slope_of, raw=True)


df["slope_refrigerant_pressure_psi"] = rolling_slope(df["refrigerant_pressure_psi"])
df["slope_co2_level_ppm"] = rolling_slope(df["co2_level_ppm"])
df["slope_compressor_vibration_mms"] = rolling_slope(df["compressor_vibration_mms"])

# Threshold method: ROBUST z-score using median + MAD (Median Absolute
# Deviation) instead of mean/std. A plain mean/std threshold has a
# "masking" problem: the anomaly itself pulls the mean and inflates the
# std, making it look LESS extreme relative to its own distorted baseline.
# Median/MAD are far less sensitive to a handful of outliers, so real
# drifts stand out more reliably - a standard trick for robust thresholds.
def slope_threshold_flag(slope_series, n_mad=3.0):
    median = slope_series.median()
    mad = (slope_series - median).abs().median()
    mad = mad if mad > 1e-9 else 1e-9  # avoid divide-by-zero
    robust_z = 0.6745 * (slope_series - median) / mad
    return robust_z.abs() > n_mad

df["flag_pressure_drift"] = slope_threshold_flag(
    df["slope_refrigerant_pressure_psi"]
).fillna(False).astype(int)

df["flag_vibration_drift"] = slope_threshold_flag(
    df["slope_compressor_vibration_mms"]
).fillna(False).astype(int)

df["flag_co2_drift"] = slope_threshold_flag(
    df["slope_co2_level_ppm"]
).fillna(False).astype(int)

# ------------------------------------------------------------------------
# 5. MACHINE LEARNING: ISOLATION FOREST (multivariate / contextual)
#    Scenario 1 (door left open) is the key example this is meant for:
#    temperature, humidity, CO2, and power all shift together in a way
#    that isn't extreme on any ONE sensor, but IS unusual when viewed
#    jointly. Isolation Forest isolates points that are "few and
#    different" in the multivariate feature space - it doesn't need a
#    manual threshold per sensor.
#
#    contamination=0.05 tells the model to expect ~5% of points to be
#    anomalous. This is a tunable hyperparameter - in a real project you'd
#    experiment with a few values and check the resulting confusion matrix.
# ------------------------------------------------------------------------

ml_features = sensor_cols + ["door_status"]

# Impute missing values (Isolation Forest can't handle NaNs) using median -
# a simple, defensible choice for a short-window sensor dropout.
X = df[ml_features].copy()
X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
)
iso_pred = iso_forest.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
df["flag_isolation_forest"] = (iso_pred == -1).astype(int)
df["isolation_forest_score"] = iso_forest.decision_function(X_scaled)  # lower = more anomalous

# ------------------------------------------------------------------------
# 6. COMBINE ALL DETECTORS INTO ONE FINAL FLAG (ensemble / union approach)
# ------------------------------------------------------------------------

flag_cols = [
    "flag_duplicate_timestamp",
    "flag_missing_data",
    "flag_temperature_flatline",
    "flag_pressure_drift",
    "flag_vibration_drift",
    "flag_co2_drift",
    "flag_isolation_forest",
] + [f"flag_{c}_zscore" for c in sensor_cols]

df["predicted_anomaly"] = (df[flag_cols].sum(axis=1) > 0).astype(int)

# Human-readable list of WHICH detector(s) fired, useful for explaining results
def explain_row(row):
    reasons = [c.replace("flag_", "") for c in flag_cols if row[c] == 1]
    return ", ".join(reasons)

df["flagged_by"] = df.apply(explain_row, axis=1)

# ------------------------------------------------------------------------
# 7. EVALUATION: MERGE WITH GROUND TRUTH & BUILD CONFUSION MATRIX
# ------------------------------------------------------------------------

# NOTE: our dataset intentionally contains one duplicate timestamp (that's
# anomaly scenario #10 - a logging glitch). If we naively merge on
# "timestamp" while duplicates exist on BOTH sides, pandas produces a
# cross-join for that timestamp (2 rows x 2 rows = 4 rows), inflating our
# row count and distorting the confusion matrix. We deduplicate the TRUTH
# table before merging (keeping the flag_duplicate_timestamp column,
# computed earlier, as the actual detector for that scenario) so each row
# of our working data still gets exactly one ground-truth label.
truth_dedup = truth.drop_duplicates(subset="timestamp", keep="first")
eval_df = df.merge(truth_dedup, on="timestamp", how="left", suffixes=("", "_truth"))
eval_df["is_anomaly"] = eval_df["is_anomaly"].fillna(0).astype(int)

y_true = eval_df["is_anomaly"]
y_pred = eval_df["predicted_anomaly"]

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print(pd.DataFrame(
    cm,
    index=["Actual: Normal", "Actual: Anomaly"],
    columns=["Predicted: Normal", "Predicted: Anomaly"],
))

print(f"\nTrue Positives  (correctly caught anomalies) : {tp}")
print(f"False Negatives (missed anomalies)            : {fn}")
print(f"False Positives (normal points wrongly flagged): {fp}")
print(f"True Negatives  (correctly left alone)        : {tn}")

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\nPrecision: {precision:.3f}  (of the points we flagged, how many were real anomalies)")
print(f"Recall   : {recall:.3f}  (of the real anomalies, how many did we catch)")
print(f"F1 Score : {f1:.3f}  (balance of precision & recall)")

print("\nFull classification report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))

# ------------------------------------------------------------------------
# 8. PER-SCENARIO BREAKDOWN
#    This tells us WHICH of the 10 injected anomaly types we caught, and
#    which detector(s) caught them - useful to explain in a report.
# ------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PER-ANOMALY-TYPE BREAKDOWN (recall by scenario)")
print("=" * 70)

anomaly_rows = eval_df[eval_df["is_anomaly"] == 1].copy()
anomaly_rows["anomaly_type"] = anomaly_rows["anomaly_type"].fillna("unknown")

summary = (
    anomaly_rows.assign(single_type=anomaly_rows["anomaly_type"].str.split("|"))
    .explode("single_type")
    .groupby("single_type")
    .agg(
        total_points=("predicted_anomaly", "size"),
        caught=("predicted_anomaly", "sum"),
    )
)
summary["missed"] = summary["total_points"] - summary["caught"]
summary["catch_rate_%"] = (summary["caught"] / summary["total_points"] * 100).round(1)
print(summary.to_string())

print("\nFalse positives (flagged but NOT a true anomaly) - sample of what triggered them:")
fp_rows = eval_df[(eval_df["is_anomaly"] == 0) & (eval_df["predicted_anomaly"] == 1)]
if len(fp_rows) > 0:
    print(fp_rows[["timestamp", "flagged_by"]].to_string(index=False))
else:
    print("None - every flagged point was a true anomaly.")

# ------------------------------------------------------------------------
# 9. VISUALIZATIONS (saved as PNGs for the report)
# ------------------------------------------------------------------------

# 9a. Confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Normal", "Predicted Anomaly"],
    yticklabels=["Actual Normal", "Actual Anomaly"],
)
plt.title("Confusion Matrix - Cold Storage Anomaly Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 9b. Temperature time series with flagged anomalies overlaid
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

plot_specs = [
    ("temperature_C", "Temperature (°C)"),
    ("refrigerant_pressure_psi", "Refrigerant Pressure (psi)"),
    ("compressor_vibration_mms", "Compressor Vibration (mm/s)"),
    ("power_consumption_kWh", "Power Consumption (kWh)"),
]

for ax, (col, label) in zip(axes, plot_specs):
    ax.plot(eval_df["timestamp"], eval_df[col], color="steelblue", linewidth=1, label=label)

    true_pts = eval_df[eval_df["is_anomaly"] == 1]
    pred_pts = eval_df[eval_df["predicted_anomaly"] == 1]

    ax.scatter(true_pts["timestamp"], true_pts[col], color="orange", s=25,
               label="Ground-truth anomaly", zorder=3, marker="o")
    ax.scatter(pred_pts["timestamp"], pred_pts[col], facecolors="none",
               edgecolors="red", s=70, label="Detected anomaly", zorder=4, marker="o")

    ax.set_ylabel(label)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Timestamp")
fig.suptitle("Detected vs. Ground-Truth Anomalies Across Key Sensors", fontsize=14)
plt.tight_layout()
plt.savefig("anomaly_timeseries.png", dpi=150)
plt.close()

# ------------------------------------------------------------------------
# 10. EXPORT RESULTS
# ------------------------------------------------------------------------

output_cols = [
    "timestamp",
    "temperature_C",
    "humidity_pct",
    "door_status",
    "power_consumption_kWh",
    "co2_level_ppm",
    "compressor_vibration_mms",
    "airflow_ms",
    "refrigerant_pressure_psi",
    "predicted_anomaly",
    "flagged_by",
    "is_anomaly",
    "anomaly_type",
]
eval_df[output_cols].to_csv("anomaly_detection_results.csv", index=False)

print("\nSaved: confusion_matrix.png, anomaly_timeseries.png, anomaly_detection_results.csv")
print("\nDone.")
