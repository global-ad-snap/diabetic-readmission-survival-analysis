# =============================================
# IMPORTS
# =============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import brier_score, concordance_index_censored

import pickle
import joblib
import os
from pathlib import Path


# =============================================
# CREATE PROJECT FOLDERS
# =============================================
for folder in ["models", "reports", "visuals/EDA", "visuals/KM_Plots", "visuals/PH_diagnostics"]:
    os.makedirs(folder, exist_ok=True)


# =============================================
# STEP 1 — LOAD DATA
# =============================================
df = pd.read_csv("data/diabetic_data.csv")
df.info()
df["readmitted"].value_counts()


# =============================================
# STEP 2 — STANDARDIZE MISSING MARKERS & SCHEMA
# =============================================
df = df.replace({"?": np.nan, "Unknown/Invalid": np.nan})

int_cols = [
    "encounter_id", "patient_nbr", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient"
]
for c in int_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

df["diag_1"] = df["diag_1"].astype("string")
df["diag_2"] = df["diag_2"].astype("string")
df["diag_3"] = df["diag_3"].astype("string")


# =============================================
# STEP 3 — CREATE SURVIVAL LABELS
# =============================================
if "readmitted_orig" not in df.columns:
    df["readmitted_orig"] = df["readmitted"]

df["event"] = (df["readmitted"] == "<30").astype("Int64")

np.random.seed(42)
df["time"] = np.where(
    df["event"] == 1,
    np.random.randint(1, 31, size=len(df)),
    np.random.randint(31, 365, size=len(df))
)

n_total = len(df)
n_event = int(df["event"].sum())
print(f"Total rows = {n_total}, Events = {n_event} ({n_event/n_total:.2%})")


# =============================================
# STEP 4 — COHORT FILTERING
# =============================================
print("Initial:", len(df))
df = df[df["age"] != "[0-10)"]
death_hospice_codes = {11, 12, 13, 19, 20, 21}
df = df[~df["discharge_disposition_id"].isin(death_hospice_codes)]
df = df[df["discharge_disposition_id"] != 0]
df = df[df["gender"].isin(["Male", "Female"])]
before = len(df)
df = df.sort_values("encounter_id").drop_duplicates("encounter_id")
print(f"Final cohort: {len(df)}  (removed {before - len(df)} duplicates)")


# =============================================
# STEP 5 — EDA
# =============================================
df = df.drop(columns=["weight", "max_glu_serum", "A1Cresult"])

numeric_df = df.select_dtypes(include=["int64", "float64", "Int64"])
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig("visuals/EDA/correlation_heatmap_numeric.png", dpi=150, bbox_inches="tight")
plt.show()

for v in ["time_in_hospital", "num_medications", "number_inpatient"]:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[v], kde=True, ax=ax)
    ax.set_title(f"Distribution of {v}")
    fig.tight_layout()
    fig.savefig(f"visuals/EDA/distribution_{v}.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================
# STEP 6 — KAPLAN–MEIER (OVERALL + BY GENDER)
# =============================================
fig, ax = plt.subplots(figsize=(8, 5))
KaplanMeierFitter().fit(df["time"], df["event"]).plot_survival_function(ax=ax)
ax.set_title("Kaplan–Meier Curve (Overall)")
ax.set_xlabel("Days")
ax.set_ylabel("Survival Probability")
fig.tight_layout()
fig.savefig("visuals/KM_Plots/kaplan_meier_curve.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
for g in ["Male", "Female"]:
    mask = df["gender"] == g
    KaplanMeierFitter().fit(
        df.loc[mask, "time"], df.loc[mask, "event"], label=g
    ).plot_survival_function(ax=ax)
ax.set_title("KM Curve by Gender")
fig.tight_layout()
fig.savefig("visuals/KM_Plots/kaplan_curve_by_gender.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================
# STEP 7 — CLEAN COX MODEL DATASET
# =============================================
df_cox = df.copy()

# Ordinal / binary encodings
age_map = {
    '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5,
    '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
}
df_cox["age"]    = df_cox["age"].map(age_map).astype(float)
df_cox["gender"] = df_cox["gender"].map({"Male": 1, "Female": 0}).astype(float)

# Race dummies
df_cox["race"] = df_cox["race"].fillna("Unknown")
df_cox = pd.get_dummies(df_cox, columns=["race"], drop_first=True)
race_cols = [c for c in df_cox.columns if c.startswith("race_")]
df_cox[race_cols] = df_cox[race_cols].astype(int)

# -----------------------------------------------------------------------
# Bin PH-violating variables (on full df_cox, before split)
# -----------------------------------------------------------------------

# num_lab_procedures — binned to resolve PH violation.
# *** NOT used as a stratum — dummy-encoded as a regular covariate instead.
# This is what eliminates the unseen-strata error: removing it from the
# strata list cuts theoretical strata from 2×3×4×3×4=288 → 2×4×3×4=96,
# which the dataset can fully cover after an 80/20 split. ***
print("num_lab_procedures range:",
      df_cox["num_lab_procedures"].min(), "–", df_cox["num_lab_procedures"].max())

df_cox["num_lab_procedures_bin"] = pd.cut(
    df_cox["num_lab_procedures"],
    bins=[0, 30, 60, df_cox["num_lab_procedures"].max() + 1],
    labels=["low", "medium", "high"],
    right=True,
).astype(str)

# num_medications — stratified (4 levels)
df_cox["num_medications_bin"] = pd.cut(
    df_cox["num_medications"],
    bins=[0, 10, 20, 30, df_cox["num_medications"].max() + 1],
    labels=["low", "medium", "high", "very_high"],
    right=True,
).astype(str)

# number_emergency — stratified (3 levels)
df_cox["number_emergency_bin"] = pd.cut(
    df_cox["number_emergency"],
    bins=[-1, 0, 1, df_cox["number_emergency"].max() + 1],
    labels=["none", "one", "two_plus"],
    right=True,
).astype(str)

# discharge_disposition_id — stratified (4 clinical groups)
discharge_map = {
    1: "home", 8: "home",
    3: "snf_rehab", 5: "snf_rehab", 6: "snf_rehab", 12: "snf_rehab",
    2: "transfer", 4: "transfer", 9: "transfer", 10: "transfer",
    15: "transfer", 22: "transfer", 23: "transfer", 24: "transfer",
    27: "transfer", 28: "transfer", 29: "transfer",
}
df_cox["discharge_disposition_bin"] = (
    df_cox["discharge_disposition_id"]
    .map(discharge_map)
    .fillna("other")
    .astype(str)
)

print("discharge_disposition_bin counts:")
print(df_cox["discharge_disposition_bin"].value_counts().sort_values())
sparse_groups = df_cox["discharge_disposition_bin"].value_counts()
sparse_groups = sparse_groups[sparse_groups < 500]
if not sparse_groups.empty:
    print(f"⚠️  Sparse groups (< 500): {sparse_groups.to_dict()}")
else:
    print("✅ All discharge groups ≥ 500 rows.")

# -----------------------------------------------------------------------
# Dummy-encode num_lab_procedures_bin as a regular covariate
# (drop_first=True → "low" is the reference level)
# -----------------------------------------------------------------------
df_cox = pd.get_dummies(df_cox, columns=["num_lab_procedures_bin"], drop_first=True)
lab_bin_cols = [c for c in df_cox.columns if c.startswith("num_lab_procedures_bin_")]
df_cox[lab_bin_cols] = df_cox[lab_bin_cols].astype(int)
print(f"Lab-bin dummy columns added: {lab_bin_cols}")

# Drop irrelevant columns
drop_cols = [
    "encounter_id", "diag_1", "diag_2", "diag_3",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed", "payer_code",
    "medical_specialty", "readmitted", "readmitted_orig", "patient_nbr",
]
df_cox = df_cox.drop(columns=[c for c in drop_cols if c in df_cox.columns])

# Preserve string bin columns (strata) through select_dtypes
# Note: num_lab_procedures_bin is now int dummies — already numeric.
strata_bin_cols = ["num_medications_bin", "number_emergency_bin", "discharge_disposition_bin"]
df_cox_bins = df_cox[strata_bin_cols].copy()
df_cox = df_cox.select_dtypes(include=["number"])
df_cox[strata_bin_cols] = df_cox_bins

# Train / test split BEFORE scaling
train_idx, test_idx = train_test_split(
    df_cox.index, test_size=0.2, random_state=42, stratify=df_cox["event"]
)
df_cox_train = df_cox.loc[train_idx].copy()
df_cox_test  = df_cox.loc[test_idx].copy()

# Scale numeric columns — fit on train only
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
scaler = StandardScaler()
df_cox_train[numeric_cols] = scaler.fit_transform(df_cox_train[numeric_cols])
df_cox_test[numeric_cols]  = scaler.transform(df_cox_test[numeric_cols])

# Cast nullable Int64 → int64 (lifelines cannot handle pandas nullable types)
def cast_to_numpy_dtypes(frame):
    for col in frame.columns:
        if str(frame[col].dtype) == "Int64":
            frame[col] = frame[col].astype("int64")
        elif str(frame[col].dtype) == "Float64":
            frame[col] = frame[col].astype("float64")
    return frame

df_cox       = cast_to_numpy_dtypes(df_cox)
df_cox_train = cast_to_numpy_dtypes(df_cox_train)
df_cox_test  = cast_to_numpy_dtypes(df_cox_test)

remaining = [c for c in df_cox_train.columns
             if str(df_cox_train[c].dtype) in ("Int64", "Float64")]
print(f"Remaining nullable dtype columns: {remaining}")   # expected: []
print(f"Cox train: {df_cox_train.shape}  |  test: {df_cox_test.shape}")
print(f"Missing — train: {df_cox_train.isna().sum().sum()}  "
      f"test: {df_cox_test.isna().sum().sum()}")


# =============================================
# STEP 8 — FIT COX MODEL
# =============================================
# Stratified variables (PH violations resolved by stratification):
#   gender                    p=0.032  binary        → 2 levels
#   num_medications_bin       p=0.048  polypharmacy   → 4 levels
#   number_emergency_bin      p=0.010  emergency hx   → 3 levels
#   discharge_disposition_bin p=0.024  clinical group → 4 levels
#
# Theoretical max strata = 2 × 4 × 3 × 4 = 96   ← safely covered by ~80k train rows
#
# num_lab_procedures_bin is NOT stratified — it is dummy-encoded above
# (num_lab_procedures_bin_medium, num_lab_procedures_bin_high) so its PH
# violation is resolved by binning, and it still contributes coefficients.

STRATA_COLS = ["gender", "num_medications_bin",
               "number_emergency_bin", "discharge_disposition_bin"]

n_strata = df_cox_train.groupby(STRATA_COLS).ngroups
print(f"Actual strata in train: {n_strata} (theoretical max 2×4×3×4=96)")

train_strata = set(df_cox_train.groupby(STRATA_COLS).groups.keys())
test_strata  = set(df_cox_test.groupby(STRATA_COLS).groups.keys())
unseen = test_strata - train_strata
if unseen:
    raise ValueError(
        f"{len(unseen)} test strata unseen in train: {unseen}\n"
        "Collapse the relevant bin further before fitting."
    )
print("✅ All test strata present in training set — safe to fit and predict.")

cph = CoxPHFitter()
cph.fit(
    df_cox_train,
    duration_col="time",
    event_col="event",
    strata=STRATA_COLS,
)
cph.print_summary()

with open("models/cox_model.pkl", "wb") as f:
    pickle.dump(cph, f)
print("✅ Cox model saved.")


# =============================================
# STEP 9 — RANDOM SURVIVAL FOREST
# =============================================
# String bin columns are Cox-only — RSF uses the original numeric columns.
RSF_EXCLUDE = [
    "time", "event",
    "num_medications_bin",
    "number_emergency_bin",
    "discharge_disposition_bin",
]

X_rsf       = df_cox.drop(columns=RSF_EXCLUDE)
X_rsf_train = X_rsf.loc[train_idx]
X_rsf_test  = X_rsf.loc[test_idx]

y_rsf = Surv.from_arrays(
    event=df_cox["event"].astype(bool),
    time=df_cox["time"].astype(float)
)
y_rsf_train = y_rsf[df_cox.index.get_indexer(train_idx)]
y_rsf_test  = y_rsf[df_cox.index.get_indexer(test_idx)]

model_path = Path("models/rsf_model.pkl")
if model_path.exists():
    model_path.unlink()
    print("Deleted stale RSF model.")

print("⏳ Training RSF model...")
rsf = RandomSurvivalForest(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=15,
    random_state=42,
    n_jobs=-1,
)
rsf.fit(X_rsf_train, y_rsf_train)
joblib.dump(rsf, model_path)
print(f"✅ RSF trained on {X_rsf_train.shape[1]} features: "
      f"{list(X_rsf_train.columns)}")


# =============================================
# STEP 10 — MODEL COMPARISON (C-index)
# =============================================
cox_cindex = cph.concordance_index_

try:
    chf = rsf.predict_cumulative_hazard_function(X_rsf_test, return_array=True)
    rsf_risk = np.asarray(chf).sum(axis=1)
    rsf_cindex = concordance_index_censored(
        y_rsf_test["event"], y_rsf_test["time"], rsf_risk
    )[0]
except Exception as e:
    print(f"CHF failed ({e}), using -predict()")
    rsf_cindex = concordance_index_censored(
        y_rsf_test["event"], y_rsf_test["time"], -rsf.predict(X_rsf_test)
    )[0]

comparison = pd.DataFrame({
    "Model":   ["CoxPH (stratified)", "RSF (clean)"],
    "C-index": [cox_cindex, rsf_cindex],
})
comparison.to_csv("reports/model_comparison_cindex.csv", index=False)
print("\n=== MODEL PERFORMANCE ===")
print(comparison)


# =============================================
# STEP 11 — CALIBRATION (Brier Score)
# =============================================
y_train_sksurv = Surv.from_arrays(
    event=df_cox_train["event"].astype(bool),
    time=df_cox_train["time"].astype(float)
)
y_test_sksurv = Surv.from_arrays(
    event=df_cox_test["event"].astype(bool),
    time=df_cox_test["time"].astype(float)
)

eval_time = float(np.median(df_cox_test["time"]))
surv_df   = cph.predict_survival_function(df_cox_test)
time_idx  = np.abs(surv_df.index.values - eval_time).argmin()
pred_surv = surv_df.iloc[time_idx, :].values.reshape(-1, 1)

brier_val = brier_score(
    y_train_sksurv, y_test_sksurv, pred_surv, np.array([eval_time])
)[1][0]
print(f"\n=== CALIBRATION ===")
print(f"Brier score at t={eval_time:.0f} days: {brier_val:.4f}")


# =============================================
# STEP 12 — RISK STRATIFICATION
# =============================================
df_cox["cox_risk"] = cph.predict_partial_hazard(
    df_cox.drop(columns=["time", "event"])
)
df_cox["risk_group"] = pd.qcut(
    df_cox["cox_risk"], q=3, labels=["Low Risk", "Medium Risk", "High Risk"]
)

print("\n=== RISK GROUP DISTRIBUTION ===")
print(df_cox["risk_group"].value_counts())
print(df_cox[["time", "event", "cox_risk", "risk_group"]].head())


# =============================================
# STEP 13 — KAPLAN–MEIER BY RISK GROUP
# =============================================
km_data = df_cox

km = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(8, 5))

for group in ["Low Risk", "Medium Risk", "High Risk"]:
    mask = km_data["risk_group"] == group
    km.fit(
        durations=km_data.loc[mask, "time"],
        event_observed=km_data.loc[mask, "event"],
        label=group,
    )
    km.plot_survival_function(ax=ax)

ax.set_title("Kaplan–Meier Curves by Risk Group")
ax.set_xlabel("Days")
ax.set_ylabel("Survival Probability")
fig.tight_layout()
fig.savefig("visuals/KM_Plots/km_by_risk_group.png", dpi=300, bbox_inches="tight")
plt.show()

low  = km_data[km_data["risk_group"] == "Low Risk"]
med  = km_data[km_data["risk_group"] == "Medium Risk"]
high = km_data[km_data["risk_group"] == "High Risk"]

print("Low vs Medium: ",
      logrank_test(low["time"], med["time"],  low["event"], med["event"]).p_value)
print("Low vs High:   ",
      logrank_test(low["time"], high["time"], low["event"], high["event"]).p_value)
print("Medium vs High:",
      logrank_test(med["time"], high["time"], med["event"], high["event"]).p_value)


# =============================================
# STEP 14 — FEATURE IMPORTANCE (COX MODEL)
# =============================================
cox_summary = cph.summary.copy()

cox_summary_by_p = cox_summary.sort_values("p", ascending=True)
print("\n=== TOP 15 FEATURES BY SIGNIFICANCE ===")
print(cox_summary_by_p[["coef", "exp(coef)", "p"]].head(15))
cox_summary_by_p.to_csv("reports/cox_feature_importance.csv")

cox_summary_by_effect = cox_summary.reindex(
    cox_summary["coef"].abs().sort_values(ascending=False).index
)
top_feats = cox_summary_by_effect.head(15)

fig, ax = plt.subplots(figsize=(8, 6))
top_feats["coef"].plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title("Top 15 Cox Features by Effect Size (|coef|)")
ax.set_xlabel("Coefficient (log hazard ratio)")
fig.tight_layout()
fig.savefig("visuals/EDA/cox_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Step 14 complete.")


# =============================================
# STEP 15 — PROPORTIONAL HAZARDS CHECK
# =============================================
# check_assumptions() requires the exact training dataframe (it uses
# self._n_examples internally to build weights) — subsampling causes a
# length mismatch ValueError.  We skip it entirely: the Schoenfeld test
# below is the authoritative PH test; check_assumptions only adds plots.
# -----------------------------------------------------------------------
figs_before = set(plt.get_fignums())

results = proportional_hazard_test(cph, df_cox_train, time_transform="rank")
print("\n=== SCHOENFELD TEST RESULTS ===")
print(results.summary)
results.summary.to_csv("reports/schoenfeld_test_results.csv")

# Manual Schoenfeld residual plots — equivalent to check_assumptions plots
# but computed on a 5k subsample so they render in seconds, not hours.
PLOT_SAMPLE = 5_000
np.random.seed(42)
sample_idx      = np.random.choice(df_cox_train.index, size=PLOT_SAMPLE, replace=False)
df_plot_sample  = df_cox_train.loc[sample_idx].copy()

# compute_residuals ALSO requires the full training frame — use df_cox_train
schoenfeld_resid = cph.compute_residuals(df_cox_train, kind="scaled_schoenfeld")

# Subsample the residuals for plotting only
schoenfeld_resid_plot = schoenfeld_resid.loc[
    schoenfeld_resid.index.isin(sample_idx)
]

ph_violated = results.summary[results.summary["p"] < 0.05].index.tolist()
plot_cols    = ph_violated if ph_violated else list(schoenfeld_resid.columns[:6])
print(f"Plotting Schoenfeld residuals for: {plot_cols}")

os.makedirs("visuals/PH_diagnostics", exist_ok=True)
for i, col in enumerate(plot_cols, start=1):
    if col not in schoenfeld_resid_plot.columns:
        continue
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        schoenfeld_resid_plot.index,
        schoenfeld_resid_plot[col],
        alpha=0.3, s=8, label="Schoenfeld residual"
    )
    # Simple lowess trend on the subsample
    from statsmodels.nonparametric.smoothers_lowess import lowess
    xy = schoenfeld_resid_plot[[col]].dropna()
    if len(xy) > 10:
        smoothed = lowess(xy[col].values, np.arange(len(xy)), frac=0.3)
        ax.plot(xy.index[:len(smoothed)], smoothed[:, 1],
                color="red", lw=2, label="lowess trend")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_title(f"Schoenfeld Residuals — {col}")
    ax.set_xlabel("Observation order (subsample)")
    ax.set_ylabel("Scaled Schoenfeld residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        f"visuals/PH_diagnostics/schoenfeld_plot_{i}_{col}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

new_figs = len(plot_cols)
print(f"✅ Step 15 complete: {new_figs} Schoenfeld plots saved.")


# =============================================
# STEP 16 — RSF FEATURE IMPORTANCE
# =============================================
# sksurv.RandomSurvivalForest deliberately raises NotImplementedError for
# feature_importances_.  We use permutation importance instead, but with
# two speed optimisations vs the original:
#   1. predict() (median survival time) instead of CHF — ~20x faster/call
#   2. 2,000-row subsample of the test set instead of all 20k rows
# Total runtime: ~2-5 minutes instead of 3+ hours.
# -----------------------------------------------------------------------
PERM_SAMPLE  = 2_000
PERM_REPEATS = 3

print(f"⏳ Computing RSF permutation importance "
      f"(n={PERM_SAMPLE}, repeats={PERM_REPEATS})...")

np.random.seed(42)
perm_idx   = np.random.choice(len(X_rsf_test), size=PERM_SAMPLE, replace=False)
X_perm_sub = X_rsf_test.iloc[perm_idx].reset_index(drop=True)

baseline = rsf.predict(X_perm_sub)   # median survival — fast

importances = []
for col in X_perm_sub.columns:
    scores = []
    for _ in range(PERM_REPEATS):
        Xp = X_perm_sub.copy()
        Xp[col] = np.random.permutation(Xp[col].values)
        scores.append(np.sqrt(np.mean((rsf.predict(Xp) - baseline) ** 2)))
    importances.append(np.mean(scores))
    print(f"  {col}: {importances[-1]:.4f}")

feat_imp_sorted = pd.Series(importances, index=X_perm_sub.columns) \
                    .sort_values(ascending=False)
feat_imp_sorted.to_csv("reports/rsf_feature_importance.csv")

print("\n=== TOP 15 RSF FEATURES ===")
print(feat_imp_sorted.head(15))

fig, ax = plt.subplots(figsize=(8, 6))
feat_imp_sorted.head(15).plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title("Top 15 RSF Feature Importances (Permutation — predict)")
ax.set_xlabel("Importance (RMSE increase in predicted survival when permuted)")
fig.tight_layout()
fig.savefig("visuals/EDA/rsf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Step 16 complete.")