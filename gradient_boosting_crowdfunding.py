"""
Gradient Boosting regression on the group's agreed crowdfunding dataset (crowdfunding_clean.csv).

Follows the README methodology:
  * 224 unique campaigns, 7 values left missing (5 fixed_fund, 2 iot_plus)
  * missing values are filled with the TRAINING-FOLD mode inside each
    cross-validation split (SimpleImputer in the pipeline), so test rows
    never influence the fill
  * state_grp -> California / New York dummies (Other = baseline)
  * target: log_frac_raised = log(1 + frac_raised); metrics are reported on
    the ORIGINAL frac_raised scale after back-transforming

Evaluation: nested CV. Inner 5-fold randomized search tunes hyperparameters;
outer 5-fold CV repeated twice (10 held-out test sets) scores them.

Usage:  python3 gradient_boosting_crowdfunding.py crowdfunding_clean.csv
Output: gradient_boosting_results.xlsx, gradient_boosting_importance.png
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, KFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "crowdfunding_clean.csv"

# ------------------------------------------------------------------ data
df = pd.read_csv(IN_PATH)
df["state_CA"] = (df["state_grp"] == "California").astype(int)
df["state_NY"] = (df["state_grp"] == "New York").astype(int)
features = ["log_goal", "pre_order_perk", "tech_campaign", "sbiz_campaign",
            "iot_plus", "fixed_fund", "year_start", "state_CA", "state_NY"]
X = df[features].to_numpy(float)          # contains NaN; imputed inside the pipeline
y_log = df["log_frac_raised"].to_numpy(float)
y_raw = df["frac_raised"].to_numpy(float)
print(f"n = {len(df)}, features = {len(features)}, missing cells = {int(np.isnan(X).sum())}")

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("gb", GradientBoostingRegressor(loss="squared_error", random_state=SEED)),
])
param_dist = {
    "gb__n_estimators": [100, 200, 300],
    "gb__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "gb__max_depth": [1, 2, 3],
    "gb__min_samples_leaf": [3, 5, 8, 12],
    "gb__subsample": [0.6, 0.8, 1.0],
    "gb__max_features": [0.5, 0.8, 1.0],
}


def metrics(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


outer = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
rows, params, imps, imps_log = [], [], [], []
for k, (tr, te) in enumerate(outer.split(X)):
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED + k)
    search = RandomizedSearchCV(clone(pipe), param_dist, n_iter=12, cv=inner,
                                scoring="neg_mean_squared_error",
                                random_state=SEED + k, n_jobs=1)
    search.fit(X[tr], y_log[tr])
    best = search.best_estimator_
    pred = np.clip(np.expm1(best.predict(X[te])), 0, None)
    m = metrics(y_raw[te], pred)
    mean_b = metrics(y_raw[te], np.full(len(te), y_raw[tr].mean()))
    med_mae = mean_absolute_error(y_raw[te], np.full(len(te), np.median(y_raw[tr])))
    # log-scale accuracy too (the scale the model is actually fitted on)
    m_log = metrics(y_log[te], best.predict(X[te]))
    base_log = metrics(y_log[te], np.full(len(te), y_log[tr].mean()))
    rows.append({"fold": k, **m, "baseline_mean_RMSE": mean_b["RMSE"],
                 "baseline_mean_MAE": mean_b["MAE"], "baseline_median_MAE": med_mae,
                 "log_RMSE": m_log["RMSE"], "log_R2": m_log["R2"],
                 "log_baseline_RMSE": base_log["RMSE"]})
    params.append({k_.replace("gb__", ""): v for k_, v in search.best_params_.items()})
    pi = permutation_importance(best, X[te], y_log[te], n_repeats=10,
                                random_state=SEED, scoring="r2")
    imps.append(pi.importances_mean)
    print(f"outer fold {k + 1}/10 done")

folds = pd.DataFrame(rows)
summary = pd.DataFrame([{
    "RMSE (original scale)": folds.RMSE.mean(), "RMSE sd": folds.RMSE.std(),
    "MAE (original scale)": folds.MAE.mean(), "MAE sd": folds.MAE.std(),
    "R2 (original scale)": folds.R2.mean(), "R2 sd": folds.R2.std(),
    "Baseline RMSE (mean)": folds.baseline_mean_RMSE.mean(),
    "Baseline MAE (mean)": folds.baseline_mean_MAE.mean(),
    "Baseline MAE (median)": folds.baseline_median_MAE.mean(),
    "RMSE improvement vs mean baseline": 1 - folds.RMSE.mean() / folds.baseline_mean_RMSE.mean(),
    "MAE improvement vs median baseline": 1 - folds.MAE.mean() / folds.baseline_median_MAE.mean(),
    "Log-scale RMSE": folds.log_RMSE.mean(),
    "Log-scale baseline RMSE": folds.log_baseline_RMSE.mean(),
    "Log-scale R2": folds.log_R2.mean(), "Log-scale R2 sd": folds.log_R2.std(),
}]).T.reset_index()
summary.columns = ["Metric (mean over 10 outer folds)", "Value"]
print(summary.round(3).to_string(index=False))

hp = pd.DataFrame(params)
hp_summary = pd.DataFrame([{
    "hyperparameter": c,
    "most frequent choice": hp[c].value_counts().index[0],
    "chosen in x of 10 folds": int(hp[c].value_counts().iloc[0]),
    "values seen": ", ".join(str(v) for v in sorted(hp[c].unique()))} for c in hp.columns])

imps = np.array(imps)
imp = pd.DataFrame({"feature": features,
                    "importance (mean drop in R2, log scale)": imps.mean(axis=0),
                    "sd across folds": imps.std(axis=0)}
                   ).sort_values("importance (mean drop in R2, log scale)", ascending=False)
print(imp.round(4).to_string(index=False))

# final model on all data using the most common hyperparameters (for the record)
final_params = {c: hp[c].mode()[0] for c in hp.columns}
final = clone(pipe).set_params(**{f"gb__{k}": v for k, v in final_params.items()}).fit(X, y_log)
final_imp = pd.DataFrame({"feature": features,
                          "built-in (impurity) importance":
                              final.named_steps["gb"].feature_importances_}
                         ).sort_values("built-in (impurity) importance", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
top = imp.iloc[::-1]
ax.barh(top.feature, top["importance (mean drop in R2, log scale)"],
        xerr=top["sd across folds"], color="#1F3864", ecolor="#999999")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Permutation importance (drop in R2, held-out folds)")
ax.set_title("Gradient boosting: variable importance")
plt.tight_layout()
plt.savefig("gradient_boosting_importance.png", dpi=150)

notes = pd.DataFrame({"Notes": [
    "Data: crowdfunding_clean.csv (224 unique campaigns; 7 missing cells left for the pipeline, as in the README).",
    "Pipeline: SimpleImputer(most_frequent) fitted on training folds only -> GradientBoostingRegressor. Tuning therefore never sees test rows, including for imputation.",
    "Target: log_frac_raised. Predictions are back-transformed with expm1 and clipped at 0 for the original-scale metrics.",
    "Nested CV: inner 5-fold RandomizedSearchCV (12 candidates, scored on MSE) inside each of 10 outer folds (5-fold x 2 repeats).",
    "Search ranges favour simple models for n = 224: depth 1-3, min_samples_leaf 3-12, learning_rate 0.01-0.1, 100-300 trees, subsample/max_features < 1 for variance reduction.",
    "Baselines: training-fold mean (RMSE-optimal constant) and median (MAE-optimal constant). Log-scale baseline is the training-fold mean of log_frac_raised.",
    "Importance: permutation importance on held-out folds. tech_campaign and sbiz_campaign are mutually exclusive, so their effects are measured against the 'neither' group; iot_plus only varies within tech campaigns.",
    "Importance shows predictive relevance, not causation.",
]})
with pd.ExcelWriter("gradient_boosting_results.xlsx", engine="openpyxl") as xw:
    summary.to_excel(xw, sheet_name="Summary", index=False)
    folds.to_excel(xw, sheet_name="Outer folds", index=False)
    hp_summary.to_excel(xw, sheet_name="Hyperparameters", index=False)
    imp.to_excel(xw, sheet_name="Permutation importance", index=False)
    final_imp.to_excel(xw, sheet_name="Final model importance", index=False)
    notes.to_excel(xw, sheet_name="Notes", index=False)
    for ws in xw.book.worksheets:
        for col in ws.columns:
            w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(12, w + 2), 110)
print("Saved gradient_boosting_results.xlsx and gradient_boosting_importance.png")
