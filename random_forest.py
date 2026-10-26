"""Random forest regression for log(1 + frac_raised).

Evaluation uses nested cross-validation: the outer loop (5-fold x 5 repeats)
estimates performance on unseen campaigns, and the inner loop (5-fold grid
search) tunes hyperparameters using only the outer training fold. With
n = 224 a single train/test split would be too noisy to trust.

Results (runtime ~6.5 min):
  - Predictive power is weak. Test R^2 averages 0.04 (sd 0.14) on the log
    scale vs 0.22 on training folds, so the forest explains little of why one
    campaign out-raises another and still overfits slightly.
  - Typical (median) error is 0.77x the goal; mean error is 1.51x because a
    handful of campaigns raised 10-26x their goal and nothing in the data
    predicts them.
  - As a reached-goal / missed-goal call it is right 50% of the time, below
    the 61% from guessing every campaign succeeds. Back-transforming log
    predictions pulls them toward the middle, so the model under-calls success.
  - Three variables carry the signal: pre-order perk, IoT product and being
    based in California. Goal size, all-or-nothing funding, category and year
    add nothing on held-out data.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)

DATA = "crowdfunding_clean.csv"
NUMERIC = ["log_goal", "pre_order_perk", "tech_campaign", "sbiz_campaign",
           "iot_plus", "fixed_fund", "year_start"]
CATEGORICAL = ["state_grp"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42

# Shallow trees and large leaves keep the forest from memorising individual
# campaigns. n_estimators is fixed: more trees only reduce variance.
PARAM_GRID = {
    "randomforestregressor__max_depth": [2, 3, 4, 6, None],
    "randomforestregressor__min_samples_leaf": [5, 10, 20],
    "randomforestregressor__max_features": [0.33, 0.5, 1.0],
}


def preprocessor():
    # Imputation is fitted inside each fold, so test rows never inform the fill.
    # Trees don't need feature scaling.
    return make_column_transformer(
        (SimpleImputer(strategy="most_frequent"), NUMERIC),
        (OneHotEncoder(drop=["Other"], sparse_output=False), CATEGORICAL),
    )


def rf_search():
    rf = make_pipeline(preprocessor(),
                       RandomForestRegressor(n_estimators=300, random_state=SEED))
    return GridSearchCV(rf, PARAM_GRID, cv=KFold(5, shuffle=True, random_state=SEED),
                        scoring="neg_root_mean_squared_error", n_jobs=-1)


def scores(y_log, pred_log):
    y, pred = np.expm1(y_log), np.expm1(pred_log)
    return {
        "r2_log": r2_score(y_log, pred_log),
        "rmse_log": root_mean_squared_error(y_log, pred_log),
        "mae_frac": mean_absolute_error(y, pred),
        "median_ae_frac": np.median(np.abs(y - pred)),
        "goal_accuracy": np.mean((y >= 1) == (pred >= 1)),
    }


def main():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["log_frac_raised"]

    outer = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    results, chosen, importances = [], [], []
    for fold, (tr, te) in enumerate(outer.split(X)):
        m = rf_search().fit(X.iloc[tr], y.iloc[tr])
        results.append({"train_r2_log": r2_score(y.iloc[tr], m.predict(X.iloc[tr])),
                        **scores(y.iloc[te], m.predict(X.iloc[te]))})
        chosen.append(m.best_params_)
        pi = permutation_importance(m.best_estimator_, X.iloc[te], y.iloc[te],
                                    n_repeats=20, random_state=SEED, scoring="r2")
        importances.append(pi.importances_mean)
        print(f"outer fold {fold + 1}/{outer.get_n_splits()} done", flush=True)

    # Test R^2 0.044 +/- 0.139 vs train 0.219: low signal, mild overfitting,
    # and high variance across folds.
    summary = pd.DataFrame(results).agg(["mean", "std"]).T.round(3)
    print("\n=== Nested CV performance (25 outer folds) ===")
    print(summary.to_string())
    print(f"(for reference: {np.mean(df['frac_raised'] >= 1):.0%} of campaigns reached their goal)")

    # max_depth=2 won 13 of 25 folds and min_samples_leaf=5 won 18: the inner
    # CV prefers very simple trees, a sign there is little nonlinear structure
    # for a forest to exploit at this sample size.
    print("\n=== Hyperparameters chosen by inner CV (count of outer folds) ===")
    picked = pd.DataFrame(chosen)
    picked.columns = [c.split("__")[1] for c in picked.columns]
    for col in picked:
        print(f"{col}: {picked[col].astype(str).value_counts().to_dict()}")

    # Permutation importance is measured on held-out folds, unlike the forest's
    # built-in impurity importance, which inflates continuous variables. Only
    # pre_order_perk (helps in 100% of folds), iot_plus (84%) and state_grp
    # (80%) matter. log_goal ranks last: it helps in only 36% of folds.
    imp = pd.DataFrame(importances, columns=FEATURES)
    imp_summary = pd.DataFrame({"mean_drop_in_r2": imp.mean(), "std": imp.std(),
                                "share_of_folds_positive": (imp > 0).mean()})
    print("\n=== Permutation importance on held-out folds ===")
    print(imp_summary.sort_values("mean_drop_in_r2", ascending=False).round(4).to_string())

    # Final model: tune on all data, then show direction of each effect as the
    # median predicted fraction raised when a feature is set to each value.
    # Final pick: max_depth=4, max_features=0.5, min_samples_leaf=10.
    # Effects: pre-order perk 0.71x -> 1.11x goal, IoT 0.83x -> 1.46x,
    # California 1.16x vs ~0.82x elsewhere. Tech, small business and
    # all-or-nothing funding barely move the prediction, and goal size is flat
    # apart from a small bump at $1,000.
    final = rf_search().fit(X, y)
    print("\n=== Final model (tuned on all 224 campaigns) ===")
    print("best params:", {k.split("__")[1]: v for k, v in final.best_params_.items()})
    print(f"inner-CV RMSE (log scale): {-final.best_score_:.3f}")

    print("\n=== Median predicted fraction of goal raised, by feature value ===")
    for col in ["pre_order_perk", "tech_campaign", "sbiz_campaign", "iot_plus", "fixed_fund"]:
        row = {v: np.median(np.expm1(final.predict(X.assign(**{col: v})))) for v in (0, 1)}
        print(f"{col:15s} 0 -> {row[0]:.2f}x   1 -> {row[1]:.2f}x")
    for v in ["California", "New York", "Other"]:
        print(f"state_grp={v:10s} -> {np.median(np.expm1(final.predict(X.assign(state_grp=v)))):.2f}x")
    for goal in [1000, 5000, 10000, 25000, 100000]:
        p = np.median(np.expm1(final.predict(X.assign(log_goal=np.log(goal)))))
        print(f"goal ${goal:>7,} -> {p:.2f}x")


if __name__ == "__main__":
    main()
