"""
CROWDFUNDING PROJECT - ALL MODELS IN ONE FILE
What drives success in crowdfunding campaigns?

Combines the group's four models, each copied from its source file:
  Model 1  OLS regression        OLS_crowdfunding_clean.ipynb       (Arnav Gannamani)
  Model 2  GAM                   GAM_crowdfunding.py                (Jairaj Narang)
  Model 3  Random forest         random_forest.py                   (Anirudh Gopalan)
  Model 4  Gradient boosting     gradient_boosting_crowdfunding.py  (Ethan Dsouza)

Every model receives the same DataFrame, produced once by clean.clean() and
passed through the same CSV formatting as crowdfunding_clean.csv, which each
source file reads. The data is therefore byte-for-byte what the individual
files see, and results match them exactly. The model code is unchanged apart
from the data-loading line and the notebook-to-script adaptations marked
"[combined]".

Usage:
  python3 all_models.py                 # run all four (~15 min; RF ~6.5 min)
  python3 all_models.py ols gam         # run a subset: ols, gam, rf, gb

Needs: pandas, numpy, scipy, scikit-learn, statsmodels, seaborn, matplotlib, openpyxl
Writes: GAM figures (Figure1/2 PNGs), gradient_boosting_results.xlsx,
        gradient_boosting_importance.png
"""
import io
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # [combined] scripts save figures; notebook plt.show() becomes a no-op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean import clean


def display(obj):
    # [combined] Jupyter's display() for script use; show full tables, not "..." columns.
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(obj)


# =====================================================================
# MODEL 1 - OLS REGRESSION (from OLS_crowdfunding_clean.ipynb)
# =====================================================================
def run_ols(df):
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import scipy.stats as stats

    # Set clean visual styles for charts
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 120

    # 1. Load the cleaned dataset
    # [combined] was: DATA_PATH = "crowdfunding_clean.csv"; df = pd.read_csv(DATA_PATH)

    print(f"Dataset successfully loaded: {df.shape[0]} campaigns, {df.shape[1]} columns")
    display(df.head())
    display(df.info())

    # --- Step 1: missing values and summary statistics
    # Check if there are any missing values left
    missing_counts = df.isna().sum()
    print("Missing values per column:")
    print(missing_counts[missing_counts > 0])

    # Fill the few remaining blanks with 0 (the most common value)
    # 2 missing in iot_plus, 5 missing in fixed_fund
    df_model = df.copy()
    df_model['iot_plus'] = df_model['iot_plus'].fillna(0.0)
    df_model['fixed_fund'] = df_model['fixed_fund'].fillna(0.0)

    print("\nMissing values after filling:", df_model.isna().sum().sum())

    # Display basic summary statistics
    summary_stats = df_model[['frac_raised', 'log_frac_raised', 'amt_goal', 'log_goal',
                              'pre_order_perk', 'tech_campaign', 'sbiz_campaign',
                              'iot_plus', 'fixed_fund']].describe().T
    summary_stats['median'] = df_model[['frac_raised', 'log_frac_raised', 'amt_goal', 'log_goal',
                                        'pre_order_perk', 'tech_campaign', 'sbiz_campaign',
                                        'iot_plus', 'fixed_fund']].median()
    display(summary_stats[['mean', 'std', 'min', 'median', 'max']])  # [combined] notebook showed the last expression

    # --- Step 2-3: distributions of raw vs log target and goal
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw Target: frac_raised
    sns.histplot(df_model['frac_raised'], kde=True, ax=axes[0, 0], color='#2b5c8f', bins=30)
    axes[0, 0].set_title(f"Raw Target: frac_raised (Skew = {df_model['frac_raised'].skew():.2f})", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Fraction of Goal Raised (Raw)")
    axes[0, 0].set_ylabel("Number of Campaigns")

    # 2. Transformed Target: log(1 + frac_raised)
    sns.histplot(df_model['log_frac_raised'], kde=True, ax=axes[0, 1], color='#2a9d8f', bins=25)
    axes[0, 1].set_title(f"Log Target: log(1 + frac_raised) (Skew = {df_model['log_frac_raised'].skew():.2f})", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("log(1 + frac_raised)")
    axes[0, 1].set_ylabel("Number of Campaigns")

    # 3. Raw Funding Goal
    sns.histplot(df_model['amt_goal'], kde=True, ax=axes[1, 0], color='#e76f51', bins=30)
    axes[1, 0].set_title(f"Raw Goal Amount ($) (Skew = {df_model['amt_goal'].skew():.2f})", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Goal in Dollars ($)")
    axes[1, 0].set_ylabel("Number of Campaigns")

    # 4. Transformed Log Goal
    sns.histplot(df_model['log_goal'], kde=True, ax=axes[1, 1], color='#e9c46a', bins=25)
    axes[1, 1].set_title(f"Log Goal: log(amt_goal) (Skew = {df_model['log_goal'].skew():.2f})", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("log(amt_goal)")
    axes[1, 1].set_ylabel("Number of Campaigns")

    plt.tight_layout()
    plt.show()

    # --- Step 4: correlations
    # Correlation heatmap
    num_cols = ['log_frac_raised', 'log_goal', 'pre_order_perk', 'tech_campaign',
                'sbiz_campaign', 'iot_plus', 'fixed_fund']
    corr_matrix = df_model[num_cols].corr()

    plt.figure(figsize=(9, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-0.5, vmax=0.5,
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (How Features Relate to Each Other)", fontsize=13, fontweight='bold', pad=15)
    plt.show()

    # --- Step 5-6: four nested OLS specifications
    # Define the formulas for our 4 models
    f1 = "log_frac_raised ~ log_goal"
    f2 = "log_frac_raised ~ log_goal + pre_order_perk + tech_campaign + sbiz_campaign + iot_plus + fixed_fund"
    f3 = ("log_frac_raised ~ log_goal + pre_order_perk + tech_campaign + sbiz_campaign + "
          "iot_plus + fixed_fund + C(state_grp, Treatment(reference='Other')) + C(year_start)")
    f4 = ("log_frac_raised ~ log_goal + pre_order_perk * tech_campaign + sbiz_campaign + "
          "iot_plus + fixed_fund + C(state_grp, Treatment(reference='Other')) + C(year_start)")

    # Fit each model using Ordinary Least Squares
    m1 = smf.ols(f1, data=df_model).fit()
    m2 = smf.ols(f2, data=df_model).fit()
    m3 = smf.ols(f3, data=df_model).fit()
    m4 = smf.ols(f4, data=df_model).fit()

    # Build a clean comparison table
    comparison_df = pd.DataFrame({
        'Model 1 (Goal Only)': [m1.rsquared, m1.rsquared_adj, m1.fvalue, m1.f_pvalue, m1.aic, m1.bic, m1.df_model],
        'Model 2 (+ Features)': [m2.rsquared, m2.rsquared_adj, m2.fvalue, m2.f_pvalue, m2.aic, m2.bic, m2.df_model],
        'Model 3 (Full Model)': [m3.rsquared, m3.rsquared_adj, m3.fvalue, m3.f_pvalue, m3.aic, m3.bic, m3.df_model],
        'Model 4 (+ Interaction)': [m4.rsquared, m4.rsquared_adj, m4.fvalue, m4.f_pvalue, m4.aic, m4.bic, m4.df_model]
    }, index=['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-stat)', 'AIC (lower is better)', 'BIC (lower is better)', 'Num Features'])

    display(comparison_df.round(4))

    # --- Step 7: assumption checks
    # 1. Check for Multicollinearity using Variance Inflation Factor (VIF)
    # (Are features repeating the same information?)
    X_vif = df_model[['log_goal', 'pre_order_perk', 'tech_campaign', 'sbiz_campaign', 'iot_plus', 'fixed_fund']].copy()
    X_vif['CA'] = (df_model['state_grp'] == 'California').astype(float)
    X_vif['NY'] = (df_model['state_grp'] == 'New York').astype(float)
    X_vif['intercept'] = 1.0

    vif_table = pd.DataFrame({
        'Feature': X_vif.columns,
        'VIF Score': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    }).sort_values('VIF Score', ascending=False).reset_index(drop=True)

    print("--- Multicollinearity Check (VIF) ---")
    display(vif_table.round(2))

    # 2. Check for Unequal Error Spread (Heteroskedasticity) using the Breusch-Pagan Test
    bp_test = het_breuschpagan(m3.resid, m3.model.exog)
    bp_results = pd.Series({
        'Breusch-Pagan Test Stat': bp_test[0],
        'p-value': bp_test[1]
    })

    print("\n--- Breusch-Pagan Test for Heteroskedasticity ---")
    display(bp_results.to_frame('Value').round(5))

    # --- Step 8: Model 3 with HC3 robust standard errors
    # Run Model 3 using HC3 Robust Standard Errors
    m3_robust = smf.ols(f3, data=df_model).fit(cov_type='HC3')

    print(m3_robust.summary())

    # --- Step 9: 5-fold cross-validation
    # 5-Fold Cross Validation: Testing the model on unseen data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rmse, cv_mae, cv_r2 = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df_model), 1):
        train_data = df_model.iloc[train_idx]
        test_data = df_model.iloc[test_idx]

        # Train model on 4 folds
        model_fold = smf.ols(f3, data=train_data).fit()

        # Predict on the 1 held-out test fold
        predictions = model_fold.predict(test_data)

        # Calculate errors
        rmse = np.sqrt(mean_squared_error(test_data['log_frac_raised'], predictions))
        mae = mean_absolute_error(test_data['log_frac_raised'], predictions)
        r2 = r2_score(test_data['log_frac_raised'], predictions)

        cv_rmse.append(rmse)
        cv_mae.append(mae)
        cv_r2.append(r2)

    cv_results = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(1, 6)] + ['Average', 'Std Dev'],
        'RMSE (Root Mean Sq Error)': cv_rmse + [np.mean(cv_rmse), np.std(cv_rmse)],
        'MAE (Mean Absolute Error)': cv_mae + [np.mean(cv_mae), np.std(cv_mae)],
        'R-squared': cv_r2 + [np.mean(cv_r2), np.std(cv_r2)]
    })

    print("--- 5-Fold Cross-Validation (Out-of-Sample Test) ---")
    display(cv_results.round(4))

    # --- Step 10: diagnostic plots
    # 4-Panel Diagnostic Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Actual vs. Predicted Values
    fitted_vals = m3_robust.fittedvalues
    actual_vals = df_model['log_frac_raised']
    axes[0, 0].scatter(fitted_vals, actual_vals, alpha=0.6, color='#2b5c8f', edgecolors='k', linewidth=0.5)
    min_v, max_v = min(fitted_vals.min(), actual_vals.min()), max(fitted_vals.max(), actual_vals.max())
    axes[0, 0].plot([min_v, max_v], [min_v, max_v], '--', color='#d62828', linewidth=2, label='Perfect Fit Line (45°)')
    axes[0, 0].set_title("1. Actual vs. Predicted Values", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Predicted Funding (Log)")
    axes[0, 0].set_ylabel("Actual Funding (Log)")
    axes[0, 0].legend()

    # 2. Residuals vs. Fitted Values
    residuals = m3_robust.resid
    axes[0, 1].scatter(fitted_vals, residuals, alpha=0.6, color='#2a9d8f', edgecolors='k', linewidth=0.5)
    axes[0, 1].axhline(0, color='#d62828', linestyle='--', linewidth=2)
    lowess = sm.nonparametric.lowess(residuals, fitted_vals)
    axes[0, 1].plot(lowess[:, 0], lowess[:, 1], color='#e76f51', linewidth=2, label='Error Trend Line')
    axes[0, 1].set_title("2. Residuals (Errors) vs. Predicted Values", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Prediction Error")
    axes[0, 1].legend()

    # 3. Normal Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_color('#2b5c8f')
    axes[1, 0].get_lines()[0].set_alpha(0.7)
    axes[1, 0].get_lines()[1].set_color('#d62828')
    axes[1, 0].get_lines()[1].set_linewidth(2)
    axes[1, 0].set_title("3. Normal Q-Q Plot (Are Errors Normal?)", fontsize=12, fontweight='bold')

    # 4. Coefficient Plot with 95% Confidence Intervals
    coefs = m3_robust.params.drop(['Intercept', 'C(year_start)[T.2012]', 'C(year_start)[T.2013]'])
    conf = m3_robust.conf_int().loc[coefs.index]
    errors = coefs - conf[0]

    clean_labels = [
        'California (vs Other)', 'New York (vs Other)', 'Funding Goal (Log)',
        'Pre-Order Perk', 'Tech Campaign', 'Small Business', 'IoT Smart Tech', 'Fixed Funding'
    ]

    y_pos = np.arange(len(coefs))
    colors = ['#2a9d8f' if p < 0.05 else '#adb5bd' for p in m3_robust.pvalues.loc[coefs.index]]

    axes[1, 1].errorbar(coefs, y_pos, xerr=errors, fmt='o', color='#2b5c8f', ecolor=colors, elinewidth=2.5, capsize=4)
    axes[1, 1].axvline(0, color='#d62828', linestyle='--', linewidth=1.5)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(clean_labels, fontsize=10)
    axes[1, 1].set_title("4. Variable Importance (Green = Significant)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Impact on Funding (with 95% Confidence Interval)")

    plt.tight_layout()
    plt.show()


# =====================================================================
# MODEL 2 - GAM, PENALIZED CUBIC SPLINES (from GAM_crowdfunding.py)
# =====================================================================
from scipy import stats
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class PGAM:
    """Gaussian GAM: y = f(smooth cols) + linear terms; f = B-spline with difference penalty."""
    def __init__(self, lam=1.0, n_knots=10, degree=3, smooth_cols=("log_goal",)):
        self.lam, self.n_knots, self.degree = lam, n_knots, degree
        self.smooth_cols = list(smooth_cols)

    def _design(self, X, fit=False):
        if fit:
            self.spl_ = {c: SplineTransformer(n_knots=self.n_knots, degree=self.degree,
                         knots="quantile", extrapolation="constant",
                         include_bias=True).fit(X[[c]]) for c in self.smooth_cols}
            self.lin_cols_ = [c for c in X.columns if c not in self.smooth_cols]
        parts, k, self.blocks_ = [], 0, {}
        for c in self.smooth_cols:
            B = self.spl_[c].transform(X[[c]]); parts.append(B)
            self.blocks_[c] = (k, k + B.shape[1]); k += B.shape[1]
        parts.append(X[self.lin_cols_].values.astype(float))
        return np.hstack(parts)

    def _penalty(self, p):
        P = np.zeros((p, p))
        for c, (a, b) in self.blocks_.items():
            D = np.diff(np.eye(b - a), n=2, axis=0); P[a:b, a:b] = D.T @ D
        return P + 1e-8 * np.eye(p)

    def fit(self, X, y):
        y = np.asarray(y); A = self._design(X, fit=True)
        self.P_ = self._penalty(A.shape[1]); self.n_ = len(y)
        self.Ginv_ = np.linalg.inv(A.T @ A + self.lam * self.P_)
        self.coef_ = self.Ginv_ @ A.T @ y
        self.edf_total_ = np.trace(A @ self.Ginv_ @ A.T)
        res = y - A @ self.coef_
        self.rss_ = float(res @ res)
        self.sigma2_ = self.rss_ / (self.n_ - self.edf_total_)
        self.A_, self.X_train_ = A, X
        return self

    def predict(self, X):
        return self._design(X) @ self.coef_

    def smooth_edf(self, c="log_goal"):
        a, b = self.blocks_[c]
        F = self.Ginv_ @ self.A_.T @ self.A_
        return float(np.trace(F[a:b, a:b]))

    def partial(self, grid, c="log_goal"):
        """Centred partial effect of the smooth term with 95% CI."""
        a, b = self.blocks_[c]
        B = self.spl_[c].transform(pd.DataFrame({c: grid}))
        ft = self.spl_[c].transform(self.X_train_[[c]]) @ self.coef_[a:b]
        V = self.sigma2_ * self.Ginv_[a:b, a:b]
        se = np.sqrt(np.einsum("ij,jk,ik->i", B, V, B))
        return B @ self.coef_[a:b] - ft.mean(), 1.96 * se


def run_gam(clean_df):
    warnings.filterwarnings("ignore")
    OUT_DIR = "."

    # PART 1 - LOAD & CLEAN DATA
    raw = clean_df  # [combined] was: raw = pd.read_csv(DATA_PATH)
    df = raw.dropna(subset=["iot_plus", "fixed_fund"]).reset_index(drop=True)
    print(f"PART 1: {len(raw)} rows -> {len(df)} after dropping rows with missing iot_plus/fixed_fund")

    y = df["log_frac_raised"].values
    X = pd.get_dummies(
        df[["log_goal", "pre_order_perk", "tech_campaign", "sbiz_campaign", "iot_plus",
            "fixed_fund", "year_start", "state_grp"]].assign(year_start=lambda d: d.year_start.astype(str)),
        drop_first=True, dtype=float)      # baselines: year 2011, state = California

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    def cv_eval(make, Xd=X):
        r2, rm, ma = [], [], []
        for tr, te in cv.split(Xd):
            p = make().fit(Xd.iloc[tr], y[tr]).predict(Xd.iloc[te])
            r2.append(r2_score(y[te], p)); rm.append(mean_squared_error(y[te], p) ** .5)
            ma.append(mean_absolute_error(y[te], p))
        return dict(R2=np.mean(r2), R2_sd=np.std(r2), RMSE=np.mean(rm), MAE=np.mean(ma))

    # PART 2 - HYPERPARAMETER TUNING (n_knots x smoothing penalty lambda)
    rows = []
    for k in [5, 6, 8, 10, 12]:
        for lam in [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]:
            r = cv_eval(lambda: PGAM(lam=lam, n_knots=k)); r.update(n_knots=k, lam=lam); rows.append(r)
    tune = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    K, LAM = int(tune.n_knots[0]), float(tune.lam[0])
    print("\nPART 2: TUNING - top 5 settings by cross-validated RMSE")
    print(tune.head(5)[["n_knots", "lam", "R2", "RMSE", "MAE"]].round(4).to_string(index=False))
    print(f"Chosen: n_knots={K}, lambda={LAM:g}  (more smoothing = flatter curve; small n favours smooth fits)")

    # PART 3 - FINAL GAM FIT
    gam = PGAM(lam=LAM, n_knots=K).fit(X, y)
    fit = gam.predict(X)
    print(f"\nPART 3: FINAL GAM - n={gam.n_}, total effective df={gam.edf_total_:.2f}, "
          f"smooth(log_goal) edf={gam.smooth_edf():.2f}, residual sd={gam.sigma2_ ** .5:.3f}")
    print(f"In-sample R2 = {r2_score(y, fit):.3f}")

    # PART 4 - PREDICTIVE POWER & ACCURACY (out-of-sample, repeated 5-fold CV)
    perf = cv_eval(lambda: PGAM(lam=LAM, n_knots=K))
    print("\nPART 4: PREDICTIVE POWER (cross-validated, log scale)")
    print(f"R2 = {perf['R2']:.3f} (sd {perf['R2_sd']:.3f})   RMSE = {perf['RMSE']:.3f}   MAE = {perf['MAE']:.3f}")
    print(f"Baseline sd of target = {y.std():.3f}")

    # PART 5 - VARIABLE RELEVANCE
    # ---- 5a) LINEAR (PARAMETRIC) TERMS
    a, b = gam.blocks_["log_goal"]
    co = gam.coef_[b:]; se = np.sqrt(np.diag(gam.sigma2_ * gam.Ginv_[b:, b:]))
    dof = gam.n_ - gam.edf_total_
    lin = pd.DataFrame(dict(term=gam.lin_cols_, coef=co, std_err=se, t=co / se))
    lin["p_value"] = 2 * (1 - stats.t.cdf(abs(lin.t), dof))
    lin["ci_low"], lin["ci_high"] = co - 1.96 * se, co + 1.96 * se
    lin["pct_change_in_(1+frac)"] = (np.exp(co) - 1) * 100
    print("\nPART 5a: LINEAR TERMS (baselines: year 2011, state California)")
    print(lin.round(3).to_string(index=False))

    # ---- 5b) SMOOTH TERM: is log_goal needed, and is it nonlinear?
    Xd = X.drop(columns=["log_goal"])
    rss0 = ((y - np.column_stack([np.ones(len(y)), Xd.values]) @
             np.linalg.lstsq(np.column_stack([np.ones(len(y)), Xd.values]), y, rcond=None)[0]) ** 2).sum()
    df1 = gam.edf_total_ - (Xd.shape[1] + 1)
    F_goal = ((rss0 - gam.rss_) / df1) / gam.sigma2_
    p_goal = 1 - stats.f.cdf(F_goal, df1, dof)
    lin_gam = PGAM(lam=1e6, n_knots=K).fit(X, y)     # same GAM with smooth forced to a straight line
    print(f"\nPART 5b: SMOOTH TERM log_goal: overall F={F_goal:.2f}, p={p_goal:.3f} (goal size matters)")
    print(f"         edf={gam.smooth_edf():.2f} (~2 = essentially a straight line); "
          f"R2 gain over forced-linear smooth = {r2_score(y, fit) - r2_score(y, lin_gam.predict(X)):.4f}")

    # ---- 5c) DROP-ONE GROUP IMPORTANCE (loss in CV R2 when group removed)
    groups = {"log_goal (smooth)": ["log_goal"], "pre_order_perk": ["pre_order_perk"],
              "tech_campaign": ["tech_campaign"], "sbiz_campaign": ["sbiz_campaign"],
              "iot_plus": ["iot_plus"], "fixed_fund": ["fixed_fund"],
              "year_start": [c for c in X if c.startswith("year")],
              "state_grp": [c for c in X if c.startswith("state")]}
    imp = []
    for g, cols in groups.items():
        Xs = X.drop(columns=cols)
        r = (cv_eval(lambda: PGAM(lam=LAM, n_knots=K, smooth_cols=()), Xd=Xs.assign(const=1.0))
             if g.startswith("log_goal") else cv_eval(lambda: PGAM(lam=LAM, n_knots=K), Xd=Xs))
        imp.append(dict(variable=g, drop_in_CV_R2=perf["R2"] - r["R2"]))
    imp = pd.DataFrame(imp).sort_values("drop_in_CV_R2", ascending=False)
    print("\nPART 5c: DROP-ONE IMPORTANCE (bigger = more important; negative = adds noise)")
    print(imp.round(4).to_string(index=False))

    # PART 6 - DIAGNOSTICS
    resid = y - fit
    print(f"\nPART 6: DIAGNOSTICS - residual skew={stats.skew(resid):.2f}, "
          f"Shapiro p={stats.shapiro(resid).pvalue:.1e} (non-normal, right-skewed -> p-values approximate)")
    print(f"Spearman(|resid|, fitted) = {stats.spearmanr(abs(resid), fit)[0]:.3f} (no strong heteroskedasticity)")

    # PART 7 - FIGURES
    # Figure 1: smooth term for funding goal
    grid = np.linspace(X.log_goal.min(), X.log_goal.max(), 200)
    f, ci = gam.partial(grid)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(grid, f - ci, f + ci, alpha=.25, color="#1f77b4"); ax.plot(grid, f, color="#1f77b4", lw=2)
    ax.axhline(0, color="grey", lw=.8, ls="--")
    ax.plot(df.log_goal, np.full(len(df), ax.get_ylim()[0]), "|", color="k", alpha=.4)
    tk = [500, 1000, 2500, 5000, 10000, 25000, 100000]
    ax.set_xticks(np.log(tk)); ax.set_xticklabels([f"${t:,}" for t in tk], fontsize=8)
    ax.set_xlabel("Funding goal (log scale)"); ax.set_ylabel("Partial effect on log(1 + fraction raised)")
    ax.set_title("Figure 1 - GAM smooth term: funding goal (95% CI)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/Figure1_GAM_smooth_funding_goal.png", dpi=200); plt.close()

    # Figure 2: linear-term coefficients
    c = lin.iloc[::-1]; yy = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(c.coef, yy, xerr=[c.coef - c.ci_low, c.ci_high - c.coef], fmt="o", color="#d62728", capsize=3)
    ax.axvline(0, color="grey", ls="--", lw=.8); ax.set_yticks(yy); ax.set_yticklabels(c.term, fontsize=8)
    ax.set_xlabel("Coefficient (log scale), 95% CI"); ax.set_title("Figure 2 - GAM linear terms")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/Figure2_GAM_linear_terms.png", dpi=200); plt.close()
    print("\nPART 7: saved Figure1_GAM_smooth_funding_goal.png and Figure2_GAM_linear_terms.png")


# =====================================================================
# MODEL 3 - RANDOM FOREST (from random_forest.py)
# =====================================================================
def run_random_forest(clean_df):
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
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    warnings.filterwarnings("ignore", category=UserWarning)

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

    df = clean_df  # [combined] was: df = pd.read_csv(DATA)
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


# =====================================================================
# MODEL 4 - GRADIENT BOOSTING (from gradient_boosting_crowdfunding.py)
# =====================================================================
def run_gradient_boosting(clean_df):
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
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedKFold, KFold, RandomizedSearchCV
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    SEED = 42

    # ------------------------------------------------------------------ data
    df = clean_df.copy()  # [combined] was: df = pd.read_csv(IN_PATH); copy so new columns don't leak to other models
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


# =====================================================================
# RUN
# =====================================================================
MODELS = {
    "ols": ("MODEL 1 - OLS REGRESSION", run_ols),
    "gam": ("MODEL 2 - GAM", run_gam),
    "rf": ("MODEL 3 - RANDOM FOREST", run_random_forest),
    "gb": ("MODEL 4 - GRADIENT BOOSTING", run_gradient_boosting),
}

if __name__ == "__main__":
    selected = [a.lower() for a in sys.argv[1:]] or list(MODELS)
    unknown = [s for s in selected if s not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; choose from {list(MODELS)}")

    # One shared cleaning step for every model. The CSV round trip reproduces
    # crowdfunding_clean.csv exactly (to_csv rounds floats in the last digit).
    data = pd.read_csv(io.StringIO(clean(verbose=True).to_csv(index=False)))
    for key in selected:
        title, run = MODELS[key]
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        with plt.rc_context():  # keep one model's chart styling from leaking into the next
            run(data.copy())
        plt.close("all")
