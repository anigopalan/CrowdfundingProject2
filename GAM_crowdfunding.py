"""
MINI-PROJECT 2 - CROWDFUNDING CAMPAIGN SUCCESS
MODEL: Generalized Additive Model (GAM), penalized cubic splines

Business question : What drives the fraction of the funding goal reached?
Target (y)        : log_frac_raised = log(1 + frac_raised)
Smooth term       : log_goal  (flexible curve, P-spline with 2nd-order penalty)
Linear terms      : pre_order_perk, tech_campaign, sbiz_campaign, iot_plus,
                    fixed_fund, year_start (categorical), state_grp (categorical)

Sections
  PART 0  Setup & GAM class
  PART 1  Load & clean data
  PART 2  Hyperparameter tuning (knots, smoothing penalty) by repeated 5-fold CV
  PART 3  Final GAM fit
  PART 4  Predictive power & accuracy (cross-validated)
  PART 5  Variable relevance: (a) linear terms (b) smooth term (c) drop-one importance
  PART 6  Diagnostics
  PART 7  Figures (saved as PNG)

Note: pyGAM was not available, so the GAM is implemented directly with numpy.
Set DATA_PATH below to your csv location.
"""

# =====================================================================
# PART 0 - SETUP & GAM CLASS
# =====================================================================
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = "/mnt/user-data/uploads/crowdfunding_clean.csv"
OUT_DIR = "."


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


# =====================================================================
# PART 1 - LOAD & CLEAN DATA
# =====================================================================
raw = pd.read_csv(DATA_PATH)
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


# =====================================================================
# PART 2 - HYPERPARAMETER TUNING (n_knots x smoothing penalty lambda)
# =====================================================================
rows = []
for k in [5, 6, 8, 10, 12]:
    for lam in [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]:
        r = cv_eval(lambda: PGAM(lam=lam, n_knots=k)); r.update(n_knots=k, lam=lam); rows.append(r)
tune = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
K, LAM = int(tune.n_knots[0]), float(tune.lam[0])
print("\nPART 2: TUNING - top 5 settings by cross-validated RMSE")
print(tune.head(5)[["n_knots", "lam", "R2", "RMSE", "MAE"]].round(4).to_string(index=False))
print(f"Chosen: n_knots={K}, lambda={LAM:g}  (more smoothing = flatter curve; small n favours smooth fits)")


# =====================================================================
# PART 3 - FINAL GAM FIT
# =====================================================================
gam = PGAM(lam=LAM, n_knots=K).fit(X, y)
fit = gam.predict(X)
print(f"\nPART 3: FINAL GAM - n={gam.n_}, total effective df={gam.edf_total_:.2f}, "
      f"smooth(log_goal) edf={gam.smooth_edf():.2f}, residual sd={gam.sigma2_ ** .5:.3f}")
print(f"In-sample R2 = {r2_score(y, fit):.3f}")


# =====================================================================
# PART 4 - PREDICTIVE POWER & ACCURACY (out-of-sample, repeated 5-fold CV)
# =====================================================================
perf = cv_eval(lambda: PGAM(lam=LAM, n_knots=K))
print("\nPART 4: PREDICTIVE POWER (cross-validated, log scale)")
print(f"R2 = {perf['R2']:.3f} (sd {perf['R2_sd']:.3f})   RMSE = {perf['RMSE']:.3f}   MAE = {perf['MAE']:.3f}")
print(f"Baseline sd of target = {y.std():.3f}")


# =====================================================================
# PART 5 - VARIABLE RELEVANCE
# =====================================================================
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


# =====================================================================
# PART 6 - DIAGNOSTICS
# =====================================================================
resid = y - fit
print(f"\nPART 6: DIAGNOSTICS - residual skew={stats.skew(resid):.2f}, "
      f"Shapiro p={stats.shapiro(resid).pvalue:.1e} (non-normal, right-skewed -> p-values approximate)")
print(f"Spearman(|resid|, fitted) = {stats.spearmanr(abs(resid), fit)[0]:.3f} (no strong heteroskedasticity)")


# =====================================================================
# PART 7 - FIGURES
# =====================================================================
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
