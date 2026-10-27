# What Drives Crowdfunding Success: Technical Answers

Answers to the assignment questions, drawn from the group's code (`clean.py`, the four model files, `all_models.py`) and the presentation *What Drives Crowdfunding Success, and What We Should Do About It* (v4). Every number below comes from running the code in this repo; where the deck shows a different figure, it's flagged in [Section 9](#9-consistency-notes-deck-vs-code).

**Short answer.** Three campaign traits are consistently linked to raising more of the goal: **offering a pre-order perk, building a smart/connected (IoT) device, and being based in California**. Goal size, category, all-or-nothing funding and launch year have little or no effect. All four models agree on this, but none predicts individual campaigns well: they explain about 4-8% of the variation in campaigns they haven't seen. We therefore select **OLS regression**: it matches the complex models' accuracy and is the only one that gives a clear, testable size for each effect.

---

## 1. Business question and target

**Question:** What drives success in crowdfunding campaigns?

**Target:** `frac_raised`, the fraction of the funding goal reached (1.0 = exactly funded). All models predict `log_frac_raised = log(1 + frac_raised)`, because the raw value is extremely skewed:

| | Value |
| --- | --- |
| Campaigns that raised $0 | 51 of 224 (23%) |
| Campaigns that reached their goal | 137 of 224 (61%) |
| Median `frac_raised` | 1.01 |
| Maximum `frac_raised` | 25.9 |
| Skewness, raw -> log | 4.7 -> 1.3 |

Fitted on the raw value, a regression would mostly chase the 14 campaigns that raised 5x or more. The `1 +` keeps the zeros (log(0) is undefined).

## 2. Data and sample size

`clean.py` turns the raw file into the modelling table every model uses (`crowdfunding_clean.csv`). Full methodology is in the repo `README.md`; the decisions that matter for modelling:

| Step | What | Why it matters for modelling |
| --- | --- | --- |
| Remove duplicates | 439 rows -> **224 campaigns**. Rows 224-438 were exact copies. | Copies would sit in both training and test folds, overstating accuracy, and would make n look twice as large. The copies favoured big successes: with them, the mean was 2.94x goal and 69% succeeded; the honest figures are 1.75x and 61%. |
| Drop `delivery` | Only known months after funding closes; 76% missing. | Using it would be **data leakage** (predicting with information unavailable at launch). Used only in the policy analysis. |
| Drop `cid` | Arbitrary ID | No predictive meaning. |
| Group `state` | 43 states -> California / New York / Other | 21 states have 2 or fewer campaigns; 42 dummies for 224 rows would overfit. |
| Log transforms | `log_goal`, `log_frac_raised` | Goal skew 3.1 -> 0.3; target skew 4.7 -> 1.3. |
| Missing values | 7 cells (5 `fixed_fund`, 2 `iot_plus`) left blank | Each model fills them its own way (Section 4). |

**Sample size shaped every modelling choice.** With n = 224:

- **Few inputs.** The models use 8 inputs (about 10-11 parameters once state and year are coded), roughly 20 campaigns per parameter, inside the usual 10-20 rule of thumb.
- **Out-of-sample testing only.** A single 80/20 split would score just ~45 campaigns, so every model uses k-fold cross-validation, most of them repeated.
- **Nested tuning.** For models with tuning settings, the settings are chosen inside each training fold, so the reported scores are not optimistically biased.
- **Flexibility is restrained.** Tree depths, leaf sizes and smoothing penalties are biased toward simple fits, because flexible models memorise small samples. An untuned random forest, for example, explained 74% of training data and scored *below zero* on new data.

## 3. Models considered and why OLS was selected

We compared four models of increasing flexibility. Each tests a different reason the simple model might be wrong.

| Model | File | What it can capture that OLS can't | Validation design |
| --- | --- | --- | --- |
| **1. OLS regression** | `OLS_crowdfunding_clean.ipynb` | (baseline: additive, linear effects) | 5-fold CV, n = 224 |
| **2. GAM** (penalised splines) | `GAM_crowdfunding.py` | A curved effect of goal size | 5-fold x 10 repeats, n = 217 |
| **3. Random forest** | `random_forest.py` | Interactions and thresholds, automatically | Nested CV: 5-fold x 5 repeats outer, 5-fold inner |
| **4. Gradient boosting** | `gradient_boosting_crowdfunding.py` | Interactions and thresholds, built sequentially | Nested CV: 5-fold x 2 repeats outer, 5-fold inner |

**Out-of-sample performance (log scale, averaged over test folds):**

| Model | MAE | RMSE | R² (sd across folds) |
| --- | --- | --- | --- |
| OLS | 0.515 | 0.654 | 0.045 (0.209) |
| GAM | **0.508** | **0.645** | **0.075** (0.150) |
| Random forest | 0.512 | 0.654 | 0.044 (0.139) |
| Gradient boosting | not computed | 0.650 | 0.068 (0.144) |
| *Predict the training mean (no model)* | | *0.685* | *about 0* |

**Why OLS, even though it doesn't have the lowest error:**

1. **The accuracy differences are noise.** The spread of MAE is 0.007 and of R² is 0.03, while R² varies by 0.14-0.21 *between folds of the same model*. The models also used different fold designs, so differences this small can't rank them.
2. **Flexibility found nothing to exploit.** The GAM's goal-size curve came back as a straight line (effective degrees of freedom 2.03; R² gain over a forced straight line 0.0000). The tree models' tuning repeatedly chose the shallowest trees available. If the relationships were curved or interacting, the flexible models would have beaten OLS. They didn't.
3. **Interpretability.** OLS gives one effect size per factor with a confidence interval and a significance test (with HC3 robust standard errors). Tree models only rank importance. For a business and policy audience, "pre-orders: +42%, p < 0.001" is actionable; "pre-orders: importance 0.09" is not.
4. **Robustness at small n.** OLS has no tuning settings to overfit, and it fits in under 0.1 s, against ~6.5 min for the nested random forest.
5. **Agreement across methods.** The complex models are still valuable: they confirm OLS isn't missing structure, and all four select the same three drivers (Section 6).

**Within OLS**, four nested specifications were compared (in-sample, n = 224):

| Specification | R² | Adj. R² | AIC |
| --- | --- | --- | --- |
| 1. Goal only | 0.003 | -0.001 | 468.8 |
| 2. + campaign features | 0.160 | 0.137 | 440.4 |
| **3. + state and year (primary)** | **0.193** | **0.155** | **439.4** |
| 4. + pre-order x tech interaction | 0.214 | 0.173 | 435.5 |

Specification 3 is the primary model: each variable has one effect, which is easiest to explain. Specification 4 fits better in-sample (interaction p = 0.018, AIC 4 points lower) and suggests pre-orders help tech campaigns more. But it was not cross-validated, and with n = 224 an extra interaction is a risk of overfitting; it's reported as a supporting finding.

## 4. Hyperparameter selection and tuning

| Model | Settings searched | How chosen | Chosen | What the choice tells us |
| --- | --- | --- | --- | --- |
| **OLS** | None (no hyperparameters). Specification chosen from 4 nested formulas. | AIC/BIC and adjusted R², plus interpretability | Specification 3 | Adding features helps a lot (R² 0.003 -> 0.193); the interaction adds a little. |
| **GAM** | Number of spline knots {5, 6, 8, 10, 12} x smoothing penalty λ {0.01 ... 1000} (45 combinations) | Lowest RMSE, 5-fold x 10 repeated CV | 5 knots, λ = 1000 | The most-smoothed option won. The curve became a straight line (edf 2.03). |
| **Random forest** | Max depth {2, 3, 4, 6, unlimited} x min leaf size {5, 10, 20} x features per split {0.33, 0.5, 1.0} (45 combinations); 300 trees fixed | Grid search, 5-fold inner CV inside each of 25 outer folds | Depth 2 in 13/25 folds; final model depth 4, leaf 10, features 0.5 | Shallow trees win: little nonlinear structure to find. |
| **Gradient boosting** | Trees {100, 200, 300} x learning rate {0.01-0.1} x depth {1, 2, 3} x min leaf {3-12} x subsample {0.6-1.0} x features {0.5-1.0} | Randomised search (12 candidates), 5-fold inner CV inside each of 10 outer folds | Depth 1 in 7/10 folds; learning rate 0.01; min leaf 12 | Depth 1 means each tree uses a single variable, i.e. an additive model like OLS. |

**Justification for the ranges.** Every search was deliberately weighted toward simple models, because n = 224:

- **Trees:** shallow depths (GB 1-3; RF mostly 2-6) and large minimum leaves stop trees isolating one or two campaigns.
- **GAM:** the penalty range runs up to very heavy smoothing.
- **Subsampling:** random subsets of rows and features (GB, RF) reduce variance.
- **Trees count (RF):** fixed rather than tuned, because more trees only stabilise the forest; they don't add complexity.

**Why nested cross-validation.** Choosing settings on the same folds used to report accuracy inflates the score. In nested CV, the inner loop picks settings using only the training part of each outer fold, and the outer fold is then scored once. The reported accuracy is therefore an honest estimate for a new campaign.

**Missing values in tuning.** RF and GB impute inside each fold (most common value from the training fold only), so test rows never influence the fill. OLS fills the 7 blanks with 0, the most common value; a constant fill uses no test-fold information. The GAM drops the 7 incomplete rows (n = 217) instead (see Section 9).

**One tuning caveat.** The GAM's chosen λ = 1000 is the edge of its grid, so an even smoother fit might score slightly better. It wouldn't change the conclusion: the fit is already a straight line, and forcing it fully linear changed in-sample R² by 0.0000.

## 5. Model outcomes: predictive power and accuracy

**Predictive power is low for every model.** On campaigns they haven't seen, the models explain 4-8% of the variation in `log_frac_raised`. In-sample, the best explain about 19-22%.

| Measure | Value | Interpretation |
| --- | --- | --- |
| Out-of-sample R² (log scale) | 0.04-0.08 | Most of what separates campaigns isn't in this data. |
| In-sample R² (OLS spec. 3 / GAM / RF train) | 0.19 / 0.20 / 0.22 | The gap to out-of-sample shows mild overfitting even in simple models. |
| Out-of-sample MAE (log scale) | 0.51 | About 0.5 in log(1 + frac) units. For a campaign that raised 1.0x, that's a typical miss down to roughly 0.2x or up to 2.3x. |
| Improvement over predicting the mean | RMSE 0.645-0.654 vs 0.685 | About 5% lower error than no model. |
| Fold-to-fold stability | R² sd 0.14-0.21 | Some folds score below zero (e.g. OLS fold 3: -0.29). |

**Accuracy on the original scale** (gradient boosting and random forest, back-transformed to multiples of the goal):

| Measure | Value |
| --- | --- |
| Mean absolute error | 1.51x goal (both models), vs 1.49x for predicting the median |
| Median absolute error (RF) | 0.77x goal |
| R² on original scale (GB) | 0.043 |
| Reached-goal / missed-goal accuracy (RF) | 50%, vs 61% for "every campaign succeeds" |

On the original scale, the average error is no better than a constant guess, because a handful of campaigns raised 10-26x their goal and nothing in the data predicts them. As a success/failure classifier, the regression is worse than the naive rule. Back-transforming log predictions pulls them toward the middle, so the model under-calls success.

**What the models *are* reliable for: direction and relative size of effects.** The three drivers are significant in OLS (HC3 p <= 0.012) and the GAM (p <= 0.004). They are also the top three in both tree models' held-out importance, and in the same direction everywhere. The right use is to **design campaigns and set platform policy, not to forecast an individual campaign's revenue**.

**Why predictive power is low.** The data has only 8 coarse descriptors. Viral outcomes depend on things we don't observe: pitch and video quality, press and influencer coverage, the founders' networks, price and product novelty.

**Diagnostics (OLS / GAM):**

- **Collinearity:** not an issue (all VIF < 3.1).
- **Unequal error spread:** present (Breusch-Pagan p = 0.0046), corrected with HC3 robust standard errors.
- **Residuals:** right-skewed and non-normal (skew 0.88), so p-values are approximate.
- **Shape:** the residual trend against fitted values is flat, so the linear form is adequate.

## 6. Relevance of variables

**OLS specification 3 (primary model, HC3 robust standard errors):**

| Variable | Coefficient | % change in (1 + frac raised) | 95% CI (coef) | p-value | Verdict |
| --- | --- | --- | --- | --- | --- |
| **Pre-order perk** | +0.352 | **+42%** | 0.166 to 0.538 | < 0.001 | Strong driver |
| **IoT / smart device** | +0.539 | **+71%** | 0.120 to 0.958 | 0.012 | Strong driver |
| **California** (vs other states) | +0.292 | **+34%** | 0.068 to 0.515 | 0.011 | Driver |
| Funding goal (log) | -0.066 | -4.5% per doubling | -0.136 to 0.004 | 0.063 | Weak, borderline |
| Small business | +0.118 | +12% | -0.099 to 0.335 | 0.288 | Not significant |
| Tech campaign | +0.043 | +4% | -0.198 to 0.284 | 0.729 | Not significant |
| Fixed (all-or-nothing) funding | -0.100 | -9% | -0.298 to 0.099 | 0.325 | Not significant |
| New York (vs other states) | -0.013 | -1% | -0.356 to 0.330 | 0.940 | Not significant |
| Year 2012 / 2013 (vs 2011) | +0.002 / +0.074 | 0% / +8% | | 0.99 / 0.57 | Not significant |

**How to read the percentages.** The model is on log(1 + frac_raised), so +42% means (1 + fraction raised) is 42% higher, holding the other factors fixed. Example: a campaign predicted to raise 1.0x its goal (1 + 1.0 = 2.0) would be predicted at 2.0 x 1.42 = 2.84, i.e. **1.84x its goal** with a pre-order perk. The presentation shortens this to "lifts funding about 42%".

**Interpreting the structure.**

- **Category is a three-way choice.** `tech_campaign` and `sbiz_campaign` never overlap, so their effects are relative to the 28 "other" campaigns.
- **IoT is nested in tech.** `iot_plus` is only ever 1 for tech campaigns, so the +71% compares IoT tech with other tech.

**Agreement across models:**

| Variable | OLS % effect | GAM % effect | RF importance (held-out) | GB importance (held-out) |
| --- | --- | --- | --- | --- |
| Pre-order perk | +42% (p < 0.001) | +39% (p = 0.001) | 0.094, rank 1 | 0.113, rank 1 |
| IoT device | +71% (p = 0.012) | +67% (p = 0.001) | 0.057, rank 2 | 0.092, rank 2 |
| California | +34% vs other (p = 0.011) | about +38% vs other (p = 0.004) | 0.021, rank 3 (state group) | 0.025, rank 3 |
| Goal size | -4.5% per doubling (p = 0.063) | straight line, F-test p = 0.028 | -0.007 (no help) | 0.015, rank 4 |
| Everything else | n.s. | n.s. | <= 0.001 | <= 0.005 |

Importance = average drop in test R² when the variable is shuffled (RF and GB). The GAM's drop-one test agrees: pre-order 0.034, IoT 0.031, state 0.026, goal 0.010, all others negative.

**Goal size is the only disagreement.** OLS and the GAM find a small negative effect (about -4.5% per doubling of the goal, borderline significant). The tree models find it adds little or nothing on unseen data. A built-in tree importance score overstates goal size because it's the only continuous input; permutation importance on held-out folds corrects for that. **Conclusion: goal size matters far less than commonly assumed.**

## 7. Policy implications

The findings are associations in observational data from 2011-2013, not proven causes. Recommendations should be piloted and measured (the deck's ask is a Q1 pilot).

### Crowdfunding portals

1. **Build pre-orders into onboarding (deck: "Perk Builder").** Pre-orders are the largest controllable driver (+42%, p < 0.001, and #1 in every model). Portals should prompt or require creators to set up a product reward before launch. Only 49% of campaigns offered one.
2. **Goal-setting guidance (deck: "Stretch-Goal Calculator").** Because goal size has at most a small effect, encouraging a modest initial goal plus stretch goals costs creators little. Any ranking or social-proof benefit from hitting 100% early is plausible but **not measured here**; the pilot should test it.
3. **Hardware due diligence.** Smart-tech campaigns get the biggest funding lift *and* carry the biggest delivery risk (below). Requiring a working prototype video for IoT campaigns protects backer trust, which is the portal's long-term asset.
4. **Use the model for guidance, not gatekeeping.** With out-of-sample R² of ~5%, the model can't reliably score or reject individual campaigns. It should inform design tools and education, not approval decisions.
5. **Address the 23% that raise nothing.** These campaigns cost the portal without earning fees. They are disproportionately campaigns without pre-orders: 29% of non-pre-order campaigns raised $0, vs 16% with pre-orders.

### Regulators

1. **Delivery risk is real and concentrated.** Of 53 pre-order campaigns with a delivery record, 41 delivered, 3 partly delivered and 9 delivered nothing (23% not fully delivered). For IoT campaigns, 6 of 18 (33%) did not fully deliver, vs 6 of 35 (17%) for others.
2. **Overfunding is a warning sign.** The 9 campaigns that delivered nothing had raised a median **9x** their goal, against about 2.2x for campaigns that delivered. Very large oversubscription can outstrip a small team's capacity to produce. Regulators could require progress reporting once a campaign passes a multiple of its goal. (Small sample, n = 9: treat as a hypothesis worth monitoring.)
3. **Pre-orders are effectively consumer sales without consumer protections.** The strongest success factor is selling a product that doesn't exist yet. Disclosure of development stage, delivery timelines and refund policy is the natural lever, particularly for hardware.
4. **Geographic concentration.** California campaigns raise about 34% more, holding other factors constant, plausibly through investor networks and media. If broad access to capital is a policy goal, support for founders outside tech hubs (incubators, matching programmes) addresses a measurable gap.
5. **Data gaps limit oversight.** Delivery was recorded for only 24% of campaigns. Mandatory outcome reporting would let regulators track fulfilment risk properly.

### Crowdfunding companies and entrepreneurs

1. **Offer a real product as a pre-order reward.** It's the single most reliable factor, and it is under the creator's control.
2. **Connected or smart features attract backers, but plan delivery carefully.** IoT campaigns raise about 71% more, but a third of recorded IoT campaigns failed to fully deliver, which damages reputation and future fundraising.
3. **Don't over-optimise the goal amount.** Doubling the goal costs at most about 4.5%. Set a goal that covers a viable first run, and use stretch goals for upside.
4. **Outside California, compensate with pre-launch outreach.** The location effect likely reflects networks and media access; building an audience before launch is the controllable substitute.
5. **Category and funding model don't decide outcomes.** Tech vs small business vs other, and all-or-nothing vs keep-what-you-raise, show no significant effect. Choose them on business grounds.
6. **Expect high variance.** Even a well-designed campaign is hard to predict: over 90% of the variation in outcomes comes from factors outside this data. Plan finances for a range of outcomes, not the point estimate.

## 8. Limitations

- **Small sample (n = 224)** limits model complexity and makes estimates noisy (wide confidence intervals, unstable folds).
- **Old data (2011-2013).** Platforms, algorithms and backer behaviour have changed since.
- **Observational, not causal.** For example, creators who offer pre-orders may differ in other ways, such as being more prepared.
- **Few, coarse variables.** No pitch quality, video, media coverage, price or team data, which is the main reason predictive power is low.
- **Zero-inflation.** 23% of campaigns raised nothing. A two-part model (did it raise anything? then how much?) might fit better than a single regression; not yet tested.
- **Different validation designs across models** (5x1, 5x10, nested 5x5, nested 5x2). They make the comparison indicative rather than exact.

## 9. Consistency notes (deck vs code)

These are small, but worth fixing before submission so the slides match the code.

| Slide | Deck says | Code gives | Suggested fix |
| --- | --- | --- | --- |
| 6 (Random forest) | Average error 0.518 | MAE 0.512 (`random_forest.py`, nested CV) | Use 0.512, or note the run it came from |
| 6 | "Strong in training" / "memorized the training data" | Training R² 0.22, test 0.04: mild overfitting. The untuned forest memorised (train 0.74, test < 0). | "More complex, no more accurate" |
| 6 | Setup time 6.5 min | Time to tune *and evaluate* via nested CV | "Time to tune and evaluate" |
| 7 (Gradient boosting) | Average error 0.509; explains 4% | Log-scale MAE isn't computed by the script; log-scale R² is 0.068 (4.3% is the *original-scale* R²) | Add log-scale MAE to the script; use 6.8% to match the other slides' log scale |
| 5 (GAM) | Compared directly with the others | Fitted on 217 campaigns (drops 7 incomplete rows) and validated with 5x10 repeats | Footnote, or refit on 224 with in-fold imputation |
| 9 | Goal doubling trims funding about 4.6% | 2^-0.066 - 1 = -4.5% (4.6% is the linear approximation) | Either is fine; "about 4.5%" is exact |

## 10. Reproducing the results

```bash
pip install pandas numpy scipy scikit-learn statsmodels seaborn matplotlib openpyxl
python3 clean.py                 # raw xlsx -> crowdfunding_clean.csv
python3 all_models.py            # all four models from one shared clean.py run (~15 min)
python3 all_models.py ols gam    # or any subset: ols, gam, rf, gb
```

`all_models.py` runs each model's code unchanged on the same cleaned data. Its output matches the individual files.
