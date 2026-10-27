# CrowdfundingProject2
code for crowdfunding project

**Business question:** What drives success in crowdfunding campaigns? We predict `frac_raised`, the fraction of the funding goal reached, with regression.

## Files

| File | Purpose |
| --- | --- |
| `Crowdfunding campaign.xlsx` | Raw data (`Sheet1`) and data dictionary (`Sheet2`) |
| `clean.py` | Reproducible cleaning: `python3 clean.py` (needs `pandas`, `numpy`, `openpyxl`) |
| `random_forest.py` | Random forest with nested CV tuning: `python3 random_forest.py` (needs `scikit-learn`; ~6.5 min) |
| `OLS_crowdfunding_clean.ipynb` | OLS linear regression analysis with HC3 robust standard errors, diagnostics, and 5-fold CV |
| `crowdfunding_clean.csv` | Model-ready output: 224 campaigns x 11 columns |
| `all_models.py` | All four models' code in one file, run on a single shared `clean.py` output: `python3 all_models.py` (or a subset, e.g. `python3 all_models.py ols gam`) |
| `technical_docs/` | Answers to the assignment questions and a glossary of technical terms |

## Cleaning methodology

Raw: 439 rows. Cleaned: **224 unique campaigns**, 7 values left missing on purpose.

### 1. Remove duplicate campaigns (439 -> 224 rows)
Rows 224-438 are exact copies of earlier campaigns. Every repeated `cid` matches its original in all columns, so dropping them loses nothing. 77 campaigns repeat 2-11 times, and the copies favour extreme successes (the top performer, 25.9x goal, appears 10 times).

| Measure | Raw (439) | Deduplicated (224) |
| --- | --- | --- |
| Mean `frac_raised` | 2.94 | 1.75 |
| Share reaching goal | 69% | 61% |
| 95th percentile `frac_raised` | 23.5 | 5.8 |

Keeping the copies would put the same campaign in both train and test folds (overstated accuracy) and make the sample look about twice its real size. **n = 224 sets the limit on model complexity.**

### 2. Drop `cid` and `delivery`
- `cid` is an arbitrary identifier.
- `delivery` is **leakage**: it's only known months after funding closes. It's also 76% missing (only recorded for pre-order campaigns) and uses an undefined 0.5 code. It is kept out of the model but used in the policy discussion: of 53 pre-order campaigns with a record, 41 delivered, 3 partly and 9 did not.

### 3. Missing values
- `iot_plus` is 1 only for tech campaigns, so the one blank on a non-tech campaign is set to 0 (a certainty, not a guess).
- The other 7 gaps (5 `fixed_fund`, 2 `iot_plus`) are **left blank** and filled with the training-fold mode inside cross-validation, so test rows never influence the fill.
- Rows aren't deleted: all 8 incomplete campaigns had no pre-order perk and 5 raised nothing, so dropping them would bias the data toward success.

### 4. Group `state` into CA / NY / Other
43 states for 224 rows would mean 42 dummy columns; 21 states have 2 or fewer campaigns. California (50) and New York (24) are the only groups big enough to estimate. Caveat: New York looks like "Other" (58% vs 57% reach goal; California 76%), so CA-vs-rest is a possible simplification.

### 5. Log transforms
- **Target:** `log_frac_raised = log(1 + frac_raised)`. Outcomes cluster at 0 (51 campaigns, 23%) and just past 1.0, with a tail to 25.9x. Skewness falls from 4.7 to 1.3; the `1 +` keeps the zeros.
- **Goal:** `log_goal = log(amt_goal)`. Goals range $500-$200,000; skewness falls from 3.1 to 0.3.

### Output columns

| Column | Role |
| --- | --- |
| `log_frac_raised` | Target |
| `frac_raised` | Raw target, for reporting |
| `log_goal` | Input |
| `amt_goal` | Raw goal, reference only |
| `pre_order_perk`, `tech_campaign`, `sbiz_campaign`, `iot_plus`, `fixed_fund`, `year_start` | Inputs |
| `state_grp` | Input, encoded as CA and NY dummies (Other = baseline) |

Notes for modelling: `tech_campaign` and `sbiz_campaign` never overlap, so with "neither" (28 campaigns) as baseline they form one 3-way category. `iot_plus` only applies within tech, so it measures IoT tech vs other tech. Only 10 campaigns started in 2011.

### Open decisions
- Target: log(1 + frac) vs capping at the 95th percentile vs a two-part model (raised anything? then how much)
- State: CA / NY / Other vs CA-vs-rest vs census regions vs drop
- Interactions: pre-order x tech, goal x pre-order
- Year: keep three levels or merge 2011 into 2012
