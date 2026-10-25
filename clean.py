"""Clean the crowdfunding dataset into a model-ready table.

Steps follow the cleaning methodology in README.md:
  1. Remove exact duplicate rows (rows 224-438 repeat earlier campaigns) -> 224 campaigns.
  2. Drop `cid` (identifier) and `delivery` (observed after the campaign ends,
     i.e. leakage, and 76% missing).
  3. `iot_plus` is only ever 1 for tech campaigns, so a missing value on a
     non-tech campaign is set to 0. Remaining gaps stay NaN and are imputed
     inside cross-validation folds during modelling.
  4. Collapse `state` (43 levels, many with 1-2 campaigns) to CA / NY / Other.
  5. Add log transforms for the skewed goal and target.
"""
import numpy as np
import pandas as pd

RAW = "Crowdfunding campaign.xlsx"
OUT = "crowdfunding_clean.csv"

COLUMNS = [
    "log_frac_raised", "frac_raised",
    "log_goal", "amt_goal",
    "pre_order_perk", "tech_campaign", "sbiz_campaign", "iot_plus",
    "fixed_fund", "year_start", "state_grp",
]


def clean(path=RAW, verbose=False):
    log = print if verbose else (lambda *a: None)
    df = pd.read_excel(path, sheet_name="Sheet1")
    log(f"raw: {df.shape[0]} rows x {df.shape[1]} cols")

    # 1. Duplicates: every repeated cid must be an exact copy, else stop.
    df = df.drop_duplicates()
    assert df["cid"].is_unique, "duplicate cid with conflicting values"
    log(f"1. deduplicated: {len(df)} unique campaigns")

    # 2. Non-inputs.
    df = df.drop(columns=["cid", "delivery"])
    log("2. dropped cid, delivery")

    # 3. Logical fill: IoT implies tech.
    assert not ((df["iot_plus"] == 1) & (df["tech_campaign"] == 0)).any()
    fill = df["iot_plus"].isna() & (df["tech_campaign"] == 0)
    df.loc[fill, "iot_plus"] = 0
    log(f"3. iot_plus set to 0 for {fill.sum()} non-tech campaign(s); "
        f"left for pipeline: {df.isna().sum()[lambda s: s > 0].to_dict()}")

    # 4. State groups.
    df["state_grp"] = np.where(df["state"].isin(["California", "New York"]), df["state"], "Other")
    df = df.drop(columns=["state"])
    log(f"4. state_grp: {df['state_grp'].value_counts().to_dict()}")

    # 5. Transforms.
    df["log_goal"] = np.log(df["amt_goal"])
    df["log_frac_raised"] = np.log1p(df["frac_raised"])
    log(f"5. skew frac_raised {df['frac_raised'].skew():.1f} -> {df['log_frac_raised'].skew():.1f}; "
        f"amt_goal {df['amt_goal'].skew():.1f} -> {df['log_goal'].skew():.1f}")

    return df[COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    df = clean(verbose=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {df.shape[0]} rows x {df.shape[1]} cols")
