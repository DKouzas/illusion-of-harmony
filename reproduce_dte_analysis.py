import sys
import pandas as pd
import numpy as np
import scipy
from scipy import stats

INPUT_FILE = "dte_raw_deidentified.xlsx"
OUTPUT_SCORED_FILE = "dte247_scored_reproduced.xlsx"
OUTPUT_DESCRIPTIVES_FILE = "dte_overall_descriptives.csv"
OUTPUT_ORDER_FILE = "dte_order_correct.csv"
OUTPUT_KW_FILE = "dte_kruskal_wallis.csv"
OUTPUT_ITEM_FILE = "dte_item_level_descriptives.csv"

REQUIRED_COLUMNS = {
    "anonymous_id",
    "group_code",
    "group_label",
    "human_raw",
    "dino_start_raw",
    "dino_end_raw",
    "life_origin_raw",
}

C_CONST = {
    "human": {"col": "human_raw", "true": 0.2},
    "dino_start": {"col": "dino_start_raw", "true": 235.0},
    "dino_end": {"col": "dino_end_raw", "true": 65.0},
    "life_origin": {"col": "life_origin_raw", "true": 3800.0},
}
TRUE_DINO_INTERVAL = 170.0


def safe_log_dev(series, true_value):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    mask = s.notna() & np.isfinite(s) & (s > 0)
    out.loc[mask] = np.abs(np.log10(s.loc[mask] / true_value))
    return out


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def kw_result(data, variable):
    groups = [
        sub[variable].dropna().to_numpy()
        for _, sub in data.groupby("group_code", sort=True)
    ]
    groups = [group for group in groups if len(group) > 0]

    if len(groups) < 2:
        return {
            "Variable": variable,
            "H": np.nan,
            "df": np.nan,
            "p": np.nan,
            "epsilon_sq": np.nan,
            "N": sum(len(group) for group in groups),
        }

    h, p = stats.kruskal(*groups)
    n = sum(len(group) for group in groups)
    k = len(groups)
    # Rank epsilon-squared (Tomczak & Tomczak, 2014): epsilon^2 = H / (n - 1)
    epsilon_sq = h / (n - 1) if n > 1 else np.nan

    return {
        "Variable": variable,
        "H": h,
        "df": k - 1,
        "p": p,
        "epsilon_sq": epsilon_sq,
        "N": n,
    }


# 1. Load raw de-identified input data
# The optional participant-status column is retained but not used in the analyses.
df = pd.read_excel(INPUT_FILE)
missing_columns = REQUIRED_COLUMNS.difference(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

# 2. Ensure raw estimates are numeric
raw_columns = [meta["col"] for meta in C_CONST.values()]
for col in raw_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Score point-estimate deviations
for key, meta in C_CONST.items():
    df[f"dev_{key}"] = safe_log_dev(df[meta["col"]], meta["true"])

# 4. Derive temporal-order variables
df["order_life_before_dino"] = np.where(
    df["life_origin_raw"].notna() & df["dino_start_raw"].notna(),
    (df["life_origin_raw"] > df["dino_start_raw"]).astype(float),
    np.nan,
)

df["order_dino_before_human"] = np.where(
    df["dino_start_raw"].notna() & df["human_raw"].notna(),
    (df["dino_start_raw"] > df["human_raw"]).astype(float),
    np.nan,
)

df["overall_order_ok"] = np.where(
    df["order_life_before_dino"].notna() & df["order_dino_before_human"].notna(),
    (
        (df["order_life_before_dino"] == 1.0)
        & (df["order_dino_before_human"] == 1.0)
    ).astype(float),
    np.nan,
)

# 5. Score dinosaur-era interval deviation
dino_interval = df["dino_start_raw"] - df["dino_end_raw"]
dino_interval = dino_interval.where(dino_interval > 0, np.nan)
df["dino_interval_dev"] = safe_log_dev(dino_interval, TRUE_DINO_INTERVAL)

# 6. Calculate overall Deep Time Error
component_cols = [
    "dev_life_origin",
    "dev_dino_start",
    "dev_dino_end",
    "dev_human",
    "dino_interval_dev",
]
df["overall_deep_time_error"] = df[component_cols].mean(axis=1)

# 7. Save derived/scored dataset
df.to_excel(OUTPUT_SCORED_FILE, index=False)

print("Rows:", len(df))
print("\nN by discipline:")
print(df.groupby(["group_code", "group_label"]).size())

# 8. Overall DTE descriptives
dte_desc = (
    df.groupby(["group_code", "group_label"])["overall_deep_time_error"]
    .agg(
        N="count",
        Mean="mean",
        SD="std",
        Median="median",
        Q1=q1,
        Q3=q3,
        Minimum="min",
        Maximum="max",
    )
    .round(3)
)
print("\n=== OVERALL DTE DESCRIPTIVES ===")
print(dte_desc)
dte_desc.to_csv(OUTPUT_DESCRIPTIVES_FILE)

# 9. Overall chronological-order accuracy
order_desc = (
    df.groupby(["group_code", "group_label"])["overall_order_ok"]
    .agg(N="count", Correct_n="sum", Correct_pct=lambda x: 100 * x.mean())
    .round(1)
)
print("\n=== OVERALL ORDER CORRECT ===")
print(order_desc)
order_desc.to_csv(OUTPUT_ORDER_FILE)

# 10. Kruskal-Wallis tests and epsilon-squared effect sizes
variables = [
    "overall_deep_time_error",
    "dev_human",
    "dev_dino_start",
    "dev_dino_end",
    "dev_life_origin",
    "dino_interval_dev",
]
kw_table = pd.DataFrame([kw_result(df, variable) for variable in variables])
print("\n=== KRUSKAL-WALLIS ===")
print(kw_table.round(4))
kw_table.to_csv(OUTPUT_KW_FILE, index=False)

# 11. Item-level deviation descriptives
item_desc = (
    df.groupby(["group_code", "group_label"])[variables]
    .agg(["count", "mean", "std", "median"])
    .round(3)
)
print("\n=== ITEM-LEVEL DTE DESCRIPTIVES ===")
print(item_desc)
item_desc.to_csv(OUTPUT_ITEM_FILE)

# 12. Software environment
print("\n=== SOFTWARE ENVIRONMENT ===")
print("Python:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("\nFiles created:")
for filename in [
    OUTPUT_SCORED_FILE,
    OUTPUT_DESCRIPTIVES_FILE,
    OUTPUT_ORDER_FILE,
    OUTPUT_KW_FILE,
    OUTPUT_ITEM_FILE,
]:
    print("-", filename)
