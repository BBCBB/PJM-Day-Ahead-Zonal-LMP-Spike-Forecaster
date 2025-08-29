from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data_raw")          # Input data dir
OUT = Path("processed_data")    # processed data dir

OUT.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]


# LMP
lmp_frames = []
for y in YEARS:
    f = RAW / f"da_lmp_zones_{y}.csv"
    df = pd.read_csv(f)

    tcol = "Local Timestamp Eastern Time (Interval Ending)"
    df["ts"] = pd.to_datetime(df[tcol], errors="raise")


    lmp_cols = [c for c in df.columns if c.endswith(" LMP")]
    if not lmp_cols:
        raise RuntimeError(f"No ' LMP' columns found in {f.name}")

    # melt to long
    long = df.melt(
        id_vars=["ts"],
        value_vars=lmp_cols,
        var_name="zone_raw",
        value_name="lmp"
    )
  
    long["zone"] = long["zone_raw"].str.replace(r"\s+LMP$", "", regex=True)
    long["lmp"]  = pd.to_numeric(long["lmp"], errors="coerce")
    long = long.drop(columns=["zone_raw"])
    lmp_frames.append(long)

lmp = pd.concat(lmp_frames, ignore_index=True).dropna(subset=["lmp"])

# actual load
load_frames = []
for y in YEARS:
    f = RAW / f"load_actual_{y}.csv"
    df = pd.read_csv(f)

    tcol = "Local Timestamp Eastern Time (Interval Ending)"
    df["ts"] = pd.to_datetime(df[tcol], errors="raise")

    total_col = "PJM Total Actual Load (MW)"
    keep = df[["ts", total_col]].copy()
    keep = keep.rename(columns={total_col: "load_pjm_mw"})
    keep["load_pjm_mw"] = pd.to_numeric(keep["load_pjm_mw"], errors="coerce")
    load_frames.append(keep)

load_sys = pd.concat(load_frames, ignore_index=True)

#Fuel mix to shares
mix_frames = []
for y in YEARS:
    f = RAW / f"fuel_mix_{y}.csv"
    df = pd.read_csv(f)
    tcol = "Local Timestamp Eastern Time (Interval Ending)"
    df["ts"] = pd.to_datetime(df[tcol], errors="raise")

    # identify MW columns present
    cols_map = {
        "coal":    "Coal Generation (MW)",
        "gas":     "Gas Generation (MW)",
        "nuclear": "Nuclear Generation (MW)",
        "wind":    "Wind Generation (MW)",
        "solar":   "Solar Generation (MW)",
        "hydro":   "Hydro Generation (MW)",
        "oil":     "Oil Generation (MW)",
        "other":   "Other Renewables Generation (MW)",
        "storage": "Storage Generation (MW)",
        "multi":   "Multiple Fuels Generation (MW)",
    }
    cols_map = {k:v for k,v in cols_map.items() if v in df.columns}

    for v in cols_map.values():
        df[v] = pd.to_numeric(df[v], errors="coerce")

    total_col = "Total Generation (MW)"
    total = pd.to_numeric(df[total_col], errors="coerce").replace(0, np.nan)

    out = df[["ts"]].copy()
    for k, v in cols_map.items():
        out[f"{k}_sh"] = df[v] / total
    mix_frames.append(out)

mix = pd.concat(mix_frames, ignore_index=True)

# Merge into zone hour
df = (lmp
      .merge(load_sys, on="ts", how="left")
      .merge(mix,      on="ts", how="left"))

df = df.sort_values(["zone", "ts"]).reset_index(drop=True)
print(df.shape)
print(df.head(3))

# Saveeee
OUT_FILE = OUT / "pjm_zone_hour.parquet"
df.to_parquet(OUT_FILE, index=False)
print("Wrote:", OUT_FILE)



#%%
# Features

INP = Path("processed_data/pjm_zone_hour.parquet")
OUT = Path("processed_data")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INP)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values(["zone", "ts"]).reset_index(drop=True)

#  Calendar features
df["hour"] = df["ts"].dt.hour
df["dow"] = df["ts"].dt.dayofweek      # 0=Mon … 6=Sun
df["month"] = df["ts"].dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)

#Lags & rolling stats
def add_lags(group, col):
    group[f"{col}_lag1"]   = group[col].shift(1)
    group[f"{col}_lag24"]  = group[col].shift(24)
    group[f"{col}_lag168"] = group[col].shift(168)   # 7 days
    # 24-hour moving average aligned to current hour
    group[f"{col}_ma24"]   = group[col].rolling(24, min_periods=1).mean()
    return group

df = df.groupby("zone", group_keys=False).apply(add_lags, col="lmp")


if "load_pjm_mw" in df.columns:
    df = df.groupby("zone", group_keys=False).apply(add_lags, col="load_pjm_mw")

df["lmp_tplus24"] = df.groupby("zone")["lmp"].shift(-24)

q90 = df.groupby("zone")["lmp_tplus24"].transform(lambda s: s.quantile(0.90))
df["spike"] = (df["lmp_tplus24"] >= q90).astype("Int64")

need = ["lmp_tplus24", "lmp_lag1", "lmp_lag24", "lmp_ma24"]
if "load_pjm_mw" in df.columns:
    need += ["load_pjm_mw", "load_pjm_mw_lag1", "load_pjm_mw_lag24", "load_pjm_mw_ma24"]
df_model = df.dropna(subset=need)

feature_cols = [
    "hour","dow","month","is_weekend",
    "lmp_lag1","lmp_lag24","lmp_lag168","lmp_ma24",
]
if "load_pjm_mw" in df_model.columns:
    feature_cols += ["load_pjm_mw","load_pjm_mw_lag1","load_pjm_mw_lag24","load_pjm_mw_lag168","load_pjm_mw_ma24"]

for c in ["gas_sh","coal_sh","nuclear_sh","wind_sh","solar_sh","hydro_sh","oil_sh","other_sh","storage_sh","multi_sh"]:
    if c in df_model.columns:
        feature_cols.append(c)

out_full = OUT / "pjm_zone_hour_features.parquet"
df_model.to_parquet(out_full, index=False)

with open(OUT / "feature_cols.txt", "w", encoding="utf-8") as f:
    for c in feature_cols:
        f.write(c + "\n")

print("Rows for modeling:", len(df_model))
print("Feature count:", len(feature_cols))
print("Saved:", out_full, "and feature_cols.txt")



#%% Modeling
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib

DATA_DIR = Path("processed_data")
FEAT_FILE = DATA_DIR / "pjm_zone_hour_features.parquet" 
FEAT_LIST = DATA_DIR / "feature_cols.txt"              
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)

try:
    df = pd.read_parquet(FEAT_FILE)
except Exception:
    csv_alt = FEAT_FILE.with_suffix(".csv")
    df = pd.read_csv(csv_alt)


# Features
with open(FEAT_LIST, "r", encoding="utf-8") as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]

#Targets
y_reg = df["lmp_tplus24"]        # regression target
y_cls = df["spike"].astype(int)  # classification target (0/1)

X = df[FEATURE_COLS].copy()


#%% Timeseries cross validation
def ts_scores_reg(model, X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        rows.append({
            "fold": fold,
            "MAE": mean_absolute_error(y.iloc[te], pred),
            # "RMSE": mean_squared_error(y.iloc[te], pred, squared=False),
            "RMSE": root_mean_squared_error(y.iloc[te], pred),
            "R2": r2_score(y.iloc[te], pred)
        })
    return pd.DataFrame(rows)

def ts_scores_cls(model, X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.iloc[te])[:,1]
        else:
            proba = model.predict(X.iloc[te])
        rows.append({
            "fold": fold,
            "ROC_AUC": roc_auc_score(y.iloc[te], proba),
            "PR_AUC": average_precision_score(y.iloc[te], proba),
        })
    return pd.DataFrame(rows)


#%% Regression model (next day LMP) HistGradientBoostingRegressor
naive_ok = "lmp_lag24" in X.columns
if naive_ok:
    # Evaluate on the last 20% of rows
    n = len(X)
    cut = int(n * 0.8)
    y_true = y_reg.iloc[cut:]
    y_hat  = X.loc[y_true.index, "lmp_lag24"]
    print("Naive baseline — last 20%:")
    print("  MAE :", mean_absolute_error(y_true, y_hat))
    print("  RMSE:", root_mean_squared_error(y_true, y_hat))
else:
    print("Naive baseline skipped (no lmp_lag24 in features).")

# HGB Regressor
reg_pipe = Pipeline([
    ("ct", ColumnTransformer([("num", "passthrough", FEATURE_COLS)], remainder="drop")),
    ("mdl", HistGradientBoostingRegressor(max_iter=400))
])

cv_reg = ts_scores_reg(reg_pipe, X, y_reg, n_splits=4)
print("\nCV (Regression) — per fold:\n", cv_reg)
print("CV (Regression) — mean:\n", cv_reg.mean(numeric_only=True))

# Fit final model
reg_pipe.fit(X, y_reg)
joblib.dump(reg_pipe, MODELS_DIR / "regressor_hgb.joblib")
print("Saved model:", MODELS_DIR / "regressor_hgb.joblib")

#%% Classification model (spike risk) HistGradientBoostingClassifier
cls_pipe = Pipeline([
    ("ct", ColumnTransformer([("num", "passthrough", FEATURE_COLS)], remainder="drop")),
    ("mdl", HistGradientBoostingClassifier(max_iter=400))
])

cv_cls = ts_scores_cls(cls_pipe, X, y_cls, n_splits=4)
print("\nCV (Classification) — per fold:\n", cv_cls)
print("CV (Classification) — mean:\n", cv_cls.mean(numeric_only=True))

# Fit final classifier on all data
cls_pipe.fit(X, y_cls)
joblib.dump(cls_pipe, MODELS_DIR / "classifier_hgb.joblib")
print("Saved model:", MODELS_DIR / "classifier_hgb.joblib")


#%% Inspect feature importance (permutation)
recent = X.tail(50_000) if len(X) > 50_000 else X
recent_y = y_reg.iloc[recent.index]

r = permutation_importance(reg_pipe, recent, recent_y, n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean": r.importances_mean,
    "std": r.importances_std}).sort_values("mean", ascending=False)

print("\nTop features (regression, permutation importance):")
print(imp.head(15))
imp.to_csv(MODELS_DIR / "regression_feature_importance.csv", index=False)
