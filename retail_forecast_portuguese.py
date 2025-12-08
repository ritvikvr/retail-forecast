
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from scipy import stats

# -----------------------
# Config / data path
# -----------------------
DATA_DIR = Path("data/retail_kaggle")
DATA_DIR.mkdir(parents=True, exist_ok=True)

data_path = DATA_DIR / "mock_kaggle.csv"


if not data_path.exists():
    raise SystemExit(f"CSV not found at {data_path}. Place it there or update data_path.")

# -----------------------
# 1) Load CSV (Portuguese column names)
# -----------------------
df = pd.read_csv(data_path, low_memory=False)
print("Columns:", df.columns.tolist())
print(df.head().to_string(index=False))

# Explicit column mapping (adjust if your CSV differs)
date_col = 'data'        # date column (Portuguese 'data')
units_col = 'venda'      # units sold
price_col = 'preco'      # unit price
# optional columns: estoque (stock) etc.

# -----------------------
# 2) Parse dates and compute revenue
# -----------------------
# Parse date; handle common formats
df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
if df[date_col].isna().any():
    nbad = df[date_col].isna().sum()
    print(f"Warning: {nbad} invalid dates detected; dropping those rows.")
    df = df.dropna(subset=[date_col])

# Make sure numeric columns are numeric
df[units_col] = pd.to_numeric(df[units_col], errors='coerce').fillna(0)
df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0.0)

# Compute revenue (in case CSV doesn't have revenue)
df['revenue'] = df[units_col] * df[price_col]

print("\nSample after revenue calc:")
print(df[[date_col, units_col, price_col, 'revenue']].head().to_string(index=False))

# -----------------------
# 3) Aggregate to monthly revenue (sum)
# -----------------------
df = df.sort_values(date_col).reset_index(drop=True)
monthly = df.set_index(date_col).resample('M')['revenue'].sum().to_frame()
monthly.index.name = 'date'
monthly = monthly.rename(columns={'revenue': 'revenue'})
print("\nMonthly aggregated (first 8 rows):")
print(monthly.head(8).to_string())

# Plot monthly revenue
plt.figure(figsize=(11,4))
plt.plot(monthly.index, monthly['revenue'], marker='o', linewidth=1)
plt.title("Monthly Revenue")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

# -----------------------
# 4) Feature engineering
# -----------------------
def create_time_features(df_in):
    df = df_in.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    return df

def create_lags(df_in, lags=[1,2,3,6,12]):
    df = df_in.copy()
    for l in lags:
        df[f'lag_{l}'] = df['revenue'].shift(l)
    return df

def create_rollings(df_in, windows=[3,6,12]):
    df = df_in.copy()
    for w in windows:
        df[f'roll_mean_{w}'] = df['revenue'].shift(1).rolling(window=w, min_periods=1).mean()
        df[f'roll_std_{w}'] = df['revenue'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    return df

df_feats = monthly.copy()
df_feats = create_time_features(df_feats)
df_feats = create_lags(df_feats, lags=[1,2,3,6,12])
df_feats = create_rollings(df_feats, windows=[3,6,12])
df_feats = df_feats.dropna().reset_index()
print("\nFeature frame head:")
print(df_feats.head().to_string(index=False))

# -----------------------
# 5) Chronological train/test split (last 12 months test)
# -----------------------
test_months = 12
df_feats = df_feats.sort_values('date').reset_index(drop=True)

if len(df_feats) <= test_months + 6:
   
    test_months = max(1, int(len(df_feats) * 0.2))
    print(f"Not enough rows for 12-month holdout. Using test_months={test_months}")

train = df_feats.iloc[:-test_months].copy()
test = df_feats.iloc[-test_months:].copy()

feature_cols = [c for c in df_feats.columns if c not in ('date','revenue')]
X_train = train[feature_cols]
y_train = train['revenue']
X_test = test[feature_cols]
y_test = test['revenue']

print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")
print("Feature columns:", feature_cols)

# -----------------------
# 6) Baseline Linear Regression
# -----------------------
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
pred_lr = lr_pipeline.predict(X_test)
rmse_lr = mean_squared_error(y_test, pred_lr) ** 0.5
mae_lr = mean_absolute_error(y_test, pred_lr)
print(f"\nLinearRegression | RMSE: {rmse_lr:.2f} | MAE: {mae_lr:.2f}")

# -----------------------
# 7) RandomForest (with small randomized search)
# -----------------------
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_dist = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [3, 5, 10, None],
    'rf__min_samples_split': [2, 5, 10]
}

rs = RandomizedSearchCV(rf_pipeline, param_dist, n_iter=8, cv=3,
                        scoring='neg_root_mean_squared_error', random_state=42)
rs.fit(X_train, y_train)
print("Best RF params:", rs.best_params_)
best_rf = rs.best_estimator_
pred_rf = best_rf.predict(X_test)
rmse_rf = mean_squared_error(y_test, pred_rf) ** 0.5
mae_rf = mean_absolute_error(y_test, pred_rf)
print(f"RandomForest | RMSE: {rmse_rf:.2f} | MAE: {mae_rf:.2f}")

# -----------------------
# 8) Plot actual vs preds for test window
# -----------------------
results = test[['date','revenue']].copy()
results['pred_lr'] = pred_lr
results['pred_rf'] = pred_rf
results = results.set_index('date')

plt.figure(figsize=(11,5))
plt.plot(monthly.index, monthly['revenue'], label='history', alpha=0.6)
plt.plot(results.index, results['revenue'], label='actual (test)', marker='o')
plt.plot(results.index, results['pred_lr'], label='pred LR', marker='x')
plt.plot(results.index, results['pred_rf'], label='pred RF', marker='s')
plt.title("Actual vs Predictions (test period)")
plt.legend()
plt.tight_layout()
plt.show()

print("\nErrors summary:")
print(f"LinearRegression RMSE={rmse_lr:.2f} MAE={mae_lr:.2f}")
print(f"RandomForest     RMSE={rmse_rf:.2f} MAE={mae_rf:.2f}")

# -----------------------
# 9) Save best model
# -----------------------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
joblib.dump(best_rf, models_dir / "rf_retail_pt.joblib")
print("Saved model to:", models_dir / "rf_retail_pt.joblib")

# -----------------------
# 10) Promotion-like hypothesis test (price-based proxy)
# -----------------------

median_price = df[price_col].median()
df['is_low_price'] = df[price_col] < median_price  
rev_low = df.loc[df['is_low_price'], 'revenue']
rev_not = df.loc[~df['is_low_price'], 'revenue']
print(f"\nLow-price rows: {len(rev_low)} | Not-low-price rows: {len(rev_not)}")
if len(rev_low) > 10 and len(rev_not) > 10:
    tstat, pval = stats.ttest_ind(rev_low, rev_not, equal_var=False)
    print(f"t-test low-price vs not-low-price: t={tstat:.3f}, p={pval:.4f}")
    if pval < 0.05:
        print("Difference is statistically significant (p < 0.05).")
    else:
        print("No statistically significant difference found (p >= 0.05).")
else:
    print("Not enough rows to run a reliable t-test for price-based promo proxy.")

# -----------------------
# 11) Feature importance -> surface drivers
# -----------------------
if hasattr(best_rf.named_steps['rf'], "feature_importances_"):
    fi = best_rf.named_steps['rf'].feature_importances_
    feat_imp = pd.Series(fi, index=feature_cols).sort_values(ascending=False)
    print("\nTop features by importance:\n", feat_imp.head(10).to_string())
    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_imp.head(10).values, y=feat_imp.head(10).index)
    plt.title("Top 10 feature importances (RandomForest)")
    plt.tight_layout()
    plt.show()
else:
    print("Model has no feature_importances_ attribute.")

# Actionable lines for PM
top_feats = feat_imp.head(6).index.tolist() if 'feat_imp' in locals() else []
print("\nSuggested top drivers to discuss with Product Manager:")
for f in top_feats:
    print("-", f)

print("\nScript finished.")
