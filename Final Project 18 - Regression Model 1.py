import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv('ChampionStats2.csv')

# Data Preprocessing
champion_stats_cleaned = df.copy()

# Convert percentage columns to numeric
columns_to_convert = ['KP', 'CS%P15', 'DMG%', 'GOLD%', 'W%']
for col in columns_to_convert:
    champion_stats_cleaned[col] = pd.to_numeric(champion_stats_cleaned[col].str.rstrip('%'), errors='coerce') / 100.0

# Convert other features to numeric
features = ['KDA', 'KP', 'GD10', 'XPD10', 'CSD10', 'CSPM', 'CS%P15', 'DPM', 'DMG%', 'GOLD%', 'WPM', 'WCPM']
for col in features:
    champion_stats_cleaned[col] = pd.to_numeric(champion_stats_cleaned[col], errors='coerce')

# Convert GP column to numeric
champion_stats_cleaned['GP'] = pd.to_numeric(champion_stats_cleaned['GP'], errors='coerce')
champion_stats_cleaned.dropna(inplace=True)

# Calculating weighted averages for each group
def weighted_average(group, weights_column, feature_columns):
    weighted_stats = {}
    for column in feature_columns:
        weighted_stats[column] = np.average(group[column], weights=group[weights_column])
    return pd.Series(weighted_stats, index=feature_columns)

# Apply weighted average to the grouped data
grouped = champion_stats_cleaned.groupby('Champion', group_keys=False).apply(lambda x: weighted_average(x, 'GP', features + ['W%'])).reset_index()

# Ensure 'GP' is included in the grouped DataFrame
grouped['GP'] = champion_stats_cleaned.groupby('Champion')['GP'].sum().values

# Normalizing features using per-game average
for col in features:
    grouped[col] = grouped[col] / grouped['GP']

X = grouped[features]
y = grouped['W%']

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Define and fit the XGBRegressor model
xgb_model = XGBRegressor(
    learning_rate=0.01,
    max_depth=6,
    n_estimators=200,
    subsample=0.9,
    colsample_bytree=1.0
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=kf, scoring='neg_root_mean_squared_error')
cv_scores = -cv_scores  # Convert negative RMSE scores to positive

print(f'Cross-Validation RMSE Scores for XGBRegressor: {cv_scores}')
print(f'Mean RMSE for XGBRegressor: {cv_scores.mean()}')
print(f'Standard Deviation of RMSE for XGBRegressor: {cv_scores.std()}')

# Fit the model on the entire dataset
xgb_model.fit(X_scaled, y)

# Save the model and scaler
model_filename = 'xgb_reg_model.joblib'
joblib.dump(xgb_model, model_filename)

scaler_filename = 'xgb_reg_scaler.joblib'
joblib.dump(scaler, scaler_filename)

print(f"XGBRegressor model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")
