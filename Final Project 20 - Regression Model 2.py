import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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

# Grouping by 'Champion' and aggregating numeric columns using median
numeric_cols = champion_stats_cleaned.select_dtypes(include=np.number).columns
grouped = champion_stats_cleaned.groupby('Champion')[numeric_cols].median().reset_index()

# Normalizing features using per-game average
for col in features:
    grouped[col] = grouped[col] / grouped['GP']

# Define X and y
X = grouped[features]
y = grouped['W%']

# Standardize the features using RobustScaler
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

xgb_model.fit(X_scaled, y)

# Evaluate the model using cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=kf, scoring='neg_root_mean_squared_error')
cv_scores = -cv_scores  # Convert negative RMSE scores to positive

print(f"Cross-Validation RMSE Scores: {cv_scores}")
print(f"Mean RMSE: {cv_scores.mean()}")
print(f"Standard Deviation of RMSE: {cv_scores.std()}")

# Predictions on the scaled dataset (or you can use a separate test set)
y_pred = xgb_model.predict(X_scaled)

# Calculate and print the Mean Squared Error on the training set
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"Training RMSE: {rmse}")

# Save the model and the scaler
model_filename = 'xgb_reg_model2.joblib'
joblib.dump(xgb_model, model_filename)

scaler_filename = 'xgb_reg_scaler2.joblib'
joblib.dump(scaler, scaler_filename)
