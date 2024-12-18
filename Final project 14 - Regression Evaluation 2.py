import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv('ChampionStats2.csv')

# Data Preprocessing
champion_stats_cleaned = df.copy()

# Converting columns to numeric
columns_to_convert = ['KP', 'CS%P15', 'DMG%', 'GOLD%', 'W%']
for col in columns_to_convert:
    champion_stats_cleaned[col] = pd.to_numeric(champion_stats_cleaned[col].str.rstrip('%'), errors='coerce') / 100.0

features = ['KDA', 'KP', 'GD10', 'XPD10', 'CSD10', 'CSPM', 'CS%P15', 'DPM', 'DMG%', 'GOLD%', 'WPM', 'WCPM']
for col in features:
    champion_stats_cleaned[col] = pd.to_numeric(champion_stats_cleaned[col], errors='coerce')

champion_stats_cleaned['GP'] = pd.to_numeric(champion_stats_cleaned['GP'], errors='coerce')
champion_stats_cleaned.dropna(inplace=True)

# Grouping by 'Champion' and aggregating numeric columns using median
numeric_cols = champion_stats_cleaned.select_dtypes(include=np.number).columns
grouped = champion_stats_cleaned.groupby('Champion')[numeric_cols].median().reset_index()

# Normalizing features using per-game average
for col in features:
    grouped[col] = grouped[col] / grouped['GP']

X = grouped[features]
y = grouped['W%']

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Best parameters from GridSearchCV
best_params = {
    'GradientBoostingRegressor': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200},
    'RandomForestRegressor': {'max_depth': 4, 'n_estimators': 100},
    'AdaBoostRegressor': {'learning_rate': 0.2, 'n_estimators': 50},
    'XGBRegressor': {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8},
    'DecisionTreeRegressor': {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 10}
}

models = {
    'GradientBoostingRegressor': GradientBoostingRegressor(**best_params['GradientBoostingRegressor']),
    'RandomForestRegressor': RandomForestRegressor(**best_params['RandomForestRegressor']),
    'AdaBoostRegressor': AdaBoostRegressor(estimator=DecisionTreeRegressor(), **best_params['AdaBoostRegressor']),
    'XGBRegressor': XGBRegressor(**best_params['XGBRegressor']),
    'DecisionTreeRegressor': DecisionTreeRegressor(**best_params['DecisionTreeRegressor'])
}

# Training and Evaluation
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    neg_rmse_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_root_mean_squared_error')
    neg_mae_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
    neg_mse_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')

    rmse_scores = -neg_rmse_scores  # Convert negative RMSE scores to positive
    mae_scores = -neg_mae_scores  # Convert negative MAE scores to positive
    mse_scores = -neg_mse_scores  # Convert negative MSE scores to positive

    results[model_name] = {
        'RMSE': rmse_scores.mean(),
        'MAE': mae_scores.mean(),
        'R2': r2_scores.mean(),
        'MSE': mse_scores.mean(),
        'RMSE Std': rmse_scores.std(),
        'MAE Std': mae_scores.std(),
        'R2 Std': r2_scores.std(),
        'MSE Std': mse_scores.std()
    }

    print(f'{model_name} Evaluation:')
    print(f'Cross-Validation RMSE Scores: {rmse_scores}')
    print(f'Mean RMSE: {rmse_scores.mean()}')
    print(f'Standard Deviation of RMSE: {rmse_scores.std()}')
    print(f'Mean MAE: {mae_scores.mean()}')
    print(f'Mean R2: {r2_scores.mean()}')
    print(f'Mean MSE: {mse_scores.mean()}')
    print('---------------------------------------')

    # Fit the model on the whole dataset
    model.fit(X_scaled, y)

# Visualizing Results
mean_rmse_scores = {model_name: result['RMSE'] for model_name, result in results.items()}
mean_mae_scores = {model_name: result['MAE'] for model_name, result in results.items()}
mean_r2_scores = {model_name: result['R2'] for model_name, result in results.items()}
mean_mse_scores = {model_name: result['MSE'] for model_name, result in results.items()}
std_rmse_scores = {model_name: result['RMSE Std'] for model_name, result in results.items()}
std_mae_scores = {model_name: result['MAE Std'] for model_name, result in results.items()}
std_r2_scores = {model_name: result['R2 Std'] for model_name, result in results.items()}
std_mse_scores = {model_name: result['MSE Std'] for model_name, result in results.items()}

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].bar(mean_rmse_scores.keys(), mean_rmse_scores.values(), yerr=std_rmse_scores.values(), capsize=5)
ax[0, 0].set_title('Model Mean RMSE with Standard Deviation')
ax[0, 0].set_xticklabels(mean_rmse_scores.keys(), rotation=45)

ax[0, 1].bar(mean_mae_scores.keys(), mean_mae_scores.values(), yerr=std_mae_scores.values(), capsize=5)
ax[0, 1].set_title('Model Mean MAE with Standard Deviation')
ax[0, 1].set_xticklabels(mean_mae_scores.keys(), rotation=45)

ax[1, 0].bar(mean_r2_scores.keys(), mean_r2_scores.values(), yerr=std_r2_scores.values(), capsize=5)
ax[1, 0].set_title('Model Mean R2 with Standard Deviation')
ax[1, 0].set_xticklabels(mean_r2_scores.keys(), rotation=45)

ax[1, 1].bar(mean_mse_scores.keys(), mean_mse_scores.values(), yerr=std_mse_scores.values(), capsize=5)
ax[1, 1].set_title('Model Mean MSE with Standard Deviation')
ax[1, 1].set_xticklabels(mean_mse_scores.keys(), rotation=45)

plt.tight_layout()
plt.show()
