import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt


# Upload the data file

data = pd.read_csv('picktable.csv')

# Data Preprocessing
winning_data = data[data['result'] == 1]

# Using OneHot Encoder to encode the champion names
encoder = OneHotEncoder()
encoded_winning_picks = encoder.fit_transform(winning_data[['pick1', 'pick2', 'pick3', 'pick4', 'pick5']]).toarray()

# Using pick1 as input variable and the rest 4 as target variable
X_winning = encoded_winning_picks[:, :encoder.categories_[0].size]  # pick1
y_winning = encoded_winning_picks[:, encoder.categories_[0].size:]  # pick2 to pick5

# Using a subset of the data for faster grid search
X_subset, _, y_subset, _ = train_test_split(X_winning, y_winning, test_size=0.9, random_state=42)
X_subset_train, X_subset_val, y_subset_train, y_subset_val = train_test_split(X_subset, y_subset, test_size=0.2,
                                                                              random_state=42)

# Best parameters
best_params = {
    'XGBRegressor': {'estimator__learning_rate': 0.01, 'estimator__max_depth': 3, 'estimator__n_estimators': 50},
    'RandomForestRegressor': {'estimator__max_depth': 3, 'estimator__n_estimators': 100},
    'GradientBoostingRegressor': {'estimator__learning_rate': 0.01, 'estimator__max_depth': 3,
                                  'estimator__n_estimators': 50},
    'DecisionTreeRegressor': {'estimator__max_depth': 3, 'estimator__min_samples_leaf': 4,
                              'estimator__min_samples_split': 2}
}

models = {
    'XGBRegressor': MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42)),
    'RandomForestRegressor': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    'GradientBoostingRegressor': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
    'DecisionTreeRegressor': MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
}

# Training and Evaluation
results = {}

for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")
    model.set_params(**best_params[model_name])
    model.fit(X_subset_train, y_subset_train)
    y_subset_pred = model.predict(X_subset_val)

    mse = mean_squared_error(y_subset_val, y_subset_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_subset_val, y_subset_pred)
    r2 = r2_score(y_subset_val, y_subset_pred)

    results[model_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    print(f'{model_name} Evaluation:')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R2: {r2}')
    print('---------------------------------------')

# Visualizing Results
mean_mse_scores = {model_name: result['MSE'] for model_name, result in results.items()}
mean_rmse_scores = {model_name: result['RMSE'] for model_name, result in results.items()}
mean_mae_scores = {model_name: result['MAE'] for model_name, result in results.items()}
mean_r2_scores = {model_name: result['R2'] for model_name, result in results.items()}

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].bar(mean_mse_scores.keys(), mean_mse_scores.values())
ax[0, 0].set_title('Model MSE')
ax[0, 0].set_xticklabels(mean_mse_scores.keys(), rotation=45)

ax[0, 1].bar(mean_rmse_scores.keys(), mean_rmse_scores.values())
ax[0, 1].set_title('Model RMSE')
ax[0, 1].set_xticklabels(mean_rmse_scores.keys(), rotation=45)

ax[1, 0].bar(mean_mae_scores.keys(), mean_mae_scores.values())
ax[1, 0].set_title('Model MAE')
ax[1, 0].set_xticklabels(mean_mae_scores.keys(), rotation=45)

ax[1, 1].bar(mean_r2_scores.keys(), mean_r2_scores.values())
ax[1, 1].set_title('Model R2')
ax[1, 1].set_xticklabels(mean_r2_scores.keys(), rotation=45)

plt.tight_layout()
plt.show()