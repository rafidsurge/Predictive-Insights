import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import RobustScaler
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('Final players df 2.csv')

# Select specific columns for modeling
columns = [
    'kills', 'deaths', 'assists', 'earnedgold', 'earned gpm', 'earnedgoldshare',
    'totalgold', 'golddiffat15', 'opp_goldat15', 'goldspent', 'result'
]
columns_df = df[columns]

# Features and target
X = columns_df.drop(columns=['result'])
y = columns_df['result']

# Split the data before scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=23)

# Scaling only on training data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and scale the training data
X_test_scaled = scaler.transform(X_test)       # Only transform the test data

# Save the scaler for future use
joblib.dump(scaler, 'robust_scaler.pkl')

# XGBoost Classifier with tuned parameters
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, max_depth=3, n_estimators=200)

# Fit the model on the training data
xgb_clf.fit(X_train_scaled, y_train)

# Predict on the test data
y_pred = xgb_clf.predict(X_test_scaled)
y_pred_proba = xgb_clf.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)

# Output metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Log Loss: {logloss:.2f}')

# Save the model for future use
xgb_clf.save_model('winloss.json')
