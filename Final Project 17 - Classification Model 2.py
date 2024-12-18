import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import RobustScaler
import joblib

# Load dataset
df = pd.read_csv('Final players df.csv')

# Define the columns for features and target
columns = [
    'goldat10', 'xpat10', 'csat10', 'opp_goldat10', 'opp_xpat10', 'opp_csat10',
    'killsat10', 'assistsat10', 'deathsat10', 'opp_killsat10', 'opp_assistsat10', 'opp_deathsat10',
    'goldat15', 'xpat15', 'csat15', 'opp_goldat15', 'opp_xpat15', 'opp_csat15',
    'killsat15', 'assistsat15', 'deathsat15', 'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15',
    'result'
]

# Subset the dataframe based on selected columns
columns_df = df[columns]

# Split features and target
X = columns_df.drop(columns=['result'])
y = columns_df['result']

# Split the data into training and test sets before scaling (to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=23)

# Initialize the scaler
scaler = RobustScaler()

# Fit the scaler only on the training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'winloss2_scaler.pkl')

# Initialize and fit the XGBoost classifier
xgb_clf = XGBClassifier(use_label_encoder=False, max_depth=3, n_estimators=100)
xgb_clf.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = xgb_clf.predict(X_test_scaled)

# Get the predicted probabilities for log loss calculation
y_pred_proba = xgb_clf.predict_proba(X_test_scaled)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate and print the log loss
logloss = log_loss(y_test, y_pred_proba)
print(f'Log Loss: {logloss:.2f}')

# Save the trained XGBoost model
xgb_clf.save_model('winloss2.json')
