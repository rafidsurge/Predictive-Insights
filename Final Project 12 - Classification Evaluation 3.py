import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, \
    recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier



df = pd.read_csv('Final players df.csv')

# Data Preprocessing
columns = [
    'goldat10', 'xpat10', 'csat10', 'opp_goldat10', 'opp_xpat10', 'opp_csat10', 'killsat10', 'assistsat10',
    'deathsat10', 'opp_killsat10', 'opp_assistsat10', 'opp_deathsat10',
    'goldat15', 'xpat15', 'csat15', 'opp_goldat15', 'opp_xpat15', 'opp_csat15', 'killsat15', 'assistsat15',
    'deathsat15', 'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15',
    'result'
]

columns_df = df[columns]

columns_df.isna().sum()

X = columns_df.drop(columns=['result'])
y = columns_df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=23)

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)

# Best parameters from GridSearchCV
best_params = {
    'KNN': {'metric': 'euclidean', 'n_neighbors': 10},
    'Decision Tree': {'criterion': 'gini', 'max_depth': 6},
    'Logistic Regression': {'C': 100, 'solver': 'liblinear'},
    'Random Forest': {'criterion': 'gini', 'n_estimators': 200},
    'Naive Bayes': {'var_smoothing': 1e-08},
    'Gradient Boosting': {'max_depth': 3, 'n_estimators': 200},
    'XGBoost': {'max_depth': 3, 'n_estimators': 100},
    'LightGBM': {'max_depth': 3, 'n_estimators': 200},
    'Bagging': {'max_features': 0.5, 'n_estimators': 200}
}

models = {
    'KNN': KNeighborsClassifier(**best_params['KNN']),
    'Decision Tree': DecisionTreeClassifier(**best_params['Decision Tree']),
    'Logistic Regression': LogisticRegression(max_iter=1000, **best_params['Logistic Regression']),
    'Random Forest': RandomForestClassifier(**best_params['Random Forest']),
    'Naive Bayes': GaussianNB(**best_params['Naive Bayes']),
    'Gradient Boosting': GradientBoostingClassifier(**best_params['Gradient Boosting']),
    'XGBoost': XGBClassifier(**best_params['XGBoost']),
    'LightGBM': LGBMClassifier(**best_params['LightGBM']),
    'Bagging': BaggingClassifier(**best_params['Bagging'])
}

# Training and Evaluation
results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)
        if len(np.unique(y)) > 2:  # Multiclass case
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = 'N/A'

    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Classification Report': class_report,
        'Confusion Matrix': conf_matrix
    }

    print(f'{model_name} Evaluation:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'ROC AUC: {roc_auc}')
    print('Classification Report:')
    print(class_report)
    print('Confusion Matrix:')
    print(conf_matrix)
    print('---------------------------------------')

# Visualizing Results
accuracy_scores = {model_name: result['Accuracy'] for model_name, result in results.items()}
precision_scores = {model_name: result['Precision'] for model_name, result in results.items()}
recall_scores = {model_name: result['Recall'] for model_name, result in results.items()}
f1_scores = {model_name: result['F1 Score'] for model_name, result in results.items()}

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].bar(accuracy_scores.keys(), accuracy_scores.values())
ax[0, 0].set_title('Model Accuracy')
ax[0, 0].set_xticklabels(accuracy_scores.keys(), rotation=45)

ax[0, 1].bar(precision_scores.keys(), precision_scores.values())
ax[0, 1].set_title('Model Precision')
ax[0, 1].set_xticklabels(precision_scores.keys(), rotation=45)

ax[1, 0].bar(recall_scores.keys(), recall_scores.values())
ax[1, 0].set_title('Model Recall')
ax[1, 0].set_xticklabels(recall_scores.keys(), rotation=45)

ax[1, 1].bar(f1_scores.keys(), f1_scores.values())
ax[1, 1].set_title('Model F1 Score')
ax[1, 1].set_xticklabels(f1_scores.keys(), rotation=45)

plt.tight_layout()
plt.show()