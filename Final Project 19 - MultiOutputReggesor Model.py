import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib



data = pd.read_csv('picktable.csv')

winning_data = data[data['result'] == 1]

encoder = OneHotEncoder()

encoded_winning_picks = encoder.fit_transform(winning_data[['pick1', 'pick2', 'pick3', 'pick4', 'pick5']]).toarray()

X_winning = encoded_winning_picks[:, :encoder.categories_[0].size]

y_winning = encoded_winning_picks[:, encoder.categories_[0].size:]

X_winning_train, X_winning_test, y_winning_train, y_winning_test = train_test_split(X_winning, y_winning, test_size=0.2, random_state=42)

best_params_xgb = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}

best_model_xgb = MultiOutputRegressor(XGBRegressor(**best_params_xgb, objective='reg:squarederror', random_state=42))


best_model_xgb.fit(X_winning_train, y_winning_train)

y_winning_pred_best_xgb = best_model_xgb.predict(X_winning_test)

mae = mean_absolute_error(y_winning_test, y_winning_pred_best_xgb)
mse = mean_squared_error(y_winning_test, y_winning_pred_best_xgb)
rmse = np.sqrt(mse)
r2 = r2_score(y_winning_test, y_winning_pred_best_xgb)

print("XGBRegressor")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)



def suggest_champions_winning(first_pick):
    # Encode the input champion
    encoded_pick1 = encoder.transform([[first_pick, first_pick, first_pick, first_pick, first_pick]])[:, :encoder.categories_[0].size]

    # Predict the remaining champions
    y_pred = best_model_xgb.predict(encoded_pick1.toarray())

    # Decode the predicted champions
    decoded_picks = encoder.inverse_transform(
        np.hstack((encoded_pick1.toarray(), y_pred))
    )[0]

    return decoded_picks[1:6]

joblib.dump(best_model_xgb, 'draft_model_xgb.pkl')
joblib.dump(encoder, 'draft_encoder.pkl')

first_pick = 'Vi'

suggested_champions_winning = suggest_champions_winning(first_pick)

print("Suggested champions:", suggested_champions_winning)

