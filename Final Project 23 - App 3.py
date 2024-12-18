from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import joblib
import xgboost as xgb
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Loading the pre-trained models and scalers
best_model_xgb = joblib.load('xgb_reg_model.joblib')
scaler = joblib.load('xgb_reg_scaler.joblib')

binary_model = xgb.Booster()
binary_model.load_model('winloss.json')
scaler_bin = joblib.load('robust_scaler.pkl')

# Loading the second pre-trained model and its scaler
winloss2_model = xgb.Booster()
winloss2_model.load_model('winloss2.json')
scaler_winloss2 = joblib.load('winloss2_scaler.pkl')

# Load the new champion suggestion model and encoder
champion_suggestion_model = joblib.load('draft_model_xgb.pkl')
champion_encoder = joblib.load('draft_encoder.pkl')

# Loading the dataset for the champion list
file_path = 'ChampionStats2.csv'
df = pd.read_csv(file_path)

# Converting percentage columns to numeric
columns_to_convert = ['KP', 'CS%P15', 'DMG%', 'GOLD%', 'W%']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100.0

features = ['KDA', 'KP', 'GD10', 'XPD10', 'CSD10', 'CSPM', 'CS%P15', 'DPM', 'DMG%', 'GOLD%', 'WPM', 'WCPM']
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Converting 'GP' to numeric
df['GP'] = pd.to_numeric(df['GP'], errors='coerce')

df.dropna(inplace=True)

# Function to calculate weighted average
def weighted_average(group, weights_column, feature_columns):
    weighted_stats = {}
    for column in feature_columns:
        weighted_stats[column] = np.average(group[column], weights=group[weights_column])
    return pd.Series(weighted_stats, index=feature_columns)

# Grouping by 'Champion' and applying the weighted average function
grouped = df.groupby('Champion', group_keys=False).apply(lambda x: weighted_average(x, 'GP', features + ['W%'])).reset_index()

# Summing the GP column
grouped['GP'] = df.groupby('Champion')['GP'].sum().values

# Normalizing features using per-game average
for col in features:
    grouped[col] = grouped[col] / grouped['GP']

champions = grouped['Champion'].unique()

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def clean_data(df):
    for column in df.columns:
        if df[column].dtype == object:
            # Remove percentage signs and convert to float
            df[column] = df[column].str.replace('%', '').astype(float)
    return df

@app.route('/')
def index():
    return render_template('index.html', champions=champions)

@app.route('/predict', methods=['POST'])
def predict():
    champions_side1 = request.form.getlist('champions_side1')
    champions_side2 = request.form.getlist('champions_side2')

    avg_win_percentage_side1, avg_win_percentage_side2 = predict_win_percentage_for_sides(grouped,
                                                                                          best_model_xgb, scaler,
                                                                                          champions_side1,
                                                                                          champions_side2)

    if avg_win_percentage_side1 is None or avg_win_percentage_side2 is None:
        return jsonify({
            'error': 'Missing data for some champions. Please check the selected champions.'
        })

    # Normalize win percentages to sum to 100%
    total = avg_win_percentage_side1 + avg_win_percentage_side2
    win_percentage_side1_normalized = (avg_win_percentage_side1 / total) * 100
    win_percentage_side2_normalized = (avg_win_percentage_side2 / total) * 100

    return jsonify({
        'win_percentage_side1': float(win_percentage_side1_normalized),
        'win_percentage_side2': float(win_percentage_side2_normalized)
    })

@app.route('/suggest', methods=['POST'])
def suggest():
    first_pick = request.form['champion']
    encoded_pick1 = champion_encoder.transform([[first_pick, first_pick, first_pick, first_pick, first_pick]])[:,
                    :champion_encoder.categories_[0].size]
    y_pred = champion_suggestion_model.predict(encoded_pick1.toarray())
    decoded_picks = champion_encoder.inverse_transform(np.hstack((encoded_pick1.toarray(), y_pred)))[0]
    suggested_champions = decoded_picks[1:6].tolist()
    return jsonify({'suggested_champions': suggested_champions})

@app.route('/compare_champions', methods=['POST'])
def compare_champions():
    champion1 = request.form['champion1']
    champion2 = request.form['champion2']

    stats1 = grouped[grouped['Champion'] == champion1]
    stats2 = grouped[grouped['Champion'] == champion2]

    if stats1.empty or stats2.empty:
        return jsonify({
            'error': 'Missing data for one or both champions. Please check the selected champions.'
        })

    win_percentage1 = stats1['W%'].values[0]
    win_percentage2 = stats2['W%'].values[0]

    # Normalize win percentages to sum to 100%
    total = win_percentage1 + win_percentage2
    win_percentage1_normalized = (win_percentage1 / total) * 100
    win_percentage2_normalized = (win_percentage2 / total) * 100

    return jsonify({
        'champion1': champion1,
        'win_percentage1': float(win_percentage1_normalized),
        'champion2': champion2,
        'win_percentage2': float(win_percentage2_normalized)
    })

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('predict_binary', filename=filename))
    return render_template('upload.html')

@app.route('/predict_binary/<filename>')
def predict_binary(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(file_path)
    data = clean_data(data)
    data_scaled = scaler_bin.transform(data)

    dtest = xgb.DMatrix(data_scaled)
    predictions = binary_model.predict(dtest)

    threshold = 0.5
    prediction_labels = ["Win" if pred >= threshold else "Loss" for pred in predictions]

    return render_template('results.html', predictions=prediction_labels)

@app.route('/predict_winloss', methods=['POST'])
def predict_winloss():
    data = request.form
    features = [
        'goldat10', 'xpat10', 'csat10', 'opp_goldat10', 'opp_xpat10', 'opp_csat10',
        'killsat10', 'assistsat10', 'deathsat10', 'opp_killsat10', 'opp_assistsat10', 'opp_deathsat10',
        'goldat15', 'xpat15', 'csat15', 'opp_goldat15', 'opp_xpat15', 'opp_csat15',
        'killsat15', 'assistsat15', 'deathsat15', 'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15'
    ]

    input_data = [[
        float(data.get(feature, 0)) for feature in features
    ]]
    input_df = pd.DataFrame(input_data, columns=features)
    input_scaled = scaler_winloss2.transform(input_df)
    dmatrix = xgb.DMatrix(input_scaled)
    prediction = winloss2_model.predict(dmatrix)

    threshold = 0.5
    result = "Win" if prediction[0] >= threshold else "Loss"

    return jsonify({'result': result})

def predict_win_percentage_for_sides(champions_stats, model, scaler, champions_side1, champions_side2):
    selected_champions = champions_stats[champions_stats['Champion'].isin(champions_side1 + champions_side2)]

    unique_champions_side1 = selected_champions[selected_champions['Champion'].isin(champions_side1)].drop_duplicates('Champion')
    unique_champions_side2 = selected_champions[selected_champions['Champion'].isin(champions_side2)].drop_duplicates('Champion')

    missing_champions_side1 = set(champions_side1) - set(unique_champions_side1['Champion'])
    missing_champions_side2 = set(champions_side2) - set(unique_champions_side2['Champion'])

    if missing_champions_side1 or missing_champions_side2:
        print(f"Missing data for champions: {missing_champions_side1.union(missing_champions_side2)}")

    for champ in missing_champions_side1:
        empty_row = {col: 0 for col in features}
        empty_row['Champion'] = champ
        empty_row['W%'] = 0.0
        unique_champions_side1 = unique_champions_side1.append(empty_row, ignore_index=True)

    for champ in missing_champions_side2:
        empty_row = {col: 0 for col in features}
        empty_row['Champion'] = champ
        empty_row['W%'] = 0.0
        unique_champions_side2 = unique_champions_side2.append(empty_row, ignore_index=True)

    selected_features_side1 = unique_champions_side1[features]
    selected_features_side2 = unique_champions_side2[features]

    selected_features_side1 = clean_data(selected_features_side1)
    selected_features_side2 = clean_data(selected_features_side2)

    selected_features_scaled_side1 = scaler.transform(selected_features_side1)
    selected_features_scaled_side2 = scaler.transform(selected_features_side2)

    win_percentage_pred_side1 = model.predict(selected_features_scaled_side1)
    win_percentage_pred_side2 = model.predict(selected_features_scaled_side2)

    avg_win_percentage_side1 = win_percentage_pred_side1.mean()
    avg_win_percentage_side2 = win_percentage_pred_side2.mean()

    return avg_win_percentage_side1, avg_win_percentage_side2

if __name__ == '__main__':
    app.run(debug=True)
