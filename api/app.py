import os
import sys

# --- FIX PATH FIRST ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# --- NOW SAFE TO IMPORT ---
import joblib
import pandas as pd
from flask import Flask, request, jsonify

from config import CHURN_THRESHOLD
from feature_engineering import create_features
from preprocessing import clean_data

app = Flask(__name__)

pipeline = joblib.load("models/churn_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # SAME preprocessing as training
    df = clean_data(df)
    df = create_features(df,mode='infer')

    proba = pipeline.predict_proba(df)[0][1]
    churn_flag = int(proba >= CHURN_THRESHOLD)

    return jsonify({
        "churn_probability": round(float(proba), 3),
        "churn_prediction": churn_flag,
        "threshold_used": CHURN_THRESHOLD
    })

if __name__ == "__main__":
    app.run(debug=True)
