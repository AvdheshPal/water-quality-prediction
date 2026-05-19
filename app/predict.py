#!/usr/bin/env python3
"""
Flask prediction API for Water Quality Prediction.
Usage: python3 app/predict.py
POST /predict with JSON body containing 9 water quality features.
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE, 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE, 'models', 'scaler.pkl')

FEATURES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]
THRESHOLD = 0.44

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        values = [float(data[f]) for f in FEATURES]
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'All fields must be numeric: {e}'}), 400

    sample  = np.array(values).reshape(1, -1)
    scaled  = scaler.transform(sample)
    prob    = float(model.predict_proba(scaled)[0][1])
    label   = int(prob >= THRESHOLD)

    return jsonify({
        'potability': label,
        'label':      'Safe' if label == 1 else 'Unsafe',
        'confidence': round(prob, 4),
        'threshold':  THRESHOLD
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'XGBoost', 'threshold': THRESHOLD})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
