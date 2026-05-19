# Water Quality Prediction

Machine learning system to predict whether water is safe to drink based on physicochemical measurements (pH, turbidity, chloramines, etc.).

**Dataset:** [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability) — 3,276 samples, 9 features, binary target (0 = unsafe, 1 = safe)

## Project Structure

```
water_quality_prediction/
├── data/                    # water_potability.csv (not in git — download separately)
├── notebooks/
│   ├── 01_eda.ipynb         # EDA — distributions, correlations, outlier analysis
│   ├── 02_preprocessing.ipynb  # Imputation, Winsorisation, scaling
│   ├── 03_model_training.ipynb # LR, RF, XGBoost, MLP training
│   ├── 04_evaluation.ipynb     # Confusion matrices, ROC curves, feature importance
│   └── 05_prediction.ipynb     # End-to-end prediction pipeline
├── models/                  # Saved model artifacts (.pkl) and training_metrics.json
├── reports/figures/         # All 21 generated plots and diagrams
├── app/
│   └── predict.py           # Flask REST API — POST /predict
└── run_tests.py             # Runs all 25 milestone tests
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **macOS note:** `requirements.txt` pins `xgboost==1.7.6` which bundles its own OpenMP library — no `brew install libomp` needed.

## Dataset

Download `water_potability.csv` from Kaggle and place it in `data/`:

```bash
kaggle datasets download -d adityakadiwal/water-potability -p data/ --unzip
```

Or download manually and copy to `data/water_potability.csv`.

## Running Notebooks

```bash
source venv/bin/activate
jupyter notebook
```

Open notebooks in order: `01_eda` → `02_preprocessing` → `03_model_training` → `04_evaluation` → `05_prediction`.

## Models Implemented

| Model | Expected Accuracy |
|---|---|
| Logistic Regression | ~60% |
| Random Forest | ~67% |
| XGBoost | ~69% |
| Neural Network (MLP) | ~66% |

## Flask Prediction API

```bash
pip install flask
python3 app/predict.py        # starts on http://localhost:5000
```

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ph":7.0,"Hardness":204.0,"Solids":20791.0,"Chloramines":7.3,
       "Sulfate":368.5,"Conductivity":564.0,"Organic_carbon":10.4,
       "Trihalomethanes":86.0,"Turbidity":2.96}'
# {"potability":0,"label":"Unsafe","confidence":0.3821,"threshold":0.44}
```

## Running Tests

```bash
python3 run_tests.py          # all 25 milestone tests
```

## Key Finding

The dataset has a class imbalance (~61% unsafe, ~39% safe), so accuracy alone is misleading. F1-score and ROC-AUC are the primary evaluation metrics. Best model: XGBoost at threshold 0.44 — F1=0.558, AUC=0.652.
