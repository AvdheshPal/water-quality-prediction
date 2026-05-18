# Water Quality Prediction

Machine learning system to predict whether water is safe to drink based on physicochemical measurements (pH, turbidity, chloramines, etc.).

**Dataset:** [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability) — 3,276 samples, 9 features, binary target (0 = unsafe, 1 = safe)

## Project Structure

```
water_quality_prediction/
├── data/                    # water_potability.csv (not in git — download separately)
├── notebooks/
│   ├── 01_eda.ipynb         # M1: Setup & dataset loading
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_prediction.ipynb
├── models/                  # Saved model artifacts (.pkl)
├── reports/figures/         # Generated plots
└── app/                     # Optional Flask prediction API
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **macOS note:** XGBoost requires OpenMP. If `import xgboost` fails, run `brew install libomp`.  
> The `requirements.txt` pins `xgboost==1.7.6` which bundles its own libomp and avoids this issue.

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

## Key Finding

The dataset has a class imbalance (~61% unsafe, ~39% safe), so accuracy alone is misleading. F1-score and ROC-AUC are the metrics that matter.
