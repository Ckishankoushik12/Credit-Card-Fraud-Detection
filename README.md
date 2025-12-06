# Credit Card Fraud Detection (ML)

Built an end-to-end credit-card fraud detection pipeline using XGBoost and SMOTE with model explainability (SHAP) and a Streamlit demo.

## Contents
- `src/` : preprocessing, training, evaluation, explainability scripts
- `models/` : trained model artifacts (gitignored)
- `app/` : Streamlit demo to upload transactions and get predictions + SHAP plots
- `notebooks/` : EDA and experiments (optional)
- `data/` : instructions to download dataset (do not commit large raw data)

## Dataset
- Place `creditcard.csv` inside `data/`

## Quickstart (local)
1. Create virtual environment and install:
   ```bash
   python -m venv venv
   source venv/bin/activate         # macOS/Linux
   venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```
2. Put `creditcard.csv` into `data/` (see dataset link).
3. Train a model (XGBoost):
   ```bash
   python src/train.py --model xgboost --data data/creditcard.csv
   ```
4. Run Streamlit demo:
   ```bash
   streamlit run app/streamlit_app.py
   ```
5. (Optional) Generate SHAP explanations:
   ```bash
   python src/explain.py --model models/fraud_model.joblib --data data/creditcard.csv
   ```

## Results (fill in after you run experiments)
- Validation AUC: **0.9687**
- Validation Recall: **0.8878**

## Project setup — quick steps (Windows)

1. Clone the repo:
```powershell
git clone <your-repo-url>
cd fraud-detection-ml
```

2. Create a Python virtual environment and install dependencies:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Place dataset:
- Download the dataset file `creditcard.csv`, and run it:
Ex: "C:\Users\C Kishan Koushik\Desktop\Creditcard fraud ML\data\creditcard.csv"(creditcard.csv path)

4. Train the model (example):
```powershell
python src/train.py --model xgboost --data "C:\Users\C Kishan Koushik\Desktop\Creditcard fraud ML\data\creditcard.csv"
```

5. Run Streamlit demo:
```powershell
streamlit run app/streamlit_app.py
```

## Helpful scripts

- `setup_windows.bat` — creates venv, activates it, and installs requirements (Windows)
- `setup_unix.sh` — same for macOS/Linux

## CI / Tests

A GitHub Actions workflow is included at `.github/workflows/ci.yml` which runs flake8 and basic tests on push/PR.
