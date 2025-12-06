import argparse
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.preprocess import load_data

def explain(model_path, data_path, n_display=5):
    model = joblib.load(model_path)
    X, y = load_data(data_path)
    # take a small sample to compute SHAP (avoid huge compute)
    X_sample = X.sample(n=1000, random_state=42) if len(X) > 1000 else X
    explainer = shap.TreeExplainer(model) if hasattr(model, 'get_booster') or hasattr(model, 'feature_importances_') else shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    # summary plot
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_summary.png')
    print('Saved SHAP summary to results/shap_summary.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/fraud_model.joblib')
    parser.add_argument('--data', default='data/creditcard.csv')
    args = parser.parse_args()
    import os
    os.makedirs('results', exist_ok=True)
    explain(args.model, args.data)
