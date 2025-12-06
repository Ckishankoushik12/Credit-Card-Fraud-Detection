from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

def evaluate(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
    if probs is not None:
        print('AUC:', roc_auc_score(y, probs))
    print('Precision:', precision_score(y, preds))
    print('Recall:', recall_score(y, preds))
    print('F1:', f1_score(y, preds))
    print('\nClassification Report:\n', classification_report(y, preds))
