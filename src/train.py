import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from src.preprocess import load_data, train_val_split, resample_smote
from src.evaluate import evaluate
import os

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='xgboost')
    parser.add_argument('--data', default=r'C:\Users\C Kishan Koushik\Desktop\Creditcard fraud ML\data\creditcard.csv')
    parser.add_argument('--out', default='models/fraud_model.joblib')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    X, y = load_data(args.data)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    X_res, y_res = resample_smote(X_train, y_train)

    if args.model == 'xgboost':
        model = train_xgboost(X_res, y_res)
    elif args.model == 'rf':
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_res, y_res)
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_res, y_res)

    joblib.dump(model, args.out)
    print('Saved model to', args.out)
    evaluate(model, X_val, y_val)
