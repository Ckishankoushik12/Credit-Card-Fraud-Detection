import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title='Fraud Detection Demo', layout='centered')

st.title('Credit Card Fraud Detection — Demo')

model_path = os.path.join('..','models','fraud_model.joblib')
if not os.path.exists(model_path):
    st.warning('Model not found. Train a model first with `python src/train.py` and place it in models/')
else:
    model = joblib.load(model_path)

st.markdown('Upload a CSV with the same features as the dataset (without `Class`) to get predictions.')

uploaded = st.file_uploader('Upload CSV (first rows used for demo)', type=['csv'])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        X = df.copy()
        preds = model.predict(X)
        probs = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else np.zeros(len(preds))
        out = X.copy()
        out['fraud_pred'] = preds
        out['fraud_prob'] = probs
        st.write(out.head(20))
        st.download_button('Download predictions CSV', out.to_csv(index=False), file_name='predictions.csv')
    except Exception as e:
        st.error('Error processing file: ' + str(e))
