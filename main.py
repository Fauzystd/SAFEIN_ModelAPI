# main.py — Financial Health API
# Menggabungkan Model 1 (Classifier) dan Model 2 (Forecaster) dalam satu server
# Deploy: uvicorn main:app --host 0.0.0.0 --port $PORT

import os
import json
import joblib
import numpy as np
from typing import Dict, List

import tensorflow as tf
from tensorflow import keras

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai


# ── Load semua artefak saat server startup ───────────────────────

print("Memuat Model 1 (Classifier)...")
MODEL_CLASSIFIER  = tf.saved_model.load('classifier_savedmodel')
SCALER_CLASSIFIER = joblib.load('scaler.pkl')
with open('feature_cols.json')  as f: FEATURE_COLS_CLASSIFIER = json.load(f)
with open('label_classes.json') as f: LABEL_CLASSES           = json.load(f)
print("Model 1 berhasil dimuat.")

print("Memuat Model 2 (Forecaster)...")
MODEL_FORECASTER = tf.saved_model.load('forecaster_savedmodel')
SCALER_X         = joblib.load('scaler_forecast_x.pkl')
SCALER_INCOME    = joblib.load('scaler_income.pkl')
SCALER_EXPENSE   = joblib.load('scaler_expense.pkl')
with open('feature_cols_forecast.json') as f: FEATURE_COLS_FORECASTER = json.load(f)
print("Model 2 berhasil dimuat.")

print("Menghubungkan Gemini...")
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)
print("Gemini terhubung.")


# ── FastAPI app ──────────────────────────────────────────────────

app = FastAPI(
    title='Financial Health API',
    description='API untuk klasifikasi kondisi keuangan dan prediksi bulan depan.',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Schema request & response ────────────────────────────────────

class ClassifyRequest(BaseModel):
    features: Dict[str, float]

class ClassifyResponse(BaseModel):
    label        : str
    confidence   : float
    probabilities: Dict[str, float]
    rekomendasi  : str

class ForecastRequest(BaseModel):
    monthly_history: List[Dict[str, float]]

class ForecastResponse(BaseModel):
    status              : str
    current_income      : int
    current_expense     : int
    pred_income         : int
    pred_expense        : int
    income_change_pct   : float
    expense_change_pct  : float
    current_savings_rate: float
    pred_savings_rate   : float
    rekomendasi_ai      : str


# ── Helper: generate rekomendasi Gemini ─────────────────────────

def generate_rekomendasi(context: str) -> str:
    if not GEMINI_API_KEY:
        return "API key Gemini belum dikonfigurasi."
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=context
        )
        return response.text.strip()
    except Exception as e:
        return f"Rekomendasi AI tidak tersedia: {str(e)}"


# ── Endpoint: health check ───────────────────────────────────────

@app.get('/health')
def health_check():
    return {
        'status' : 'ok',
        'model_1': 'FinancialHealthClassifier v1.0',
        'model_2': 'FinancialHealthForecaster v1.0',
    }


# ── Endpoint: klasifikasi kondisi bulan ini ──────────────────────

@app.post('/classify', response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    try:
        X    = np.array([[req.features.get(c, 0.0) for c in FEATURE_COLS_CLASSIFIER]], dtype=np.float32)
        X_sc = SCALER_CLASSIFIER.transform(X).astype(np.float32)

        #probs = MODEL_CLASSIFIER(X_sc, training=False).numpy()[0]
        infer = MODEL_CLASSIFIER.signatures['serving_default']
        output = infer(tf.constant(X_sc))
        #kode di atas di ubah

        probs = list(output.values())[0].numpy()[0]
        pred  = int(np.argmax(probs))
        label = LABEL_CLASSES[pred]
        conf  = float(probs[pred])

        sr = req.features.get('savings_rate', 0)
        er = req.features.get('expense_ratio', 0)

        prompt = (
            f"Kamu adalah asisten keuangan personal. "
            f"Kondisi keuangan pengguna saat ini: {label} dengan confidence {conf*100:.1f}%. "
            f"Savings rate: {sr*100:.1f}%, expense ratio: {er*100:.1f}%. "
            f"Berikan rekomendasi singkat 2-3 kalimat dalam Bahasa Indonesia yang actionable."
        )
        rekomendasi = generate_rekomendasi(prompt)

        return {
            'label'        : label,
            'confidence'   : round(conf, 4),
            'probabilities': {cls: round(float(p), 4) for cls, p in zip(LABEL_CLASSES, probs)},
            'rekomendasi'  : rekomendasi,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint: prediksi kondisi bulan depan ───────────────────────

@app.post('/forecast', response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if len(req.monthly_history) != 3:
        raise HTTPException(status_code=400, detail='monthly_history harus berisi tepat 3 bulan data.')
    try:
        X_raw  = np.array([[m.get(c, 0.0) for c in FEATURE_COLS_FORECASTER] for m in req.monthly_history], dtype=np.float32)
        X_norm = SCALER_X.transform(X_raw).astype(np.float32)
        X_in   = X_norm[np.newaxis, :, :]

        #pred         = MODEL_FORECASTER(X_in, training=False).numpy()[0]
        infer = MODEL_FORECASTER.signatures['serving_default']
        output = infer(tf.constant(X_in))
        pred = list(output.values())[0].numpy()[0]
        #kode di atas di ubah
        pred_income  = float(SCALER_INCOME.inverse_transform([[pred[0]]])[0][0])
        pred_expense = max(float(SCALER_EXPENSE.inverse_transform([[pred[1]]])[0][0]), 0)

        curr = req.monthly_history[-1]
        ci   = curr.get('income', 0)
        ce   = curr.get('expense', 0)
        cs   = (ci - ce) / ci if ci > 0 else 0
        ps   = (pred_income - pred_expense) / pred_income if pred_income > 0 else 0

        delta = ps - cs
        if delta > 0.05:    status = 'MEMBAIK'
        elif delta < -0.05: status = 'MEMBURUK'
        else:               status = 'STABIL'

        prompt = (
            f"Kamu adalah asisten keuangan personal. "
            f"Prediksi kondisi keuangan bulan depan: {status}. "
            f"Income: Rp{ci:,} menjadi Rp{pred_income:,.0f} ({(pred_income-ci)/ci*100:+.1f}%). "
            f"Expense: Rp{ce:,} menjadi Rp{pred_expense:,.0f} ({(pred_expense-ce)/ce*100:+.1f}%). "
            f"Savings rate: {cs*100:.1f}% menjadi {ps*100:.1f}%. "
            f"Berikan rekomendasi singkat 2-3 kalimat dalam Bahasa Indonesia yang actionable."
        )
        rekomendasi_ai = generate_rekomendasi(prompt)

        return {
            'status'              : status,
            'current_income'      : int(ci),
            'current_expense'     : int(ce),
            'pred_income'         : int(pred_income),
            'pred_expense'        : int(pred_expense),
            'income_change_pct'   : round((pred_income  - ci) / ci  * 100, 2) if ci  > 0 else 0,
            'expense_change_pct'  : round((pred_expense - ce) / ce  * 100, 2) if ce  > 0 else 0,
            'current_savings_rate': round(cs, 4),
            'pred_savings_rate'   : round(ps, 4),
            'rekomendasi_ai'      : rekomendasi_ai,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))