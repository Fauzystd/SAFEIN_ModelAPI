
# forecast_api.py
# Jalankan dengan: uvicorn forecast_api:app --reload --host 0.0.0.0 --port 8001

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np, joblib, json
import tensorflow as tf
from tensorflow import keras
import google.generativeai as genai


class AttentionLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feature_dim      = input_shape[-1]
        self.W_attention = self.add_weight('W_att', shape=(feature_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.b_attention = self.add_weight('b_att', shape=(self.units,), initializer='zeros', trainable=True)
        self.V           = self.add_weight('V_att', shape=(self.units, 1), initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        score   = tf.tanh(tf.matmul(inputs, self.W_attention) + self.b_attention)
        score   = tf.matmul(score, self.V)
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def get_config(self):
        return {**super().get_config(), 'units': self.units}


class WeightedMAELoss(keras.losses.Loss):
    def __init__(self, income_weight=1.0, expense_weight=1.5, **kwargs):
        super().__init__(**kwargs)
        self.income_weight  = income_weight
        self.expense_weight = expense_weight

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            self.income_weight  * tf.abs(y_true[:, 0] - y_pred[:, 0]) +
            self.expense_weight * tf.abs(y_true[:, 1] - y_pred[:, 1])
        )

    def get_config(self):
        return {**super().get_config(), 'income_weight': self.income_weight, 'expense_weight': self.expense_weight}


CUSTOM_OBJECTS = {'AttentionLayer': AttentionLayer, 'WeightedMAELoss': WeightedMAELoss}

MODEL      = keras.models.load_model('financial_health_forecaster.keras', custom_objects=CUSTOM_OBJECTS)
SCALER_X   = joblib.load('scaler_forecast_x.pkl')
SCALER_INC = joblib.load('scaler_income.pkl')
SCALER_EXP = joblib.load('scaler_expense.pkl')
with open('feature_cols_forecast.json') as f:
    FEATURE_COLS = json.load(f)

GEMINI_API_KEY = 'ISI_API_KEY_GEMINI_KAMU'
genai.configure(api_key=GEMINI_API_KEY)
GEMINI = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title='Financial Health Forecaster API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


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


@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'FinancialHealthForecaster v1.0'}


@app.post('/forecast', response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if len(req.monthly_history) != 3:
        raise HTTPException(status_code=400, detail='monthly_history harus berisi tepat 3 bulan data.')
    try:
        X_raw  = np.array([[m.get(c, 0.0) for c in FEATURE_COLS] for m in req.monthly_history], dtype=np.float32)
        X_norm = SCALER_X.transform(X_raw)[np.newaxis, :, :]
        pred   = MODEL.predict(X_norm, verbose=0)[0]

        pred_income  = float(SCALER_INC.inverse_transform([[pred[0]]])[0][0])
        pred_expense = max(float(SCALER_EXP.inverse_transform([[pred[1]]])[0][0]), 0)

        curr = req.monthly_history[-1]
        ci, ce = curr.get('income', 0), curr.get('expense', 0)
        cs = (ci - ce) / ci if ci > 0 else 0
        ps = (pred_income - pred_expense) / pred_income if pred_income > 0 else 0

        delta = ps - cs
        if delta > 0.05:    status = 'MEMBAIK'
        elif delta < -0.05: status = 'MEMBURUK'
        else:               status = 'STABIL'

        prompt = (
            f"Asisten keuangan personal. Status: {status}. "
            f"Income: Rp{ci:,} -> Rp{pred_income:,.0f}. "
            f"Expense: Rp{ce:,} -> Rp{pred_expense:,.0f}. "
            f"Savings rate: {cs*100:.1f}% -> {ps*100:.1f}%. "
            f"Berikan rekomendasi 2-3 kalimat dalam Bahasa Indonesia."
        )
        try:    rek = GEMINI.generate_content(prompt).text.strip()
        except: rek = 'Rekomendasi AI tidak tersedia saat ini.'

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
            'rekomendasi_ai'      : rek,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
