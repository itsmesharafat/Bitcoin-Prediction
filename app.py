from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'Bitcoin_Price_prediction_Model.keras'

# Load model from the project directory using the active TensorFlow/Keras runtime.
model = tf.keras.models.load_model(MODEL_PATH)

st.header('Bitcoin Price Prediction Model')
st.subheader('Bitcoin Price Data')
data = pd.DataFrame(yf.download('BTC-USD', '2015-01-01', '2023-11-30'))
if not data.empty:
    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    data.columns = [str(col).strip() for col in data.columns]

    # yfinance can vary a little between versions, so only drop columns that exist.
    extra_columns = [col for col in ['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'] if col in data.columns]
    data = data.drop(columns=extra_columns)
    if 'Close' not in data.columns:
        raise ValueError("The downloaded BTC data does not contain a 'Close' column.")
    data = data[['Close']].copy()
else:
    st.error('No BTC data was returned by yfinance.')
    st.stop()
st.write(data)

st.subheader('Bitcoin Line Chart')
st.line_chart(data)

train_data = data[:-100]
test_data = data[-200:]

scaler = MinMaxScaler(feature_range=(0,1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)
base_days = 100
x = []
y = []
for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i,0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0],x.shape[1],1))

st.subheader('Predicted vs Original Prices ')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1,1)
ys = scaler.inverse_transform(y.reshape(-1,1))
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)
st.write(chart_data)
st.subheader('Predicted vs Original Prices Chart ')
st.line_chart(chart_data)

m = y
z= []
future_days = 5
for i in range(base_days, len(m)+future_days):
    m = m.reshape(-1,1)
    inter = [m[-base_days:,0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
    pred = model.predict(inter)
    m = np.append(m ,pred)
    z = np.append(z, pred)
st.subheader('Predicted Future Days Bitcoin Price')
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1,1))
st.line_chart(z)