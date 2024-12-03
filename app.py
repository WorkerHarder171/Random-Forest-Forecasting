# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load dataset
st.title("Aplikasi Prediksi Penjualan Menggunakan Random Forest")
uploaded_file = st.file_uploader("Upload file CSV dataset penjualan", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Preprocessing
    data['TGL'] = pd.to_datetime(data['TGL'], format='%d/%m/%Y')
    data.set_index('TGL', inplace=True)
    median_value = data['JML_TERJUAL'].median()
    data['JML_TERJUAL'].fillna(median_value, inplace=True)
    data['JML_TERJUAL'] = data['JML_TERJUAL'].astype(int)

    st.write("**Dataset yang diunggah:**")
    st.dataframe(data.head())

    # Plot Data Asli
    st.subheader("Grafik Penjualan")
    st.line_chart(data['JML_TERJUAL'])

    # Decomposisi Time Series
    result = seasonal_decompose(data['JML_TERJUAL'], model='additive')
    trend = result.trend.fillna(0)
    seasonal = result.seasonal
    residual = result.resid.fillna(0)

    # Tambahkan komponen ke data
    data['trend'] = trend
    data['seasonal'] = seasonal
    data['residual'] = residual

    # Split Data
    X = data[['trend', 'seasonal', 'residual']]
    y = data['JML_TERJUAL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # Evaluasi
    st.subheader("Evaluasi Model")
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    st.write(f"R² Test: {r2_test:.2f}")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")

    # Prediksi 5 Hari ke Depan
    st.subheader("Prediksi Penjualan 5 Hari ke Depan")
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
    future_data = pd.DataFrame(future_dates, columns=['TGL']).set_index('TGL')

    future_data['trend'] = trend[-5:].values
    future_data['seasonal'] = seasonal[-5:].values
    future_data['residual'] = residual[-5:].values
    future_data['prediksi'] = model.predict(future_data[['trend', 'seasonal', 'residual']])

    st.write(future_data)
    st.line_chart(future_data['prediksi'])
