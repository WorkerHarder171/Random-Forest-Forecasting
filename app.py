import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats

st.title("Forecasting Penjualan AOKA")
st.sidebar.header("Upload Data")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataframe Asli")
    st.write(data.head(10))

    # Preprocessing
    data['TGL'] = pd.to_datetime(data['TGL'], format='%d/%m/%Y')
    data = data.reset_index(drop=True)
    data['JML_TERJUAL'].fillna(data['JML_TERJUAL'].median(), inplace=True)
    data['JML_TERJUAL'] = data['JML_TERJUAL'].astype(int)
    data = data.sort_values(by='TGL')
    data.set_index('TGL', inplace=True)
    data.drop(['NAMA_BARANG'], axis=1, inplace=True)

    st.write("Data Setelah Preprocessing")
    st.write(data.head(10))

    # Menampilkan grafik data awal
    st.subheader("Grafik Penjualan")
    fig, ax = plt.subplots(figsize=(8, 3))
    plt.plot(data.index, data['JML_TERJUAL'], label='Penjualan AOKA')
    plt.title('Penjualan AOKA')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Terjual')
    plt.legend()
    st.pyplot(fig)

    # Deteksi Outlier
    st.subheader("Deteksi Outlier")
    z = np.abs(stats.zscore(data['JML_TERJUAL']))
    outlier_mask = z > 3

    # Handling Outlier
    data_cleaned = data.copy()
    data_cleaned.loc[outlier_mask, 'JML_TERJUAL'] = data_cleaned['JML_TERJUAL'].median()

    st.write("Data Setelah Menghilangkan Outlier")
    st.write(data_cleaned.describe())

    # Test Stationarity
    result = adfuller(data_cleaned['JML_TERJUAL'])
    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])

    # Seasonal Decompose
    result = seasonal_decompose(data_cleaned['JML_TERJUAL'], model='additive')
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid if result.resid is not None else np.zeros(len(result.trend))

    data_cleaned['trend'] = trend.fillna(0)
    data_cleaned['seasonal'] = seasonal.fillna(0)
    data_cleaned['residual'] = residual.fillna(0)

    # Modeling
    X = data_cleaned.drop(['JML_TERJUAL'], axis=1)
    y = data_cleaned['JML_TERJUAL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    r2_test = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("Evaluasi Model")
    st.write(f"R² Test: {r2_test:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    # Prediksi 5 Hari ke Depan
    st.subheader("Prediksi Penjualan 5 Hari ke Depan")
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
    future_data = pd.DataFrame(future_dates, columns=['TGL']).set_index('TGL')

    future_data['trend'] = trend.dropna().iloc[-5:].values
    future_data['seasonal'] = seasonal.dropna().iloc[-5:].values
    future_data['residual'] = residual[-5:] if len(residual) > 5 else residual[-1:].values

    # Mengisi jika ada NaN
    future_data = future_data.fillna(0)
    future_data['prediksi'] = model.predict(future_data)

    # Menampilkan hasil prediksi
    st.write(future_data)
    st.line_chart(future_data['prediksi'])
