import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Aplikasi Prediksi Kualitas Udara")

# Upload file
uploaded_file = st.file_uploader("Unggah file CSV untuk analisis", type="csv")
if uploaded_file is not None:
    # Membaca file CSV
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.dataframe(data)

    # Tambahkan visualisasi atau analisis
    st.write("Statistik Deskriptif:")
    st.write(data.describe())

    # Contoh visualisasi
    st.write("Visualisasi Data:")
    plt.figure(figsize=(10, 5))
    data.hist()
    st.pyplot(plt)
