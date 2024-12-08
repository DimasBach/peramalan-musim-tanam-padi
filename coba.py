import streamlit as st
from joblib import load

try:
    model_terbaik = load("best_model_neurons.pkl")
    st.write("Model berhasil dimuat:", model_terbaik)
except Exception as e:
    st.write(f"Error saat membuka file joblib: {e}")