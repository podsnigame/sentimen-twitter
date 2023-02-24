import streamlit as st
import time
import numpy as np
import joblib
import plotly.express as px
import script.functions as fn
import pandas as pd

st.set_page_config(page_title="Model Information", page_icon="📈")

st.sidebar.markdown("📈 Model Information")

st.markdown("<h1 style='text-align: center;'>📈 Model Information</h1>", unsafe_allow_html=True)
st.write("halaman ini berisi mengenai informasi model yang tersedia pada aplikasi. anda bisa melihat bagaimana performa model dalam memprediksi sentiment baik dari waktu maupun hasil prediksi.")

st.markdown("<h3>⌛ Model Perfomance</h3>", unsafe_allow_html=True)
st.caption("Perfomance model dihitung berdasarkan akurasi dan waktu yang dibutuhkan model untuk memprediksi 100 data")
df_model = joblib.load("./assets/df_model.pkl")
fig = fn.plot_model_summary(df_model)
st.plotly_chart(fig,use_container_width=True,theme="streamlit")


st.markdown("<h3>🚀 Model Evaluation</h3>", unsafe_allow_html=True)
st.caption("Hasil evaluasi model berdasarkan data IndoNLU subset smsa pada validation split")

df = pd.read_csv("./assets/valid.csv")
option = st.selectbox('Pilih Model',["IndoBERT",'Naive Bayes','Logistic Regression','XGBoost','Catboost','SVM','Random Forest'],key = "model1")
clfr_fig = fn.plot_clfr(df_model,option,df)
conf_m_fig = fn.plot_confusion_matrix(df_model,option,df)
clfr,conf_m = st.columns([1,1])
with clfr:
    st.plotly_chart(clfr_fig,use_container_width=True,theme="streamlit")
with conf_m:
    st.plotly_chart(conf_m_fig,use_container_width=True,theme="streamlit")
st.caption("CLassification Report : Classification report merupakan metode evaluasi yang menyedakan data mengenai akurasi klasifikasi, recall, precision, dan F1 score.") 
st.caption("Confusion Matrix : mengukur jumlah prediksi benar dan salah yang dibuat oleh model yang berguna  untuk menunjukkan kinerja dari model untuk setiap kelas") 