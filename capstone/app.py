import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan encoder
model = joblib.load('capstone/model_obesitas.pkl')
scaler = joblib.load('capstone/scaler.pkl')
label_encoders = joblib.load('capstone/label_encoders.pkl')
le_target = joblib.load('capstone/le_target.pkl')

input_features = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
]
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="centered")
st.title("Prediksi Tingkat Obesitas Berdasarkan Data Individu")

with st.form("form_prediksi"):
    Gender = st.selectbox('Jenis Kelamin', label_encoders['Gender'].classes_)
    Age = st.number_input('Usia', min_value=1, max_value=100, value=20)
    Height = st.number_input('Tinggi Badan (meter)', min_value=1.0, max_value=2.5, value=1.65, step=0.01)
    Weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=200, value=65)
    family_history_with_overweight = st.selectbox('Riwayat Keluarga Overweight', label_encoders['family_history_with_overweight'].classes_)
    FAVC = st.selectbox('Sering Konsumsi Makanan Tinggi Kalori?', label_encoders['FAVC'].classes_)
    FCVC = st.slider('Frekuensi Konsumsi Sayur (1-3)', min_value=1, max_value=3, value=2)
    NCP = st.slider('Jumlah Makan Besar per Hari', min_value=1, max_value=5, value=3)
    CAEC = st.selectbox('Kebiasaan Ngemil', label_encoders['CAEC'].classes_)
    SMOKE = st.selectbox('Merokok?', label_encoders['SMOKE'].classes_)
    CH2O = st.slider('Konsumsi Air (L/hari)', min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    SCC = st.selectbox('Memantau Kalori?', label_encoders['SCC'].classes_)
    FAF = st.slider('Frekuensi Aktivitas Fisik', min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    TUE = st.slider('Waktu di Depan Layar per Hari (jam)', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    CALC = st.selectbox('Konsumsi Alkohol', label_encoders['CALC'].classes_)
    MTRANS = st.selectbox('Transportasi Umum', label_encoders['MTRANS'].classes_)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_dict = {
        'Gender': Gender,
        'Age': Age,
        'Height': Height,
        'Weight': Weight,
        'family_history_with_overweight': family_history_with_overweight,
        'FAVC': FAVC,
        'FCVC': FCVC,
        'NCP': NCP,
        'CAEC': CAEC,
        'SMOKE': SMOKE,
        'CH2O': CH2O,
        'SCC': SCC,
        'FAF': FAF,
        'TUE': TUE,
        'CALC': CALC,
        'MTRANS': MTRANS
    }
    input_df = pd.DataFrame([input_dict])

    for col in cat_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)
    pred_label = le_target.inverse_transform(pred)[0]

    st.success(f"**Prediksi Tingkat Obesitas: {pred_label}**")
    st.write("---")
    st.write("Keterangan kelas prediksi:")
    st.write(", ".join(le_target.classes_))
