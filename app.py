import os
import re
import string

import joblib
import pandas as pd
import streamlit as st
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

MODEL_DIR = "model"  # taruh model.pkl, vectorizer.pkl, label_encoder.pkl di sini

st.set_page_config(page_title="Analisis Sentimen Bahasa Indonesia", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, vectorizer, encoder


@st.cache_resource
def load_sastrawi():
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover


def preprocess(text, stemmer, stopword_remover):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


def predict(text, model, vectorizer, encoder, stemmer, stopword_remover):
    clean = preprocess(text, stemmer, stopword_remover)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = encoder.inverse_transform([pred])[0]
    probs = dict(zip(encoder.classes_, proba))
    return label, probs, clean


model, vectorizer, encoder = load_artifacts()
stemmer, stopword_remover = load_sastrawi()

st.title("Analisis Sentimen Bahasa Indonesia")
st.caption("TF-IDF + Logistic Regression, preprocessing dengan Sastrawi")

tab_single, tab_batch = st.tabs(["Teks Tunggal", "Upload CSV"])

with tab_single:
    text_input = st.text_area("Masukkan teks (ulasan, komentar, dll.)", height=120)
    if st.button("Analisis", type="primary"):
        if text_input.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            label, probs, clean = predict(text_input, model, vectorizer, encoder, stemmer, stopword_remover)
            st.subheader(f"Sentimen: {label.upper()}")
            st.bar_chart(probs)
            with st.expander("Lihat teks setelah preprocessing"):
                st.code(clean)

with tab_batch:
    uploaded = st.file_uploader("Upload file CSV dengan kolom teks", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col = st.selectbox("Pilih kolom teks", df.columns)
        if st.button("Analisis Semua"):
            with st.spinner("Memproses..."):
                results = df[col].astype(str).apply(
                    lambda t: predict(t, model, vectorizer, encoder, stemmer, stopword_remover)
                )
                df["sentimen"] = results.apply(lambda r: r[0])
                df["confidence"] = results.apply(lambda r: f"{max(r[1].values()) * 100:.1f}%")
            st.dataframe(df)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download hasil", csv_bytes, "hasil_sentimen.csv", "text/csv")
