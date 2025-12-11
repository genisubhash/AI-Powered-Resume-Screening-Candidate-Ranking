import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import time
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="AI Resume Screener", page_icon="🤖", layout="wide")

# ---------------------------------------------------------
# CLEANING FUNCTION
# ---------------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        try:
            text += page.extract_text()
        except:
            pass
    return text


# ---------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------
def train_model(df):
    st.info("🔄 Cleaning dataset...")
    df["Resume"] = df["Resume"].apply(clean_text)

    st.info("🔄 Converting text to features (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(df["Resume"])
    y = df["Category"]

    st.info("🔄 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.info("🔄 Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    # Save models
    pickle.dump(tfidf, open("tfidf.pkl", "wb"))
    pickle.dump(clf, open("clf.pkl", "wb"))

    return accuracy


# ---------------------------------------------------------
# SIDEBAR MENU
# ---------------------------------------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Select Mode", ["📚 Train Model", "🔮 Predict Resume"])


# =========================================================
# 📌 TRAINING AREA
# =========================================================
if menu == "📚 Train Model":
    st.title("📚 Train Resume Classification Model")

    st.write("Upload dataset (CSV with columns: **Resume**, **Category**)")

    dataset = st.file_uploader("Upload Dataset", type=["csv"])

    if dataset:
        df = pd.read_csv(dataset)
        st.write("📄 Dataset Preview")
        st.dataframe(df.head())

        if "Resume" not in df.columns or "Category" not in df.columns:
            st.error("❌ Dataset must contain 'Resume' and 'Category' columns!")
            st.stop()

        if st.button("🚀 Train Model"):
            with st.spinner("Training model... Please wait ⏳"):
                accuracy = train_model(df)
            st.success(f"🎉 Model Trained Successfully! Accuracy: {round(accuracy*100, 2)}%")
            st.info("✅ Saved as tfidf.pkl and clf.pkl")


# =========================================================
# 📌 PREDICTION AREA
# =========================================================
elif menu == "🔮 Predict Resume":
    st.title("🔮 AI Resume Screening & Candidate Ranking")

    uploaded_file = st.file_uploader("📂 Upload Resume (PDF/TXT)", type=["pdf", "txt"])

    if uploaded_file:
        # Load models
        try:
            tfidf = pickle.load(open("tfidf.pkl", "rb"))
            clf = pickle.load(open("clf.pkl", "rb"))
        except:
            st.error("❌ Model not found. Please train the model first.")
            st.stop()

        # ---------------- READ FILE -----------------
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if len(resume_text.strip()) < 30:
            st.error("⚠ The uploaded file contains very little text or extraction failed.")
            st.stop()

        # ---------------- CLEANING -----------------
        cleaned = clean_text(resume_text)
        features = tfidf.transform([cleaned])

        # ---------------- PREDICT -----------------
        start = time.time()
        pred_label = clf.predict(features)[0]
        pred_prob = clf.predict_proba(features)[0]
        end = time.time()

        ranking_score = round(max(pred_prob) * 100, 2)

        st.success(f"🏷️ Predicted Category: **{pred_label}**")
        st.warning(f"⭐ Candidate Ranking Score: **{ranking_score}%**")
        st.info(f"⏱️ Prediction Time: {round(end - start, 3)} sec")

        # ---------------- TOP 3 -----------------
        st.subheader("📊 Top 3 Category Probabilities")

        top3 = pred_prob.argsort()[-3:][::-1]
        for idx in top3:
            st.write(f"🔹 **{clf.classes_[idx]}** — {round(pred_prob[idx] * 100, 2)}%")
