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
from authlib.integrations.requests_client import OAuth2Session

# --------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------
st.set_page_config(page_title="AI Resume Screener", page_icon="🤖", layout="wide")

# --------------------------------------------
# GOOGLE OIDC CONFIG
# --------------------------------------------
OAUTH = st.secrets["google_oauth"]

CLIENT_ID = OAUTH["client_id"]
CLIENT_SECRET = OAUTH["client_secret"]
REDIRECT_URI = OAUTH["redirect_uri"]

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

# --------------------------------------------
# OIDC FUNCTIONS
# --------------------------------------------
def get_oauth_client(state=None):
    return OAuth2Session(
        CLIENT_ID,
        CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
        state=state
    )

def login_button():
    oauth = get_oauth_client()
    auth_url, state = oauth.create_authorization_url(AUTH_URL)
    st.session_state["state"] = state

    st.markdown(
        f"""
        <a href="{auth_url}">
            <button style="padding:10px 20px; font-size:16px; border-radius:8px; background:#0A66C2; color:white; border:none;">
                🔐 Sign in with Google
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

def fetch_user(code):
    oauth = get_oauth_client()

    # ⭐ IMPORTANT FIX — REQUIRED FOR GOOGLE OIDC
    token = oauth.fetch_token(
        TOKEN_URL,
        code=code,
        client_secret=CLIENT_SECRET,
        grant_type="authorization_code",
        include_client_id=True
    )

    resp = oauth.get(USERINFO_URL, token=token)
    return resp.json() if resp.status_code == 200 else None

# --------------------------------------------
# TEXT CLEANING
# --------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# --------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        try:
            text += page.extract_text()
        except:
            pass
    return text

# --------------------------------------------
# TRAINING MODEL
# --------------------------------------------
def train_model(df):
    st.info("🔄 Cleaning dataset...")
    df["Resume"] = df["Resume"].apply(clean_text)

    st.info("🔄 Extracting features...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(df["Resume"])
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.info("🔄 Training...")
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    pickle.dump(tfidf, open("tfidf.pkl", "wb"))
    pickle.dump(clf, open("clf.pkl", "wb"))

    return accuracy

# --------------------------------------------
# AUTHENTICATION GATE
# --------------------------------------------
st.title("🔐 Welcome to AI Resume Screener")

params = st.experimental_get_query_params()

if "user" not in st.session_state:

    # Returned from Google OAuth
    if "code" in params:
        user = fetch_user(params["code"][0])

        if user:
            st.session_state["user"] = user
            st.experimental_set_query_params()
            st.experimental_rerun()
        else:
            st.error("Google login failed.")
    else:
        st.write("Please sign in to continue.")
        login_button()
        st.stop()

# --------------------------------------------
# USER LOGGED IN
# --------------------------------------------
user = st.session_state["user"]
st.success(f"Logged in as: {user['email']}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.experimental_rerun()

# --------------------------------------------
# SIDEBAR
# --------------------------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Select Mode", ["📚 Train Model", "🔮 Predict Resume"])

# --------------------------------------------
# TRAIN MODEL
# --------------------------------------------
if menu == "📚 Train Model":
    st.title("📚 Train Resume Classification Model")
    dataset = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if dataset:
        df = pd.read_csv(dataset)
        st.dataframe(df.head())

        if st.button("🚀 Train Model"):
            acc = train_model(df)
            st.success(f"Model trained! Accuracy: {acc*100:.2f}%")

# --------------------------------------------
# PREDICT RESUME
# --------------------------------------------
elif menu == "🔮 Predict Resume":

    st.title("🔮 AI Resume Screening & Ranking")
    uploaded_file = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"])

    if uploaded_file:

        # Load saved model
        try:
            tfidf = pickle.load(open("tfidf.pkl", "rb"))
            clf = pickle.load(open("clf.pkl", "rb"))
        except:
            st.error("Model not found. Train the model first.")
            st.stop()

        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")

        cleaned = clean_text(resume_text)
        X = tfidf.transform([cleaned])

        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        score = round(max(proba) * 100, 2)

        st.success(f"Predicted Category: **{pred}**")
        st.warning(f"Ranking Score: **{score}%**")

        st.subheader("Top 3 Predictions:")
        top3 = proba.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"- {clf.classes_[i]} — {proba[i]*100:.2f}%")
