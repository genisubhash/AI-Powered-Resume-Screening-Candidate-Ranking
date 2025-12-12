# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import time
from pathlib import Path
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Google OAuth libraries (official)
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="AI Resume Screener (Google Login)", page_icon="🤖", layout="wide")

# --------------------------
# Load Google OAuth config from Streamlit secrets
# secrets.toml should contain a JSON-like client config or basic fields (see README below)
# --------------------------
# We'll build a client_config dict for google-auth-oauthlib Flow using secrets
CLIENT_ID = st.secrets["google_oauth"]["client_id"]
CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
REDIRECT_URI = st.secrets["google_oauth"]["redirect_uri"]   # must match exactly in Google Console

CLIENT_CONFIG = {
    "web": {
        "client_id": CLIENT_ID,
        "project_id": "streamlit-oauth",
        "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI],
    }
}

# --------------------------
# Utilities (text cleaning, pdf extract)
# --------------------------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\-\+\#\.\,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except:
            pass
    return text

# --------------------------
# Model helpers (same as your original)
# --------------------------
MODEL_TFIDF = Path("tfidf.pkl")
MODEL_CLF = Path("clf.pkl")

def train_model(df):
    st.info("🔄 Cleaning dataset...")
    df["Resume"] = df["Resume"].astype(str).apply(clean_text)

    st.info("🔄 Converting text to features (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(df["Resume"])
    y = df["Category"]

    st.info("🔄 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.info("🔄 Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    # Save models
    pickle.dump(tfidf, open(MODEL_TFIDF, "wb"))
    pickle.dump(clf, open(MODEL_CLF, "wb"))

    return accuracy

# --------------------------
# Simple skill extraction (optional)
# --------------------------
COMMON_SKILLS = [
    "python","java","c++","c#","sql","nosql","pandas","numpy","scikit-learn","tensorflow",
    "keras","pytorch","deep learning","machine learning","nlp","natural language processing",
    "computer vision","spark","hadoop","etl","tableau","power bi","excel","aws","azure","gcp",
    "docker","kubernetes","git","rest api","flask","django","linux","bash","matlab","r"
]

def extract_skills(text):
    t = " " + clean_text(text) + " "
    found = set()
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, t):
            found.add(skill.lower())
    return sorted(found)

# --------------------------
# OAuth helpers (using google-auth-oauthlib Flow)
# --------------------------
# We'll create an authorization URL and redirect the user to Google.
# After Google redirects back with ?code=..., we'll fetch tokens and verify id_token.

def make_flow(state=None):
    flow = Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=["openid", "email", "profile"],
        redirect_uri=REDIRECT_URI
    )
    if state:
        flow.oauth2session.state = state
    return flow

def get_authorization_url():
    flow = make_flow()
    auth_url, state = flow.authorization_url(prompt="consent", access_type="offline", include_granted_scopes="true")
    return auth_url, state

def exchange_code_for_user(code):
    # code is the full redirect URL or the code value depending on how called.
    # google-auth-oauthlib expects the full authorization_response URL.
    # Build flow again and call fetch_token(authorization_response=...)
    flow = make_flow()
    # Build full URL that was used to redirect back (Streamlit gives us query params only)
    # We'll construct a minimal redirect URL: redirect_uri + "?code=...&scope=..."
    # But to be safe we use the code form:
    try:
        # If we have full authorization_response in code param, use it:
        flow.fetch_token(code=code)
    except TypeError:
        # Some versions expect authorization_response:
        auth_response = f"{REDIRECT_URI}?code={code}"
        flow.fetch_token(authorization_response=auth_response)
    credentials = flow.credentials
    # Validate & parse the id_token to get user info
    request = grequests.Request()
    idinfo = None
    if credentials and getattr(credentials, "id_token", None):
        try:
            idinfo = id_token.verify_oauth2_token(credentials.id_token, request, CLIENT_ID)
            # idinfo contains 'email', 'email_verified', 'name', 'picture', etc.
        except Exception as e:
            st.error(f"Failed to verify ID token: {e}")
            return None
    else:
        st.error("No id_token present in credentials.")
        return None

    return idinfo

# --------------------------
# Streamlit UI & Flow
# --------------------------
st.title("AI Resume Screener — Google Sign-in")

# If user not logged in, show Sign in button and redirect flow
params = st.experimental_get_query_params()

if "user" not in st.session_state:
    # if Google redirected back, Google sends 'code' param
    if "code" in params:
        # params["code"] is a list (Streamlit returns lists)
        code = params["code"][0]
        # Exchange code for user info
        user_info = exchange_code_for_user(code)
        if user_info:
            st.session_state["user"] = user_info
            # clear query params to avoid re-processing
            st.experimental_set_query_params()
            st.experimental_rerun()
        else:
            st.error("Login failed. Try signing in again.")
            st.stop()
    else:
        auth_url, state = get_authorization_url()
        st.markdown("Please sign in with Google to continue:")
        st.markdown(f'<a href="{auth_url}"><button style="padding:10px 16px;">Sign in with Google</button></a>', unsafe_allow_html=True)
        st.stop()

# User is logged in
user = st.session_state["user"]
st.success(f"Logged in as {user.get('email')} — {user.get('name','')}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.experimental_rerun()

# Sidebar - only Predict (no dataset/training required unless you want it)
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Mode", ["🔮 Predict Resume"])

if mode == "🔮 Predict Resume":
    st.header("Upload resume (PDF/TXT) to analyze")
    uploaded_resume = st.file_uploader("Resume file", type=["pdf", "txt"])

    jd_file = st.file_uploader("Optional: Job Description (PDF/TXT) for ATS match", type=["pdf", "txt"])

    if uploaded_resume:
        # Load model files (tfidf.pkl, clf.pkl) — make sure they exist
        if not MODEL_TFIDF.exists() or not MODEL_CLF.exists():
            st.error("Model artifacts (tfidf.pkl, clf.pkl) not found. Please add them to the app folder.")
        else:
            tfidf = pickle.load(open(MODEL_TFIDF, "rb"))
            clf = pickle.load(open(MODEL_CLF, "rb"))

            if uploaded_resume.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_resume)
            else:
                resume_text = uploaded_resume.read().decode("utf-8", errors="ignore")

            if len(resume_text.strip()) < 20:
                st.error("Resume extraction failed or file is too small.")
            else:
                cleaned = clean_text(resume_text)
                features = tfidf.transform([cleaned])

                start = time.time()
                pred_label = clf.predict(features)[0]
                pred_prob = clf.predict_proba(features)[0]
                end = time.time()

                ranking_score = round(float(max(pred_prob) * 100), 2)

                st.success(f"Predicted Category: **{pred_label}**")
                st.info(f"Ranking Score: {ranking_score}%")
                st.info(f"Prediction Time: {round(end - start, 3)} sec")

                st.subheader("Top 3 Predictions")
                topk = pred_prob.argsort()[-3:][::-1]
                for idx in topk:
                    st.write(f"- {clf.classes_[idx]} — {round(pred_prob[idx]*100, 2)}%")

                # Skills
                found_skills = extract_skills(resume_text)
                st.subheader("Skills detected")
                st.write(", ".join(found_skills) if found_skills else "No common skills detected")

                # Optional JD matching (ATS)
                if jd_file:
                    if jd_file.type == "application/pdf":
                        jd_text = extract_text_from_pdf(jd_file)
                    else:
                        jd_text = jd_file.read().decode("utf-8", errors="ignore")

                    # simple TF-IDF cosine for ATS similarity
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity
                        vec = TfidfVectorizer(stop_words="english").fit_transform([clean_text(resume_text), clean_text(jd_text)])
                        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
                        st.success(f"ATS similarity score: {round(sim*100,2)}%")
                    except Exception as e:
                        st.error(f"Failed to compute ATS score: {e}")
