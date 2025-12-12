import streamlit as st
import pandas as pd
import pickle
import re
import time
from pathlib import Path
from PyPDF2 import PdfReader

# ML Imports
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Google OAuth Libraries (Official)
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

# -----------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="AI Resume Screener", page_icon="🤖", layout="wide")

# -----------------------------------------------------------
# LOAD GOOGLE OAUTH FROM STREAMLIT SECRETS
# -----------------------------------------------------------
CLIENT_ID = st.secrets["google_oauth"]["client_id"]
CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
REDIRECT_URI = st.secrets["google_oauth"]["redirect_uri"]

# Build OAuth client config for Flow
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

# -----------------------------------------------------------
# CLEAN TEXT
# -----------------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\-\+\#\.\,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# -----------------------------------------------------------
# PDF TEXT EXTRACTION
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# SKILL EXTRACTION
# -----------------------------------------------------------
COMMON_SKILLS = [
    "python","java","c++","c#","sql","nosql","pandas","numpy","scikit-learn",
    "tensorflow","keras","pytorch","deep learning","machine learning","nlp",
    "natural language processing","computer vision","spark","hadoop","etl",
    "tableau","power bi","excel","aws","azure","gcp","docker","kubernetes",
    "git","rest api","flask","django","linux","bash","matlab","r"
]

def extract_skills(text):
    t = " " + clean_text(text) + " "
    found = set()
    for s in COMMON_SKILLS:
        pattern = r"\b" + re.escape(s.lower()) + r"\b"
        if re.search(pattern, t):
            found.add(s.lower())
    return sorted(found)

# -----------------------------------------------------------
# GOOGLE OAUTH FLOW
# -----------------------------------------------------------
def make_flow(state=None):
    flow = Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"
        ],
        redirect_uri=REDIRECT_URI
    )
    if state:
        flow.oauth2session.state = state
    return flow

def get_authorization_url():
    flow = make_flow()
    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true"
    )
    return auth_url, state

def exchange_code_for_user(code):
    try:
        flow = make_flow()
        
        # Build auth response URL for token exchange
        auth_response = f"{REDIRECT_URI}?code={code}"
        flow.fetch_token(authorization_response=auth_response)

        credentials = flow.credentials

        request = grequests.Request()
        userinfo = id_token.verify_oauth2_token(
            credentials.id_token, request, CLIENT_ID
        )

        return userinfo

    except Exception as e:
        st.error(f"Login Verification Failed: {e}")
        return None

# -----------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------
TFIDF_PATH = Path("tfidf.pkl")
CLF_PATH = Path("clf.pkl")

def load_models():
    if TFIDF_PATH.exists() and CLF_PATH.exists():
        tfidf = pickle.load(open(TFIDF_PATH, "rb"))
        clf = pickle.load(open(CLF_PATH, "rb"))
        return tfidf, clf
    else:
        st.error("❌ Model files not found. Upload tfidf.pkl and clf.pkl to your app folder.")
        st.stop()

# -----------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------
st.title("🔐 AI Resume Screener")

params = st.query_params

if "user" not in st.session_state:

    if "code" in params:
        code = params["code"]
        user = exchange_code_for_user(code)

        if user:
            st.session_state["user"] = user
            st.query_params.clear()
            st.rerun()
        else:
            st.error("Google login failed. Try again.")
            st.stop()
    else:
        st.write("Please sign in to continue:")
        auth_url, state = get_authorization_url()
        st.markdown(
            f'<a href="{auth_url}"><button style="padding:10px 16px; border-radius:6px; background:#0A66C2; color:white;">Sign in with Google</button></a>',
            unsafe_allow_html=True,
        )
        st.stop()

# -----------------------------------------------------------
# USER LOGGED IN — MAIN APP
# -----------------------------------------------------------
user = st.session_state["user"]
st.success(f"Logged in as: {user.get('email')}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------
# RESUME PREDICTION UI
# -----------------------------------------------------------
st.header("🔮 AI Resume Screening & Ranking System")

uploaded_file = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"])
jd_file = st.file_uploader("Optional: Upload Job Description", type=["pdf", "txt"])

if uploaded_file:

    tfidf, clf = load_models()

    # Extract text
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

    if len(resume_text.strip()) < 30:
        st.error("Resume text could not be extracted.")
        st.stop()

    cleaned = clean_text(resume_text)
    features = tfidf.transform([cleaned])

    # Predict
    start = time.time()
    pred_label = clf.predict(features)[0]
    pred_prob = clf.predict_proba(features)[0]
    end = time.time()

    ranking_score = round(max(pred_prob) * 100, 2)

    st.success(f"🏷 Predicted Job Role: **{pred_label}**")
    st.warning(f"⭐ Candidate Ranking Score: **{ranking_score}%**")
    st.info(f"⏱ Processing Time: {round(end-start, 3)} sec")

    # Top 3 probabilities
    st.subheader("📊 Top 3 Predictions")
    top3 = pred_prob.argsort()[-3:][::-1]
    for i in top3:
        st.write(f"- **{clf.classes_[i]}** — {round(pred_prob[i] * 100, 2)}%")

    # Skills
    st.subheader("🧰 Skills Found in Resume")
    skills = extract_skills(resume_text)
    st.write(", ".join(skills) if skills else "No skills detected")

    # JD match
    if jd_file:
        if jd_file.type == "application/pdf":
            jd_text = extract_text_from_pdf(jd_file)
        else:
            jd_text = jd_file.read().decode("utf-8", errors="ignore")

        vec = TfidfVectorizer(stop_words="english")
        tfidf_pair = vec.fit_transform([clean_text(resume_text), clean_text(jd_text)])
        sim = cosine_similarity(tfidf_pair[0:1], tfidf_pair[1:2])[0][0]

        st.subheader("📄 ATS Job Description Match")
        st.info(f"Similarity Score: **{round(sim*100,2)}%**")
