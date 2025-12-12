#######################################################################
# AI Resume Screener - Corporate UI Edition
# Features:
# ✔ Google Authentication
# ✔ Admin Dashboard
# ✔ Resume Upload -> Prediction
# ✔ Skill Extraction + Missing Skills
# ✔ ATS Score + JD Matching
# ✔ Top 5 Role Probabilities
# ✔ Professional Corporate UI
#######################################################################

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
import time
import pickle
from datetime import datetime
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from authlib.integrations.requests_client import OAuth2Session

# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide",
    page_icon="🤖"
)

# -------------------------------------------------------------------
# DATABASE INIT
# -------------------------------------------------------------------
DB = "resume_history.db"

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            name TEXT,
            role TEXT,
            ranking REAL,
            ats REAL,
            skills TEXT,
            missing TEXT,
            time TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            name TEXT,
            time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------------------------------------------------
# GOOGLE AUTH CONFIG
# -------------------------------------------------------------------
OAUTH = st.secrets["google_oauth"]
ADMIN_EMAILS = st.secrets["admin"]["emails"]

CLIENT_ID = OAUTH["client_id"]
CLIENT_SECRET = OAUTH["client_secret"]
REDIRECT_URI = OAUTH["redirect_uri"]

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# -------------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def extract_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        try:
            tmp = page.extract_text()
            if tmp:
                text += tmp + "\n"
        except:
            pass
    return text

# Skill list & job-role mapping
SKILLS = [
    "python","java","c++","html","css","javascript","react","nodejs","mongodb",
    "sql","excel","power bi","pandas","numpy","machine learning","deep learning",
    "nlp","aws","azure","gcp","docker","kubernetes","tensorflow","pytorch",
    "flask","django","api","git","linux"
]

JOB_SKILLS = {
    "python developer": ["python","django","flask","api","pandas","numpy"],
    "java developer": ["java","spring","hibernate","mysql"],
    "web developer": ["html","css","javascript","react"],
    "frontend developer": ["html","css","javascript","react"],
    "backend developer": ["nodejs","mongodb","api"],
    "full stack developer": ["react","nodejs","html","css","javascript"],
    "data analyst": ["excel","power bi","sql","pandas","numpy"],
    "data scientist": ["python","machine learning","pandas","numpy","tensorflow"],
    "ml engineer": ["python","pytorch","tensorflow","deep learning"],
    "devops engineer": ["aws","docker","kubernetes","linux"]
}

def extract_skills(text):
    t = clean_text(text)
    return [s for s in SKILLS if s in t]

def ats_score(resume, jd):
    t1, t2 = clean_text(resume), clean_text(jd)
    tfidf = TfidfVectorizer().fit([t1, t2])
    v = tfidf.transform([t1, t2])
    sim = cosine_similarity(v[0:1], v[1:2])[0][0]
    return round(sim * 100, 2)

# -------------------------------------------------------------------
# OAUTH HANDLERS
# -------------------------------------------------------------------
def build_oauth():
    return OAuth2Session(
        CLIENT_ID,
        CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="openid email profile"
    )

def login_button():
    oauth = build_oauth()
    url, state = oauth.create_authorization_url(AUTH_URL)
    st.session_state["state"] = state
    st.markdown(
        f"""
        <a href="{url}">
            <button style="
                padding:12px 22px;
                font-size:16px;
                border-radius:8px;
                background:#0A66C2;
                color:white;
                border:none;">
                Sign in with Google
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

def fetch_user(code):
    oauth = build_oauth()
    token = oauth.fetch_token(TOKEN_URL, code=code, client_secret=CLIENT_SECRET)
    resp = oauth.get(USERINFO_URL, token=token)
    return resp.json()

# -------------------------------------------------------------------
# ADMIN DASHBOARD
# -------------------------------------------------------------------
def admin_dashboard():
    st.title("📊 Admin Dashboard")

    conn = sqlite3.connect(DB)
    logins = pd.read_sql_query("SELECT * FROM logins ORDER BY id DESC", conn)
    results = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
    conn.close()

    st.subheader("User Login History")
    st.dataframe(logins, use_container_width=True)

    st.subheader("Resume Screening History")
    st.dataframe(results, use_container_width=True)

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():

    st.markdown("""
        <h1 style='text-align:center; color:#003566;'>AI Resume Screening System</h1>
        <p style='text-align:center; font-size:18px; color:#555;'>
        Corporate Edition — Powered by Machine Learning & NLP
        </p>
        <br>
    """, unsafe_allow_html=True)

    params = st.experimental_get_query_params()

    # ------------------------ LOGIN FLOW ------------------------
    if "user" not in st.session_state:
        if "code" in params:
            user = fetch_user(params["code"][0])
            st.session_state["user"] = user

            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("INSERT INTO logins(email,name,time) VALUES(?,?,?)",
                        (user["email"], user.get("name",""), str(datetime.now())))
            conn.commit()
            conn.close()

            st.experimental_set_query_params()
            st.experimental_rerun()

        else:
            st.write("### Please Sign In")
            login_button()
            return

    # ------------------------ USER LOGGED IN ------------------------
    user = st.session_state["user"]
    st.sidebar.success(f"Logged in: {user['email']}")

    if user["email"] in ADMIN_EMAILS:
        if st.sidebar.button("Admin Dashboard"):
            admin_dashboard()
            return

    # ------------------------ MODEL LOADING ------------------------
    try:
        tfidf = pickle.load(open("tfidf.pkl","rb"))
        clf = pickle.load(open("clf.pkl","rb"))
    except:
        st.error("Model files not found! (tfidf.pkl, clf.pkl)")
        return

    # ------------------------ RESUME UPLOAD UI ------------------------
    st.subheader("📂 Upload Resume")
    resume_file = st.file_uploader("Upload a resume (PDF/TXT)", type=["pdf","txt"])

    jd_file = st.file_uploader("📄 Optional: Upload Job Description", type=["pdf","txt"])

    if resume_file:

        # Extract resume text
        if resume_file.type == "application/pdf":
            text = extract_pdf(resume_file)
        else:
            text = resume_file.read().decode("utf-8",errors="ignore")

        cleaned = clean_text(text)
        vec = tfidf.transform([cleaned])

        start = time.time()
        pred = clf.predict(vec)[0]
        prob = clf.predict_proba(vec)[0]
        end = time.time()

        ranking = round(max(prob)*100,2)

        # --------------------- UI CARD: ROLE PREDICTION ---------------------
        st.markdown("""
            <div style="
                background:#F8F9FA;
                padding:20px;
                border-radius:10px;
                border:1px solid #DDD;">
                <h3 style="color:#0A66C2;">Predicted Job Role</h3>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"### 🎯 {pred}")

        st.metric(label="Candidate Ranking Score", value=f"{ranking}%")

        # --------------------- TOP 5 ROLES ---------------------
        st.subheader("📊 Top 5 Role Probabilities")
        top5 = prob.argsort()[-5:][::-1]
        for i in top5:
            st.write(f"- **{clf.classes_[i]}** — {round(prob[i]*100,2)}%")

        # --------------------- SKILLS ---------------------
        found = extract_skills(text)
        required = JOB_SKILLS.get(pred.lower(), [])
        missing = [s for s in required if s not in found]

        st.subheader("🧠 Skills Found")
        st.write(", ".join(found) if found else "No major skills detected.")

        st.subheader("📌 Required Skills")
        st.write(", ".join(required) if required else "Not available for this role.")

        st.subheader("❗ Missing Skills")
        st.write(", ".join(missing) if missing else "None — Looks Good!")

        # --------------------- ATS SCORE ---------------------
        if jd_file:
            if jd_file.type == "application/pdf":
                jd_text = extract_pdf(jd_file)
            else:
                jd_text = jd_file.read().decode("utf-8")

            score = ats_score(text, jd_text)
            st.success(f"📑 ATS Score: {score}%")
        else:
            score = None

        # --------------------- SAVE RESULT ---------------------
        if st.button("Save Result"):
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO results(email,name,role,ranking,ats,skills,missing,time)
                VALUES(?,?,?,?,?,?,?,?)
            """, (
                user["email"], user.get("name",""),
                pred, ranking, score,
                "|".join(found), "|".join(missing),
                str(datetime.now())
            ))
            conn.commit()
            conn.close()
            st.success("Saved Successfully!")

# Run App
main()
