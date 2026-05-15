import pickle as pkl
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer



myModel=pkl.load(open("myModel.pkl","rb"))
labelEncoder=pkl.load(open("labelEncoder.pkl","rb"))
vector=pkl.load(open("vector.pkl","rb"))
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #f8f9fa !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .title {
            font-size: 40px !important;
            font-weight: bold !important;
            # color: #ff4b4b !important;
            text-align: center !important;
            padding: 10px 0 !important;
            margin-bottom: 20px !important;
        }
    </style>
    <h1 class="title">Spam Prediction Model (Text based)</h1>
""", unsafe_allow_html=True)

def isValidEmail(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@(gmail|yahoo|outlook|hotmail|icloud|protonmail|aol|zoho|mail|yandex|gmx|tutanota)\.(com|org|net|co|tz|uk|us|in|au)$'
    return re.match(pattern, email, re.IGNORECASE)

def showError(placeholder, msg):
    placeholder.markdown(f"""
        <div style="
            background-color: #fff0f0;
            border-left: 4px solid #ff4b4b;
            border-radius: 4px;
            padding: 8px 12px;
            margin-bottom: 8px;
            color: #cc0000;
            font-size: 14px;
        ">{msg}</div>
    """, unsafe_allow_html=True)

def clearAll():
    resultPlaceholder.empty()
    errorEmail.empty()
    errorSubject.empty()
    errorBody.empty()

# ── Result placeholder at very top ────────────────
resultPlaceholder = st.empty()

# ── Email From ────────────────────────────────────
errorEmail = st.empty()
emailFrom  = st.text_input("From (Email Address)")

# ── Subject ───────────────────────────────────────
errorSubject = st.empty()
subject      = st.text_input("Subject")

# ── Email Body ────────────────────────────────────
errorBody = st.empty()
body      = st.text_area("Email Body", height=200, placeholder="Paste or type the email content here...")

button = st.button("Detect Spam")

if button:
    clearAll()

    if emailFrom == "":
        showError(errorEmail, "Please enter sender email address")
    elif not isValidEmail(emailFrom):
        showError(errorEmail, "Invalid email — accepted hosts: gmail, yahoo, outlook, hotmail, icloud, protonmail, aol, zoho, mail, yandex, gmx, tutanota")
    elif subject == "":
        showError(errorSubject, "Please enter email subject")
    elif body == "":
        showError(errorBody, "Please enter email body")
    else:
        fullText = subject + " " + body
        textData = vector.transform([fullText])
        predict  = myModel.predict(textData)
        result   = labelEncoder.inverse_transform(predict)
        proba    = myModel.predict_proba(textData)

        st.session_state.showResult = True
        st.session_state.predict    = result[0]
        st.session_state.proba      = proba.tolist()

# ── Show result with dismiss button ───────────────
if st.session_state.get('showResult', False):
    if st.session_state.predict == "spam":
        with resultPlaceholder.container():
            col1, col2 = st.columns([9, 1])
            with col1:
                st.markdown(f"""
                    <div style="
                        background-color: #fff0f0;
                        border-left: 4px solid #ff4b4b;
                        border-radius: 4px;
                        padding: 12px 16px;
                        color: #cc0000;
                        font-size: 16px;
                        font-weight: bold;
                    ">This email is Spam! (Confidence: {st.session_state.proba[0][1]:.2%})</div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✕", key="dismissSpam"):
                    st.session_state.showResult = False
                    st.session_state.emailFrom  = ""
                    st.session_state.subject    = ""
                    st.session_state.body       = ""
                    st.rerun()
    else:
        with resultPlaceholder.container():
            col1, col2 = st.columns([9, 1])
            with col1:
                st.markdown(f"""
                    <div style="
                        background-color: #f0fff4;
                        border-left: 4px solid #28a745;
                        border-radius: 4px;
                        padding: 12px 16px;
                        color: #155724;
                        font-size: 16px;
                        font-weight: bold;
                    ">This email is Not Spam! (Confidence: {st.session_state.proba[0][0]:.2%})</div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✕", key="dismissHam"):
                    st.session_state.showResult = False
                    st.session_state.emailFrom  = ""
                    st.session_state.subject    = ""
                    st.session_state.body       = ""
                    st.rerun()