import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
import nltk

# Import helper functions from utils.py
from utils import clean_resume_text, extract_info_with_gemini, extract_text

# --- Page Configuration ---
st.set_page_config(
    page_title="Resume Analyzer & JD Matcher",
    page_icon="📄",
    layout="wide"
)

# --- Download NLTK Stopwords ---
@st.cache_resource
def download_nltk_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)

download_nltk_stopwords()

# --- Load Models (SVM) ---
MODEL_DIR = "saved_models"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "svm_model_pipeline.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load(TFIDF_PATH)
        svm = joblib.load(SVM_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        return tfidf, svm, le, True
    except Exception as e:
        st.error(f"Error loading local models: {e}")
        return None, None, None, False

tfidf_vectorizer, svm_model, label_encoder, models_loaded = load_models()

# --- Session State ---
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

# --- UI Layout ---
st.title("📄 Resume Analyzer & JD Matcher ✨")

col_api, col_upload = st.columns([2, 3])

with col_api:
    st.markdown("#### 🔑 API Configuration")
    api_key_input = st.text_input("Google Gemini API Key", type="password")

with col_upload:
    st.markdown("#### 📤 Upload Resume")
    uploaded_file = st.file_uploader("Upload .docx or .pdf", type=["docx", "pdf"])

st.markdown("---")
# New JD Input Section
st.markdown("#### 📋 Job Description (Optional)")
jd_input = st.text_area("Paste the Job Description here to see how well the resume matches:", height=150)

analyze_button = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

# --- Processing Logic ---
if analyze_button:
    if not uploaded_file:
        st.warning("⚠️ Please upload a resume first.")
        st.stop()
    if not api_key_input:
        st.warning("⚠️ Please enter your Gemini API Key.")
        st.stop()

    # Configure Gemini
    try:
        genai.configure(api_key=api_key_input)
        # Using the model that works for you
        model = genai.GenerativeModel('gemini-3-flash-preview')
    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()

    with st.spinner("🤖 Analyzing Resume..."):
        # 1. Extract Text
        raw_text = extract_text(uploaded_file)
        if not raw_text or raw_text.startswith("Error"):
            st.error(f"Could not read file: {raw_text}")
            st.stop()
            
        cleaned_text = clean_resume_text(raw_text)

        # 2. Get AI Analysis (With JD if provided)
        ai_data = extract_info_with_gemini(cleaned_text, model, jd_text=jd_input)

        # 3. Get SVM Prediction (Local)
        predicted_category = "N/A"
        if models_loaded and cleaned_text:
            try:
                vector = tfidf_vectorizer.transform([cleaned_text])
                pred = svm_model.predict(vector)
                predicted_category = pred[0]
            except:
                predicted_category = "Error"

    # --- Display Results ---
    st.balloons()
    st.subheader("✨ Analysis Results")
    st.markdown("---")

    # Row 1: Metrics
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("🎯 Category (SVM)", predicted_category)
    with m2:
        st.metric("📅 Experience", f"{ai_data.get('experience_years', 0)} Years")
    with m3:
        st.metric("⭐ General Rating", f"{ai_data.get('rating', 0)}/10")
    with m4:
        # JD Match Metric (Only if JD was provided)
        match_pct = ai_data.get('jd_match', None)
        if match_pct is not None:
            st.metric("✅ JD Match", f"{match_pct}%")
        else:
            st.metric("✅ JD Match", "N/A")

    # Row 2: JD Specifics (Progress Bar & Missing Skills)
    if jd_input and ai_data.get('jd_match') is not None:
        st.markdown("### 📊 JD Match Analysis")
        
        # Progress Bar
        match_val = float(ai_data.get('jd_match', 0)) / 100
        st.progress(min(match_val, 1.0), text=f"ATS Match Score: {ai_data.get('jd_match')}%")
        
        # Missing Keywords
        missing = ai_data.get('missing_keywords', [])
        if missing:
            st.error(f"**⚠️ Missing Skills:** {', '.join(missing)}")
        else:
            st.success("**✅ All key skills matched!**")

    st.markdown("---")
    
    # Row 3: Summary & Feedback
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📝 Profile Summary")
        st.info(ai_data.get('profile_summary', "No summary available."))
    
    with c2:
        st.subheader("💡 AI Feedback")
        st.write(ai_data.get('feedback', "No feedback available."))

    # Raw Data Expander
    with st.expander("📄 View Extracted Text"):
        st.text(raw_text)
