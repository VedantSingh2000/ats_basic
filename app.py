import streamlit as st
import joblib
import pandas as pd
import os
from utils import get_pdf_text, get_docx_text, clean_text, extract_resume_data

# --- Page Config ---
st.set_page_config(page_title="Smart Resume Analyzer", page_icon="🚀", layout="wide")

# --- Load Local Models (Optional Fallback) ---
# If your SVM files exist, it uses them. If not, it skips without crashing.
MODEL_DIR = "saved_models"
svm_model = None
tfidf_vectorizer = None
label_encoder = None

if os.path.exists(MODEL_DIR):
    try:
        svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model_pipeline.joblib"))
        tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    except Exception:
        pass # Silently fail and rely on Gemini instead

# --- UI Layout ---
st.title("🚀 Smart Resume Analyzer")
st.markdown("Upload a resume to classify the role and get AI feedback.")

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if key is in secrets, otherwise ask user
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key loaded from Secrets ✅")
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        st.caption("Get key here: [Google AI Studio](https://aistudio.google.com/)")

# File Uploader
uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    # 1. Extract Text
    with st.spinner("Reading file..."):
        if uploaded_file.name.endswith(".pdf"):
            raw_text = get_pdf_text(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            raw_text = get_docx_text(uploaded_file)
        else:
            raw_text = ""
            st.error("Unsupported file format.")

    # 2. Process if text found
    if raw_text and len(raw_text) > 50:
        cleaned_text = clean_text(raw_text)
        
        col1, col2 = st.columns([1, 1])
        
        # Analyze Button
        if st.button("Analyze Resume", type="primary"):
            if not api_key:
                st.error("Please enter an API Key in the sidebar.")
            else:
                with st.spinner("Analyzing with AI..."):
                    # Call Gemini
                    ai_data = extract_resume_data(cleaned_text, api_key)
                    
                    if "error" in ai_data:
                        st.error(f"AI Analysis Failed: {ai_data['error']}")
                    else:
                        # --- Display Results ---
                        st.success("Analysis Complete!")
                        
                        # Metrics Row
                        m1, m2, m3 = st.columns(3)
                        
                        # Determine Category: Use SVM if available, else use Gemini
                        predicted_category = ai_data.get("category", "Unknown")
                        source = "AI (Gemini)"
                        
                        if svm_model and tfidf_vectorizer and label_encoder:
                            try:
                                # Transform text and predict
                                vector = tfidf_vectorizer.transform([cleaned_text])
                                prediction = svm_model.predict(vector)
                                predicted_category = label_encoder.inverse_transform(prediction)[0]
                                source = "ML Model (SVM)"
                            except Exception as e:
                                print(f"SVM Error: {e}")

                        m1.metric("Predicted Role", predicted_category, delta=source)
                        m2.metric("Experience", f"{ai_data.get('experience_years', 0)} Years")
                        m3.metric("AI Rating", f"{ai_data.get('rating', 'N/A')}/10")
                        
                        st.divider()
                        
                        # Summary & Skills
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.subheader("📝 Professional Summary")
                            st.info(ai_data.get("summary", "No summary available."))
                        
                        with c2:
                            st.subheader("🛠️ Top Skills")
                            skills = ai_data.get("skills", [])
                            if isinstance(skills, list):
                                st.write(", ".join(skills))
                            else:
                                st.write(str(skills))

                        # Raw Text Expander
                        with st.expander("View Raw Resume Text"):
                            st.text(raw_text)

    elif raw_text:
        st.warning("Resume text is too short to analyze.")