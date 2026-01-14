import streamlit as st
import google.generativeai as genai
from utils import clean_resume_text, extract_info_with_gemini, extract_text

# --- Page Configuration ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer & Matcher")
st.markdown("Pure AI Analysis - No Local Models")

# --- Inputs ---
col_api, col_upload = st.columns([2, 3])

with col_api:
    st.markdown("#### 🔑 API Key")
    api_key_input = st.text_input("Gemini API Key", type="password")

with col_upload:
    st.markdown("#### 📤 Upload Resume")
    uploaded_file = st.file_uploader("Upload .docx or .pdf", type=["docx", "pdf"])

st.markdown("---")
st.markdown("#### 📋 Job Description (Optional)")
jd_input = st.text_area("Paste JD here for matching:", height=150)

analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

# --- Processing ---
if analyze_button:
    if not uploaded_file or not api_key_input:
        st.warning("⚠️ Please provide an API Key and Upload a Resume.")
        st.stop()

    # Configure Gemini
    try:
        genai.configure(api_key=api_key_input)
        # Using the model you said works for you
        model = genai.GenerativeModel('gemini-3-flash-preview')
    except Exception as e:
        st.error(f"Configuration Error: {e}")
        st.stop()

    with st.spinner("🤖 AI is reading and analyzing..."):
        # 1. Extract
        raw_text = extract_text(uploaded_file)
        if not raw_text or raw_text.startswith("Error"):
            st.error(f"File Error: {raw_text}")
            st.stop()
            
        cleaned_text = clean_resume_text(raw_text)

        # 2. Analyze
        ai_data = extract_info_with_gemini(cleaned_text, model, jd_text=jd_input)

    # --- Display Results ---
    st.balloons()
    st.subheader("✨ Results")
    st.markdown("---")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🎯 Job Role (AI)", ai_data.get('category', 'Unknown'))
    with m2:
        st.metric("📅 Experience", f"{ai_data.get('experience_years', 0)} Years")
    with m3:
        st.metric("⭐ Rating", f"{ai_data.get('rating', 0)}/10")
    with m4:
        match_pct = ai_data.get('jd_match')
        if match_pct is not None:
            st.metric("✅ JD Match", f"{match_pct}%")
        else:
            st.metric("✅ JD Match", "N/A")

    # JD Details
    if jd_input and ai_data.get('jd_match') is not None:
        st.markdown("### 📊 Match Details")
        val = float(ai_data.get('jd_match', 0)) / 100
        st.progress(min(val, 1.0))
        
        missing = ai_data.get('missing_keywords', [])
        if missing:
            st.error(f"**Missing Skills:** {', '.join(missing)}")
        else:
            st.success("All requirements matched!")

    st.markdown("---")
    
    # Summary
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📝 Summary")
        st.info(ai_data.get('profile_summary', "N/A"))
    with c2:
        st.subheader("💡 Feedback")
        st.write(ai_data.get('feedback', "N/A"))

    with st.expander("View Raw Text"):
        st.text(raw_text)
