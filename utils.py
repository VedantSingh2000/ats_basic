import google.generativeai as genai
import re
import json
from docx import Document
from PyPDF2 import PdfReader  # Keep PyPDF2 since it works for you now

def clean_resume_text(text):
    # Simple cleaning to remove extra spaces and special characters
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text(uploaded_file):
    try:
        if uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"Error: {e}"
    return ""

def extract_info_with_gemini(resume_text, model, jd_text=None):
    """
    Analyzes resume. If jd_text is provided, it performs a match analysis.
    """
    
    # CASE 1: Resume + Job Description (Comparison)
    if jd_text and len(jd_text.strip()) > 10:
        prompt = f"""
        Act as an expert Application Tracking System (ATS).
        Compare the following Resume against the Job Description (JD).
        
        Resume:
        {resume_text[:20000]}
        
        Job Description:
        {jd_text[:10000]}
        
        Return a valid JSON object with EXACTLY these keys:
        1. "jd_match": A percentage (0-100) indicating how well the resume matches the JD (number only).
        2. "missing_keywords": A list of critical skills/keywords missing from the resume.
        3. "profile_summary": A 2-sentence summary of the candidate.
        4. "experience_years": Total years of experience (number).
        5. "rating": Overall candidate rating (1-10) based on the JD (number).
        6. "feedback": Brief feedback on why they match or don't match.

        Output strictly JSON.
        """
    
    # CASE 2: Resume Only (General Analysis)
    else:
        prompt = f"""
        Analyze this resume and extract the following details in JSON format:
        1. "profile_summary": A 2-sentence summary.
        2. "experience_years": Total years of experience (number).
        3. "rating": A score from 1-10 based on general strength (number).
        4. "feedback": Brief feedback on strengths/weaknesses.
        
        Resume:
        {resume_text[:20000]}
        
        Output strictly JSON.
        """

    try:
        response = model.generate_content(prompt)
        # Clean JSON markdown if present
        txt = response.text.replace("```json", "").replace("```", "")
        data = json.loads(txt)
        
        # Normalize keys if model changes them slightly
        if "summary" in data and "profile_summary" not in data:
            data["profile_summary"] = data["summary"]
            
        return data
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {
            "profile_summary": "Error parsing AI response.",
            "jd_match": 0,
            "missing_keywords": [],
            "experience_years": 0,
            "rating": 0,
            "feedback": "An error occurred during analysis."
        }
