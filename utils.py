import google.generativeai as genai
import re
import json
from docx import Document
from pypdf import PdfReader

def clean_resume_text(text):
    # Basic cleaning
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
        return f"Error reading file: {e}"
    return ""

def extract_info_with_gemini(resume_text, model, jd_text=None):
    """
    Pure Gemini Analysis.
    Determines Category, Experience, and Match % (if JD provided).
    """
    
    # Prompt for Resume + JD
    if jd_text and len(jd_text.strip()) > 10:
        prompt = f"""
        Act as an expert ATS (Application Tracking System).
        Compare the Resume against the Job Description (JD).
        
        Resume:
        {resume_text[:25000]}
        
        Job Description:
        {jd_text[:10000]}
        
        Extract the following strictly as JSON:
        1. "category": The most fitting job title for this resume (e.g. "Full Stack Developer").
        2. "jd_match": Match percentage (0-100) against the JD (number).
        3. "missing_keywords": List of top 5 missing technical skills from the JD.
        4. "experience_years": Total years of experience (number).
        5. "rating": Candidate rating 1-10 based on the JD (number).
        6. "profile_summary": A 2-sentence summary of the candidate.
        7. "feedback": Short feedback on why they match or don't match.
        """
    
    # Prompt for Resume Only
    else:
        prompt = f"""
        Analyze this resume as a Recruiter.
        
        Resume:
        {resume_text[:25000]}
        
        Extract the following strictly as JSON:
        1. "category": The candidate's primary job title/role (e.g. "Data Scientist").
        2. "experience_years": Total years of experience (number).
        3. "rating": General rating 1-10 (number).
        4. "profile_summary": A 2-sentence summary.
        5. "feedback": Brief feedback on strengths/weaknesses.
        
        (Set "jd_match" to null and "missing_keywords" to empty list).
        """

    try:
        response = model.generate_content(prompt)
        # Clean JSON markdown
        txt = response.text.replace("```json", "").replace("```", "")
        data = json.loads(txt)
        return data
    except Exception as e:
        return {
            "category": "Error",
            "profile_summary": f"AI Error: {e}",
            "jd_match": 0,
            "rating": 0
        }
