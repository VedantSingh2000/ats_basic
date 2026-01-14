import google.generativeai as genai
import textwrap
from pypdf import PdfReader
from docx import Document
import re
import json

def clean_text(text):
    """Cleans extracted text by removing special characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    return text.strip()

def get_pdf_text(uploaded_file):
    """Extracts text from PDF."""
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def get_docx_text(uploaded_file):
    """Extracts text from DOCX."""
    try:
        doc = Document(uploaded_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_resume_data(text, api_key):
    """
    Uses Gemini to extract structured data (Category, Experience, etc).
    """
    try:
        genai.configure(api_key=api_key)
        # Use 1.5-flash for speed and JSON capabilities
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        You are a strict, senior HR recruiter with limited time.
        Analyze the resume objectively. No assumptions, no kindness.

        Resume Text:
        {text}

        Your task:
        - Judge the resume ONLY on what is clearly written.
        - Do not infer skills or experience that are not explicitly mentioned.
        - Penalize vague content, buzzwords, weak descriptions, and poor structure.

        Extract and return STRICTLY valid JSON with these fields:

        1. "category":
           - The single most suitable job role based ONLY on skills and experience.
           - If unclear, return "Unclear / Generic Profile".

        2. "experience_years":
           - Total years of professional experience as a numeric value.
           - If not clearly mentioned, return 0.

        3. "skills":
           - Top 5 clearly mentioned technical skills.
           - Do NOT guess or add related skills.

        4. "rating":
           - Score from 1 to 10.
           - 1–3: weak, irrelevant, or poorly written resume
           - 4–6: average, some substance but gaps
           - 7–8: strong and relevant
           - 9–10: exceptional and well-documented
           - Be conservative. High scores must be justified by content.

        5. "summary":
           - 2 short, factual sentences.
           - Mention strengths AND gaps if present.

        Return ONLY raw JSON.
        No explanations. No markdown. No extra text.
        """

        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Clean up if the model adds markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        return json.loads(response_text)
        
    except Exception as e:

        return {"error": str(e)}


