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
        Act as a senior HR Recruiter. Analyze the resume text below.
        
        Resume Text:
        {text}
        
        Extract the following details and return strictly valid JSON:
        1. "category": The best fitting job role (e.g., Data Scientist, React Developer, HR).
        2. "experience_years": Total years of experience (numeric value, e.g., 2.5). Use 0 if not found.
        3. "skills": A list of top 5 technical skills found.
        4. "rating": A score from 1-10 based on content quality.
        5. "summary": A 2-sentence summary of the candidate.
        
        Return ONLY the JSON. No markdown formatting.
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

