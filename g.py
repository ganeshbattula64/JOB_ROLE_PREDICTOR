import streamlit as st
import fitz  # PyMuPDF
import docx
import google.generativeai as genai

# Setup Gemini API
genai.configure(api_key="AIzaSyDyvVg_hjVdAe6bQvkvrWFtU3sezuAir_o")  # 🔑 Replace with your key

# Function to get ATS score and suggestions
def get_ats_score_and_feedback(resume_text, job_desc):
    prompt = f"""
You are an ATS resume evaluator.

Given the following resume and job description, return:
1. ATS Score out of 100 based on keyword match, skill relevance, and formatting.
2. If score < 80, suggest improvements.

Resume:
{resume_text}

Job Description:
{job_desc}

Respond in this format:
ATS Score: XX
Improvement Suggestions:
- ...
- ...
"""
    model = genai.GenerativeModel("Gemini 1.5 Pro")
    response = model.generate_content(prompt)
    return response.text

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Streamlit UI
st.set_page_config(page_title="Resume ATS Scorer", layout="centered")
st.title("📄 Resume ATS Checker with Gemini AI")
st.write("Upload your resume and enter a job description to get an ATS score and suggestions.")

# Resume upload
uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

# Job description input
job_desc = st.text_area("Paste the Job Description here", height=250)

# Trigger evaluation
if uploaded_file and job_desc.strip():
    # Extract resume text
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type!")
        st.stop()

    with st.spinner("Analyzing your resume with Gemini..."):
        result = get_ats_score_and_feedback(resume_text, job_desc)

    st.subheader("📊 ATS Analysis Result")
    st.markdown(result)

    # Try to extract and interpret ATS Score
    try:
        ats_score_line = [line for line in result.splitlines() if "ATS Score" in line][0]
        ats_score = int(''.join(filter(str.isdigit, ats_score_line)))
        if ats_score < 80:
            st.warning("⚠️ Your ATS score is below 80. Check the improvement suggestions above.")
        else:
            st.success("✅ Great! Your resume aligns well with the job description.")
    except Exception:
        st.info("Could not parse ATS score automatically. Please review Gemini's response above.")

