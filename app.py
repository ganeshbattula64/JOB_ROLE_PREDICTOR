from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF for PDF extraction
import pdfplumber
import docx
import pandas as pd
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__, static_folder='static')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load skills from an external file
def load_skills():
    skills_file = "skills.txt"
    if not os.path.exists(skills_file):
        return []
    with open(skills_file, "r") as f:
        skills = [line.strip() for line in f.readlines()]
    return skills

# Function to extract skills from text
def extract_skills(text):
    skills_list = load_skills()
    words = re.findall(r'\b\w+\b', text.lower())
    return list(set(skill for skill in skills_list if skill.lower() in words))

# Function to generate explanation for a predicted job role
def generate_explanation(job_role):
    explanations = {
        "Software Engineer": "Strong programming skills in Python, Java, or C++. Experience with frameworks such as Django, Flask, or Spring Boot.",
        "Data Scientist": "Expertise in machine learning, data analysis, and tools like TensorFlow, PyTorch, and Scikit-learn.",
        "DevOps Engineer": "Experience with DevOps tools like Docker, Kubernetes, AWS, and CI/CD pipelines.",
        "Product Manager": "Project management skills with Agile methodologies, Jira, and strong stakeholder communication.",
        "Full Stack Developer": "Expertise in both front-end and back-end technologies including JavaScript, React, Node.js, and Django.",
        "AI/ML Engineer": "Experience in AI, deep learning, NLP, TensorFlow, PyTorch, and model deployment.",
        "Cybersecurity Analyst": "Knowledge of penetration testing, security audits, risk management, and tools like Wireshark.",
        "Cloud Engineer": "Experience with AWS, Azure, Google Cloud, Kubernetes, and Terraform.",
        "Business Analyst": "Data-driven decision-making, SQL, Power BI, Tableau, and market research.",
        "UI/UX Designer": "Proficiency in Figma, Adobe XD, wireframing, and front-end design principles."
    }
    return explanations.get(job_role, "Role assigned based on matching relevant skills and experience.")

# Load Dataset
def load_dataset():
    if not os.path.exists("dataset.csv"):
        return None
    df = pd.read_csv("dataset.csv")
    df["resume_text"] = df["resume_text"].astype(str).apply(preprocess_text)
    return df

df = load_dataset()
if df is None:
    raise Exception("Dataset file not found!")

# Load Trained Model
def load_model():
    if not os.path.exists("job_role_model.pkl"):
        return None
    return joblib.load("job_role_model.pkl")

model = load_model()
if model is None:
    raise Exception("Model file not found!")

# Function to extract text from PDF resumes
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        return str(e)
    return text.strip() if text.strip() else "No text found in PDF."

# Function to extract text from DOCX resumes
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    uploaded_file = request.files['file']
    file_type = uploaded_file.filename.split(".")[-1]
    file_path = f"uploaded_resume.{file_type}"
    
    uploaded_file.save(file_path)
    
    extracted_text = extract_text_from_pdf(file_path) if file_type == "pdf" else extract_text_from_docx(file_path)
    if not extracted_text.strip():
        return jsonify({"error": "No text extracted from resume"})
    
    skills = extract_skills(extracted_text)
    if not skills:
        return jsonify({"message": "No skills detected", "job_role": "None", "explanation": "No job role predicted due to lack of extracted skills."})
    
    processed_text = preprocess_text(extracted_text)
    predicted_role = model.predict([processed_text])[0]
    role_explanation = generate_explanation(predicted_role)
    
    return jsonify({"skills": skills, "predicted_job_role": predicted_role, "explanation": role_explanation})

if __name__ == '__main__':
    app.run(debug=True)
