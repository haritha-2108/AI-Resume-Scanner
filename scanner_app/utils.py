import spacy
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
    else:
        text = file.read().decode("utf-8")
    return text

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def get_resume_score(resume_text, job_description):
    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(job_description)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, jd_clean])
    score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100
    return round(score, 2)
