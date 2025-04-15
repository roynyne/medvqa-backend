# app/llm_guard.py

import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer on some common medical VQA phrases
vectorizer = TfidfVectorizer()
vectorizer.fit([
    "what is shown in this image?",
    "is this a chest xray?",
    "is there a tumor present?",
    "what organ is affected?",
    "is this an MRI or CT scan?",
    "is the lung abnormal?"
])

# Keywords to reject
BAD_PATTERNS = [
    r"\b(joke|banana|love|weather|your name|how are you|meaning of life)\b",
    r"^\s*\w{1,3}\s*$"  # e.g., "hi", "yes", "no", "ok"
]

def is_valid_question(question: str) -> dict:
    q_lower = question.lower().strip()

    for pattern in BAD_PATTERNS:
        if re.search(pattern, q_lower):
            return {
                "valid": False,
                "rewritten_question": None,
                "message": "This question is invalid or not relevant to medical imaging."
            }

    tfidf_score = vectorizer.transform([q_lower]).sum()
    if tfidf_score < 0.1:
        return {
            "valid": False,
            "rewritten_question": None,
            "message": "This question seems too vague or irrelevant."
        }

    return {
        "valid": True,
        "rewritten_question": question,  # No rewriting
        "message": "Question accepted."
    }



