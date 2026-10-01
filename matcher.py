import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load free local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Skills to check
SKILLS = [
    "Python",
    "FastAPI",
    "SQL",
    "REST API",
    "Git",
    "Docker",
    "AWS",
    "Machine Learning",
    "API development",
    "JavaScript",
    "React",
    "Angular",
    "C++",
    "Java",
    "PHP",
    "HTML",
    "CSS",
    "Django",
    "Flask",
    "Node.js",
    "MongoDB",
    "MySQL",
    "PostgreSQL",
    "GitHub"
]


def find_skills(text):
    """Find skills mentioned in the text."""

    found_skills = []

    for skill in SKILLS:
        if re.search(
            r"\b" + re.escape(skill) + r"\b",
            text,
            re.IGNORECASE
        ):
            found_skills.append(skill)

    return found_skills


def find_experience(text):
    """Find years of experience mentioned in resume."""

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*years?"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            years = match.group(1)
            return f"Resume mentions {years} years of experience"

    return "Experience information not clearly found"


def calculate_similarity(resume_text, jd_text):

    # Create embeddings locally
    resume_vector = model.encode(resume_text)
    jd_vector = model.encode(jd_text)

    # Calculate semantic similarity
    similarity = cosine_similarity(
        [resume_vector],
        [jd_vector]
    )[0][0]

    semantic_score = float(similarity * 100)

    # Find skills
    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(jd_text)

    # Find matched skills
    matched_skills = [
        skill
        for skill in jd_skills
        if skill in resume_skills
    ]

    # Find missing skills
    missing_skills = [
        skill
        for skill in jd_skills
        if skill not in resume_skills
    ]

    # Calculate skill score
    if len(jd_skills) > 0:
        skill_score = (
            len(matched_skills) / len(jd_skills)
        ) * 100
    else:
        skill_score = 0.0

    # Combined match percentage
    match_percentage = float(
        round(
            (semantic_score * 0.5) +
            (skill_score * 0.5),
            2
        )
    )

    # Experience
    experience_match = find_experience(resume_text)

    # Suggestions
    suggestions = []

    for skill in missing_skills:
        suggestions.append(
            f"Consider adding {skill} experience or projects"
        )

    if not suggestions:
        suggestions.append(
            "Resume skills match the listed JD skills well"
        )

    # Final JSON-compatible result
    result = {
        "match_percentage": match_percentage,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "experience_match": experience_match,
        "suggestions": suggestions
    }

    return result