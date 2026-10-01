import json

from pdf_extractor import extract_resume_text
from matcher import calculate_similarity


# Resume PDF
resume_path = "resume.pdf"


# Job Description
print("\nEnter the Job Description.")
print("Type END on a new line when finished.\n")

jd_lines = []

while True:
    line = input()

    if line.strip().upper() == "END":
        break

    jd_lines.append(line)

jd_text = "\n".join(jd_lines)


# Check JD
if not jd_text.strip():
    print("Job Description cannot be empty.")
    exit()


# Extract resume text
resume_text = extract_resume_text(resume_path)

print("\nResume text extracted successfully!")


# Calculate Resume-JD matching
result = calculate_similarity(
    resume_text,
    jd_text
)


# Display result as JSON
print("\nResume-JD Matching Result:")
print("--------------------------------")

print(json.dumps(result, indent=4))