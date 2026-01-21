import streamlit as st
import tempfile
import os
import json
import re

from google import genai
from utils.pdf_utils import extract_text_from_pdf

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI PDF Question Answering",
    layout="centered"
)

st.title("AI PDF Question Answering")

# -------------------------------------------------
# Gemini client (uses GOOGLE_API_KEY from env)
# -------------------------------------------------
client = genai.Client()

# -------------------------------------------------
# Helper: robust JSON extraction
# -------------------------------------------------
def extract_json(text: str) -> str:
    """
    Robustly extract JSON from Gemini output.
    Handles markdown fences, 'json' tags, and extra text.
    """
    # Remove markdown code fences like ```json or ```
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    return match.group(0)

# -------------------------------------------------
# UI Inputs
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

question = st.text_input("Ask a question about the documents")

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if uploaded_files and question:
    combined_text = ""
    doc_names = []

    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        text = extract_text_from_pdf(tmp_path)

        combined_text += f"\n--- Document: {f.name} ---\n{text}\n"
        doc_names.append(f.name)

        os.remove(tmp_path)

    prompt = f"""
You are a document analysis assistant.

Rules:
- Use ONLY the provided documents
- Answer strictly in JSON
- Do NOT include markdown, code fences, or explanations
- If the answer is not found, say so clearly

Return JSON in this format:
{{
  "answer": "<answer text>",
  "documents_used": {doc_names}
}}

Documents:
{combined_text}

Question:
{question}
"""

    with st.spinner("Analyzing documents..."):
        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )

    raw_text = response.text.strip()

    # -------------------------------------------------
    # Parse response and show ONLY the answer
    # -------------------------------------------------
    try:
        json_text = extract_json(raw_text)
        parsed = json.loads(json_text)

        answer = parsed.get("answer", "").strip()

        st.subheader("Answer")

        if not answer or "not found" in answer.lower():
            st.warning("The document does not contain information about this topic.")
        else:
            st.write(answer)

    except Exception:
        st.error("Unable to parse model response.")
        st.text(raw_text)
