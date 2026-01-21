import streamlit as st
import tempfile
import os
import json
import re

from google import genai
from utils.pdf_utils import extract_text_from_pdf

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="AI PDF Question Answering", layout="centered")
st.title("AI PDF Question Answering")

client = genai.Client()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return match.group(0)


def chunk_text(text: str, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# -------------------------------------------------
# Session state
# -------------------------------------------------
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("last_files", None)
st.session_state.setdefault("qa_history", [])

# -------------------------------------------------
# File uploader
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------------------------
# Process PDFs immediately
# -------------------------------------------------
if uploaded_files:
    current_files = tuple(f.name for f in uploaded_files)

    if current_files != st.session_state.last_files:
        st.session_state.last_files = current_files
        st.session_state.chunks = []
        st.session_state.qa_history = []

        with st.spinner("Processing documents..."):
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    path = tmp.name

                text = extract_text_from_pdf(path)
                os.remove(path)

                if text:
                    st.session_state.chunks.extend(chunk_text(text))

        st.success(f"Loaded {len(st.session_state.chunks)} text chunks.")

# -------------------------------------------------
# Stable Question UI (NO flicker)
# -------------------------------------------------
qa_container = st.empty()

with qa_container.container():
    with st.form("qa_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])

        with col1:
            question = st.text_input(
                "Ask a question about the documents",
                placeholder="Type your question and press Enter",
                disabled=not bool(st.session_state.chunks)
            )

        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            send = st.form_submit_button("Send")

        submitted = send

# -------------------------------------------------
# Question answering
# -------------------------------------------------
if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        context = "\n\n".join(st.session_state.chunks)

        prompt = f"""
You are a document analysis assistant.

Rules:
- Use ONLY the provided document content
- Answer strictly in JSON
- Do NOT include markdown or explanations
- If the answer is not found, say so clearly

Return JSON in this format:
{{
  "answer": "<answer text>",
  "documents_used": []
}}

Documents:
{context}

Question:
{question}
"""

        try:
            with st.spinner("Analyzing documents..."):
                response = client.models.generate_content(
                    model="models/gemini-flash-latest",
                    contents=prompt
                )

            parsed = json.loads(extract_json(response.text))
            answer = parsed.get("answer", "").strip()

            st.session_state.qa_history.append(
                {"question": question, "answer": answer}
            )

        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                st.error("⚠️ Rate limit hit. Please wait a moment and try again.")
            else:
                st.error("An unexpected error occurred.")
                st.text(msg)

# -------------------------------------------------
# Render Q&A history
# -------------------------------------------------
if st.session_state.qa_history:
    st.subheader("Answer")
    for item in reversed(st.session_state.qa_history):
        st.markdown(f"**Q:** {item['question']}")
        st.write(item["answer"])
