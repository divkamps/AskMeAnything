from google.adk.agents import LlmAgent

doc_agent = LlmAgent(
    name="document_qa_agent",
    model="models/gemini-flash-latest"
)
