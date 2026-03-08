# Enable future Python behavior for type hints
from __future__ import annotations

# Import libraries
import os
import sys
from pathlib import Path

# Import Streamlit
import streamlit as st
from dotenv import load_dotenv

# Determine the root dir = the project
ROOT = Path(__file__).resolve().parents[1]

# Add root to Python path so external modules can be imported
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import the RAG engine
from src.rag import SimpleRAGEngine

# Load the environment variables located in the Root
load_dotenv(ROOT / ".env")

# Configure streamlit page (Full width)
st.set_page_config(page_title="Medication Information RAG Assistant", layout="wide")

# Initialize the RAG engine
engine = SimpleRAGEngine()


# Display citations if available
def render_citations(citations):
    if not citations:
        st.write("No citations available.")
        return
    # Loop through citations and display each citation (title, section, and chunk id)
    for c in citations:
        st.markdown(f"- **{c.source_title}** | {c.section or 'Section N/A'} | `{c.chunk_id}`")


# Page title and disclaimer
st.title("Anticoagulant Information RAG Assistant")
st.caption(
    "Educational prototype for source-grounded medication Q&A. Not clinical decision support and not for patient-specific treatment decisions."
)

# Left sidebar
with st.sidebar:
    st.header("Scope") # Header
    st.write("Starter domain: anticoagulation-related medications") # Domain
    
    # Sources as a bullet list
    st.write("Included demo sources:")
    st.markdown("- Warfarin FDA Label\n- Apixaban FDA Label\n- Rivaroxaban FDA Label\n- Dabigatran FDA Label\n- Enoxaparin FDA Label")

    # Example questions
    st.header("Example questions")
    examples = [
        "What are the major warfarin interaction risks?",
        "Compare apixaban vs rivaroxaban monitoring considerations.",
        "Summarize dabigatran counseling points.",
        "What boxed warnings apply to enoxaparin?",
    ]

    # Display the question as a code block
    for ex in examples:
        st.code(ex, language=None)

    # List of planned upgrades
    st.header("Planned upgrades")
    st.markdown(
        "- Real ingestion pipeline\n"
        "- Embeddings + FAISS retrieval\n"
        "- Citation-aware prompts\n"
        "- Evaluation dashboard\n"
        "- Feedback logging"
    )

# A Text input box for typing questions
question = st.text_input(
    # Placed above the box
    "Ask a medication question",
    # The placeholder will be in the box until the typing starts
    placeholder="e.g., Compare apixaban vs rivaroxaban monitoring considerations.",
)

# Create a button in the UI, highlighted as primary
run = st.button("Generate answer", type="primary")

# When the button is hit and the question is typed
 # Remove the whitespace first
if run and question.strip():
    # Display a loading hourglass and a message
    with st.spinner("Generating source-grounded response..."):
        # Call the RAG engine
        answer = engine.answer(question.strip())

    # Short answer section
    st.subheader("Short answer")
    
    # Show the short answer text after the RAG
    st.write(answer.short_answer.text)
    # Citation (can be expanded/collapsed)
    with st.expander("Citations for short answer", expanded=True):
        render_citations(answer.short_answer.citations)

    # Create two columns
    col1, col2 = st.columns(2)

    # The left column: Major warnings and interactions
    with col1:
        # Major warning
        st.subheader("Major warnings")
        # Display the contents as bullet points, then citations
        for item in answer.major_warnings:
            st.write(f"- {item.text}")
            render_citations(item.citations)

        # Interactions
        st.subheader("Interactions")
        # Display the contents as bullet points, then citations
        for item in answer.interactions:
            st.write(f"- {item.text}")
            render_citations(item.citations)

    # The right column
    with col2: Monitoring and Counseling points
        # Monitoring
        st.subheader("Monitoring")
        # Display the contents as bullet points, then citations
        for item in answer.monitoring:
            st.write(f"- {item.text}")
            render_citations(item.citations)
        
        # Counseling points
        st.subheader("Counseling points")
        # Display the contents as bullet points, then citations        
        for item in answer.counseling_points:
            st.write(f"- {item.text}")
            render_citations(item.citations)

    # Limitations
    st.subheader("Limitations")
    # Limitations in the info box
    st.info(answer.limitations or "No limitations reported.")

# If the butten was clicked without a question
elif run:
    st.warning("Enter a question to continue.")

# Collapsible Implementation notes
with st.expander("Implementation notes"):
    st.write(
        "This starter app uses a tiny in-memory corpus and simple keyword retrieval so you can focus on repo structure, "
        "UI flow, structured outputs, and next-step upgrades."
    )

    # Environment description
    st.write(f"Configured model env var: {os.getenv('OPENAI_MODEL', 'not set')}")
