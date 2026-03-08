# Anticoagulant Information RAG Assistant

A retrieval-augmented generation (RAG) app that answers medication questions from trusted documents with citations, structured summaries, and basic safety guardrails.

## Why this project
Generic LLMs can hallucinate on medication questions. This project narrows the scope to a pharmacy-friendly use case and grounds answers in indexed source documents.

Initial domain focus:
- warfarin
- apixaban
- rivaroxaban
- dabigatran
- enoxaparin

## Features
- Citation-grounded answers
- Structured medication summaries
- Comparison mode
- Retrieved source viewer
- Low-confidence fallback messaging
- Query logging for evaluation

## Starter architecture
Documents -> chunking + metadata -> embeddings -> FAISS -> retrieval -> LLM -> structured answer + citations

## Project layout
```
med-rag-assistant/
├── app/
│   └── streamlit_app.py
├── src/
│   └── rag.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
├── requirements.txt
├── .env.example
└── README.md
```

## Quick start
### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Configure environment
Copy `.env.example` to `.env` and fill in your API key if you are using an OpenAI-compatible endpoint.

```bash
cp .env.example .env
```

### 3) Run the app
```bash
streamlit run app/streamlit_app.py
```

## Example questions
- What are the major warfarin interaction risks?
- Compare apixaban vs rivaroxaban monitoring considerations.
- Summarize dabigatran counseling points.
- What boxed warnings apply to enoxaparin?

## Current limitations
- Educational prototype only
- Not clinical decision support
- Minimal starter retrieval corpus
- No patient-specific recommendations

## Next steps
1. Add ingestion for FDA labels / guideline PDFs
2. Replace demo corpus with your source documents
3. Add real embeddings + vector search
4. Build evaluation set of 50 questions
5. Add logging dashboard and feedback loop

## Resume-friendly description
Built a medication information RAG assistant using Python, Streamlit, and structured LLM outputs to answer pharmacy questions from trusted documents with source citations and basic guardrails.
