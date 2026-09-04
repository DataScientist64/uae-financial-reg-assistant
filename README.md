# UAE Financial Regulatory Assistant

An AI-powered RAG assistant grounded in real UAE Central Bank regulatory documents. Built with production engineering practices including modular code, evaluation framework and automated tests.

## Live Demo

https://uae-financial-reg-assistant-62ceqpuzveujeuajdvhpvn.streamlit.app/

## What it does

Ask questions about UAE banking regulations, consumer protection rules and credit risk guidelines. The assistant retrieves relevant information from official regulatory documents and generates grounded answers with zero hallucination outside the knowledge base.

## Documents indexed

- UAE Central Bank Consumer Protection Standards (Circular No. 8 — 2020)
- UAE Central Bank Credit Risk Management Regulation (C 3/2024)
- Basel III: A Global Regulatory Framework for More Resilient Banks

## Technical architecture

- LLM: Groq API (openai/gpt-oss-20b)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, no API cost)
- Vector store: FAISS
- Framework: LangChain and Streamlit
- Testing: pytest (9 tests — all passing)

## Project structure
uae-financial-reg-assistant/
├── src/
│ ├── config.py — all settings and constants
│ ├── ingest.py — PDF loading, chunking, FAISS creation
│ ├── retrieve.py — similarity search and context formatting
│ └── generate.py — Groq API call and response generation
├── tests/
│ ├── test_retrieval.py
│ └── test_generation.py
├── eval_set/
│ └── qa_pairs.json — structured evaluation set
├── data/documents/ — regulatory PDFs
├── app.py — thin Streamlit interface
└── requirements.txt


## Engineering decisions

- Chose RAG over fine-tuning — knowledge base is document-based and updatable without retraining
- Used local sentence-transformers for embeddings — no API dependency or cost
- Chunk size 1000 characters with 200 character overlap — preserves context across regulatory clause boundaries
- Grounded refusal — model says it does not have information rather than hallucinate
- Source attribution on every answer — exact document name and page number
- Top-K retrieval of 4 chunks — balances context richness with token efficiency
- Modular code structure — separate modules for ingestion, retrieval and generation
- 9 automated tests covering retrieval accuracy, generation quality and edge cases
- Evaluation set with 6 QA pairs including out-of-scope questions to test grounded refusal

## How to run locally

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your documents to `data/documents/`
4. Set your Groq API key: `export GROQ_API_KEY=your_key_here`
5. Run: `streamlit run app.py`

## How to run tests
pytest tests/ -v

## Author

Aditya Tribhuvan
MSc Artificial Intelligence and Computer Science, University of Birmingham Dubai
linkedin.com/in/adityatribhuvan
github.com/DataScientist64