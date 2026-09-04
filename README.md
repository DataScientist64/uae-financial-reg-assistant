# UAE Financial Regulatory Assistant

An AI-powered RAG (Retrieval-Augmented Generation) assistant grounded in real UAE Central Bank regulatory documents.

## What it does

Ask questions about UAE banking regulations, consumer protection rules and credit risk guidelines. The assistant retrieves relevant information from official regulatory documents and generates grounded answers — with zero hallucination outside the knowledge base.

## Documents indexed

- UAE Central Bank Consumer Protection Standards (Circular No. 8 – 2020)
- UAE Central Bank Credit Risk Management Regulation (C 3/2024)
- Basel III: A Global Regulatory Framework for More Resilient Banks

## Technical architecture

- LLM: Groq API (openai/gpt-oss-20b)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free)
- Vector store: FAISS
- Chunking: 1000 characters with 200 character overlap
- Framework: LangChain and Streamlit

## Engineering decisions

- Chose RAG over fine-tuning — knowledge base is document-based and needs to stay updatable without retraining
- Used local sentence-transformers for embeddings — no API dependency or cost
- Implemented grounded refusal — model instructed to say it does not have information rather than hallucinate
- Source attribution on every answer — shows exact document and page number
- Similarity search returns top 4 chunks for context

## How to run locally

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your documents to `data/documents/`
4. Set your Groq API key: `export GROQ_API_KEY=your_key_here`
5. Run: `streamlit run app.py`

## Author

Aditya Tribhuvan
MSc Artificial Intelligence and Computer Science, University of Birmingham Dubai
github.com/DataScientist64