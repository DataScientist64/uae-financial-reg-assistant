import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data/documents"
FAISS_INDEX = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a UAE financial regulatory assistant.
The context below contains text from UAE Central Bank regulatory documents.
Some text may be in Arabic — focus on the English parts to answer.
Answer questions using the English content in the context provided.
If you cannot find a clear answer in the English text say: I do not have enough information to answer this.
Always cite the source and page number."""

st.set_page_config(page_title="UAE Financial Regulatory Assistant", page_icon="🏦")
st.title("🏦 UAE Financial Regulatory Assistant")
st.caption("Grounded in real UAE Central Bank regulatory documents.")
st.warning("This tool is for informational purposes only. Not legal or financial advice.")


@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(FAISS_INDEX):
        return FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join(DATA_DIR, f)).load())
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_INDEX)
    return vs


vector_store = load_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about UAE financial regulations...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            docs = vector_store.similarity_search(question, k=TOP_K)
            context = "\n\n".join([
                f"Source: {d.metadata.get('source','Unknown')} Page: {d.metadata.get('page','Unknown')}\n{d.page_content}"
                for d in docs
            ])
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Error: {e}"
            st.markdown(answer)
            if docs:
                with st.expander("Sources"):
                    for d in docs:
                        st.write(f"Page {d.metadata.get('page','?')} — {d.metadata.get('source','?')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})