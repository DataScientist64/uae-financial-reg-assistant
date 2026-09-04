import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DATA_DIR, FAISS_INDEX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents(data_dir: str = DATA_DIR) -> list:
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
    return documents


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list) -> FAISS:
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX)
    logger.info(f"Vector store saved to {FAISS_INDEX}")
    return vector_store


def load_vector_store() -> FAISS:
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        FAISS_INDEX,
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info("Vector store loaded successfully")
    return vector_store


def get_or_create_vector_store() -> FAISS:
    if os.path.exists(FAISS_INDEX):
        return load_vector_store()
    docs = load_documents()
    chunks = chunk_documents(docs)
    return create_vector_store(chunks)