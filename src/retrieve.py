import logging
from langchain_community.vectorstores import FAISS
from src.config import TOP_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrieve(query: str, vector_store: FAISS) -> list:
    try:
        results = vector_store.similarity_search(query, k=TOP_K)
        logger.info(f"Retrieved {len(results)} chunks for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []


def format_context(documents: list) -> str:
    if not documents:
        return ""
    return "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')} "
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    ])