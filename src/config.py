import os
from dotenv import load_dotenv

load_dotenv()

# LLM Settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "openai/gpt-oss-20b"

# Embedding Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
TOP_K = 4

# Paths
DATA_DIR = "data/documents"
FAISS_INDEX = "faiss_index"

# System prompt
SYSTEM_PROMPT = """You are a UAE financial regulatory assistant.
The context below contains text from UAE Central Bank regulatory documents.
Some text may be in Arabic — focus on the English parts to answer.
Answer questions using the English content in the context provided.
If you cannot find a clear answer in the English text say: I do not have enough information to answer this.
Always cite the source and page number."""