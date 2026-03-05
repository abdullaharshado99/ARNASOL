import os

class RAGConfig:
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Embedding settings
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
    # Alternative: "BAAI/bge-small-en-v1.5" or "intfloat/e5-large-v2" or "all-MiniLM-L6-v2"

    # Vector DB settings
    VECTOR_DB_PATH = "D:/Abdullah/Documents/codes/Projects/ArtiNex/static/vector_db/chroma"
    COLLECTION_NAME = "arnasol_documents"

    # LLM settings (using HuggingFace or OpenAI)
    USE_OPENAI = False  # Set to True if using OpenAI
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    OPENAI_MODEL = "gpt-3.5-turbo"

    # Local LLM (HuggingFace)
    LOCAL_LLM_MODEL = "microsoft/phi-2"  # Small but powerful
    # Alternative: "mistralai/Mistral-7B-Instruct-v0.1"

    # Search settings
    TOP_K_RESULTS = 1
    SIMILARITY_THRESHOLD = 0.7

    # File upload settings
    UPLOAD_FOLDER = "uploads"
    DATA_FOLDER = "D:\Abdullah\Documents\codes\Projects\ArtiNex\static\data"
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'md'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB