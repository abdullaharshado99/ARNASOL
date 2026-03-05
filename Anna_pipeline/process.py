from Anna_pipeline.config import RAGConfig
from Anna_pipeline.vector_store import VectorStore
from Anna_pipeline.embeddings import EmbeddingGenerator
from Anna_pipeline.document_processor import DocumentProcessor

doc_processor = DocumentProcessor()
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore()
config = RAGConfig()

def process_and_store_document():
    """Process a document and store in vector DB"""

    chunks, metadata = doc_processor.process_document(config.DATA_FOLDER)

    embeddings = embedding_generator.generate_embeddings(chunks)

    count = vector_store.add_documents(chunks, metadata, embeddings)

    return {
        'status': 'success',
        'chunks_created': len(chunks),
        'stored': count
    }

if __name__ == "__main__":
    print(process_and_store_document())