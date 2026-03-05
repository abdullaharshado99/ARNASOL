import numpy as np
from Anna_pipeline.config import RAGConfig
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self):
        self.config = RAGConfig()
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""

        return self.model.encode(text, normalize_embeddings=True)

if __name__=="__main__":
    emb = EmbeddingGenerator()