from typing import List, Dict, Any
from Anna_pipeline.config import RAGConfig
from Anna_pipeline.vector_store import VectorStore
from Anna_pipeline.embeddings import EmbeddingGenerator


class QueryEngine:
    def __init__(self):
        self.config = RAGConfig()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def search(self, query: str) -> List[Dict[str, Any]]:

        query_embedding = self.embedding_generator.generate_single_embedding(query)
        results = self.vector_store.retrieve_similar(query_embedding.tolist(), self.config.TOP_K_RESULTS)

        return results

    @staticmethod
    def format_results_for_llm(results: List[Dict[str, Any]]) -> str:

        if not results:
            return "No relevant documents found."

        formatted = "Here are the most relevant documents:\n\n"

        for i, result in enumerate(results, 1):
            # Add document header
            filename = result.get('metadata', {}).get('file_name', 'Unknown')
            formatted += f"[Document {i}] (Source: {filename}, Relevance: {result['similarity_score']:.2f})\n"

            # Add text excerpt
            text = result['text']
            if len(text) > 500:
                text = text[:500] + "..."
            formatted += f"{text}\n\n"

        return formatted

    def get_context_for_query(self, query: str) -> Dict[str, Any]:

        # Retrieve relevant documents
        results = self.search(query)

        # Format context
        context = self.format_results_for_llm(results)

        # Extract sources
        sources = list(set([
            r.get('metadata', {}).get('file_name', 'Unknown')
            for r in results if r.get('metadata')
        ]))

        return {
            'query': query,
            'context': context,
            'sources': sources,
            'num_results': len(results),
            'raw_results': results
        }

if __name__=="__main__":
    ret = QueryEngine()