import numpy as np

from src.agentic_layer.agent import QueryAnalyzer, QueryType, AgenticRAG
from src.retrieval.retriever import Retriever, RetrievalStrategy


class FakeEmbedder:
    def embed_single_text(self, text):
        return np.array([float(len(text) % 7 + 1), 1.0, 0.5])

    def calculate_similarity(self, emb1, emb2):
        denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(np.dot(emb1, emb2) / denom) if denom else 0.0


class FakeVectorStore:
    def __init__(self):
        self.chunks = [
            "GPIO configuration requires setting MODER and AFR registers",
            "USART setup includes BRR baud-rate register and enable bits",
            "General troubleshooting for not working peripheral clocks"
        ]
        self.embeddings = [
            np.array([3.0, 1.0, 0.2]),
            np.array([2.0, 1.0, 0.3]),
            np.array([1.0, 1.0, 0.5]),
        ]
        self.metadata = [
            {"chunk_id": "c1", "section_type": "peripheral_description", "section_header": "GPIO", "pages": [10]},
            {"chunk_id": "c2", "section_type": "peripheral_description", "section_header": "USART", "pages": [20]},
            {"chunk_id": "c3", "section_type": "general_content", "section_header": "Troubleshooting", "pages": [30]},
        ]

    def similarity_search(self, query_embedding, k=5, filters=None):
        results = []
        for i, emb in enumerate(self.embeddings):
            sim = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)))
            meta = self.metadata[i]
            if filters and any(meta.get(fk) != fv for fk, fv in filters.items()):
                continue
            results.append({
                "chunk": self.chunks[i],
                "similarity": sim,
                "metadata": meta,
                "embedding_index": i,
                "embedding": emb,
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def get_chunk_by_id(self, chunk_id):
        idx = [m["chunk_id"] for m in self.metadata].index(chunk_id)
        return {"chunk": self.chunks[idx], "embedding": self.embeddings[idx], "metadata": self.metadata[idx]}


def test_query_analyzer_phrase_detection():
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("my uart is not working after setup")
    assert result["type"] == QueryType.TROUBLESHOOTING


def test_retriever_hybrid_and_embedding_reuse():
    retriever = Retriever(FakeVectorStore(), FakeEmbedder())
    results = retriever.retrieve("configure gpio setup", k=2, strategy=RetrievalStrategy.HYBRID, rerank=True)
    assert len(results) == 2
    assert all("reranked_similarity" in item for item in results)


def test_agent_dynamic_strategy_selection():
    retriever = Retriever(FakeVectorStore(), FakeEmbedder())
    agent = AgenticRAG(retriever)
    analysis = {"type": QueryType.PROCEDURAL, "complexity": "medium", "components": ["GPIO"]}
    assert agent._select_retrieval_strategy(analysis) == RetrievalStrategy.HYBRID
    assert agent._determine_retrieval_k(analysis) >= 6
