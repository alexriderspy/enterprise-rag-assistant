"""Three retrievers: dense (embeddings), hybrid (embeddings + BM25), and hybrid + a cross-encoder reranker."""
import re
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.chunk import Chunk
from rag.config import EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, RERANK_MODEL
from rag.index import load_chunks

STOPWORDS = set("a an and are as at be by can do does for from how i if in is it my of on or the to what when "
                "where which who will with you your".split())


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: a chunk scores 1/(k + rank) in every list it appears in."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return scores


class Retriever:
    def __init__(self, chunking: str = "sections", mode: str = "rerank", candidates: int = 30):
        self.mode, self.candidates = mode, candidates
        self.chunks = {c.id: c for c in load_chunks(chunking)}
        client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
        self.collection = client.get_collection(f"handbook_{chunking}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(self.chunks[i].text) for i in self.ids])
        self.reranker = CrossEncoder(RERANK_MODEL) if mode == "rerank" else None

    def dense(self, query: str, n: int) -> list[str]:
        vec = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        return self.collection.query(query_embeddings=[vec.tolist()], n_results=n)["ids"][0]

    def keyword(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        return [self.ids[i] for i in scores.argsort()[::-1][:n]]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if self.mode == "dense":
            return [Hit(self.chunks[i], 0.0) for i in self.dense(query, k)]
        fused = rrf([self.dense(query, self.candidates), self.keyword(query, self.candidates)])
        ranked = sorted(fused, key=fused.get, reverse=True)[:self.candidates]
        if self.mode == "hybrid":
            return [Hit(self.chunks[i], fused[i]) for i in ranked[:k]]
        scores = self.reranker.predict([(query, self.chunks[i].text) for i in ranked])
        order = sorted(zip(ranked, scores), key=lambda x: x[1], reverse=True)
        return [Hit(self.chunks[i], float(s)) for i, s in order[:k]]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How much notice do I need to give before taking a week off?"
    for mode in ("dense", "hybrid", "rerank"):
        print(f"\n{mode}:")
        for n, hit in enumerate(Retriever(mode=mode).search(question, k=3), 1):
            print(f"  {n}. {hit.chunk.title[:100]}")
