"""Embed every chunk with bge-small and store it in Chroma, one collection per chunking strategy."""
import argparse
import json
import time
from dataclasses import asdict

import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunk import Chunk, fixed_chunks, section_chunks
from rag.config import EMBED_MODEL, INDEX_DIR
from rag.parse import load_sections

CHUNKERS = {"fixed": fixed_chunks, "sections": section_chunks}


def build(chunking: str) -> None:
    chunks = CHUNKERS[chunking](load_sections())
    model = SentenceTransformer(EMBED_MODEL)
    t0 = time.time()
    vectors = model.encode([c.text for c in chunks], batch_size=32, normalize_embeddings=True)
    print(f"embedded {len(chunks)} chunks in {time.time() - t0:.1f}s, vectors of size {vectors.shape[1]}")

    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    name = f"handbook_{chunking}"
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c.id for c in chunks], embeddings=vectors.tolist(), documents=[c.text for c in chunks],
            metadatas=[{"title": c.title, "url": c.url, "section_ids": ",".join(c.section_ids)} for c in chunks])
    (INDEX_DIR / f"{chunking}_chunks.json").write_text(json.dumps([asdict(c) for c in chunks]))
    print(f"collection {name}: {col.count()} chunks saved to index/chroma")


def load_chunks(chunking: str) -> list[Chunk]:
    return [Chunk(**c) for c in json.loads((INDEX_DIR / f"{chunking}_chunks.json").read_text())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunking", choices=list(CHUNKERS), default="sections")
    build(ap.parse_args().chunking)
