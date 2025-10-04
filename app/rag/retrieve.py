# app/rag/retrieve.py
import pickle, pathlib
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = pathlib.Path("data/index")

class RAGStore:
    def __init__(self, topic_slug: str):
        self.topic_slug = topic_slug
        self.index = faiss.read_index((INDEX_DIR / f"{topic_slug}.faiss").as_posix())
        with open(INDEX_DIR / f"{topic_slug}.pkl", "rb") as f:
            obj = pickle.load(f)
        self.texts = obj["texts"]
        self.meta = obj["meta"]
        self.embed_model_name = obj["model_name"]
        self.embedder = SentenceTransformer(self.embed_model_name)

    def search(self, query: str, k: int = 8) -> List[Dict]:
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        out = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            m = self.meta[idx]
            out.append({
                "score": float(score),
                "text": self.texts[idx],
                "title": m["title"],
                "authors": m["authors"],
                "year": m["year"],
                "pdf_url": m["pdf_url"]
            })
        return out
