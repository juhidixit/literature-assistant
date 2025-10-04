# app/rag/build_index.py

import os, json, pathlib, pickle
from typing import List, Dict
import arxiv
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

DATA_DIR = pathlib.Path("data")
INDEX_DIR = pathlib.Path("data/index")

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search arXiv for papers on a given topic."""
    results = []
    for r in arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    ).results():
        results.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "summary": r.summary,
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id
        })
    return results

def download_pdf(url: str, to_path: pathlib.Path) -> pathlib.Path:
    """Download a PDF from arXiv if not already present."""
    to_path.parent.mkdir(parents=True, exist_ok=True)
    if to_path.exists(): return to_path
    import urllib.request
    urllib.request.urlretrieve(url, to_path.as_posix())
    return to_path

def pdf_to_text(pdf_path: pathlib.Path) -> str:
    """Extract text from a PDF."""
    doc = fitz.open(pdf_path.as_posix())
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

def chunk_text(text: str) -> List[str]:
    """Split long text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_text(text)

def build_embeddings(chunks: List[str], model_name: str):
    """Generate embeddings for text chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return model, embeddings

def build_index_for_topic(topic: str, max_results: int = 10, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Complete pipeline: search papers → download → extract text → embed → save FAISS index."""
    topic_slug = topic.strip().lower().replace(" ", "_")
    topic_dir = DATA_DIR / topic_slug
    pdf_dir = topic_dir / "pdfs"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Searching arXiv for: {topic}")
    papers = search_arxiv(topic, max_results=max_results)
    topic_dir.mkdir(parents=True, exist_ok=True)

    (topic_dir / "papers.json").write_text(json.dumps(papers, indent=2))

    print(f"[2/5] Downloading PDFs")
    texts, meta = [], []
    for i, p in enumerate(papers):
        safe_name = "".join(c for c in p["title"] if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")[:60]
        pdf_path = pdf_dir / f"{i:02d}_{safe_name}.pdf"
        try:
            download_pdf(p["pdf_url"], pdf_path)
            raw_text = pdf_to_text(pdf_path)
            chunks = chunk_text(raw_text)
            for j, ch in enumerate(chunks):
                texts.append(ch)
                meta.append({"paper_idx": i, "chunk_idx": j, **p})
        except Exception as e:
            print(f"  ! Skipped {p['title'][:50]}... ({e})")

    print(f"[3/5] Creating embeddings for {len(texts)} chunks...")
    model, embs = build_embeddings(texts, embed_model)

    print(f"[4/5] Building FAISS index...")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    print(f"[5/5] Saving index and metadata...")
    faiss.write_index(index, (INDEX_DIR / f"{topic_slug}.faiss").as_posix())
    with open(INDEX_DIR / f"{topic_slug}.pkl", "wb") as f:
        pickle.dump({"texts": texts, "meta": meta, "model_name": embed_model}, f)
    print("✅ Done. Index built successfully!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python app/rag/build_index.py 'your topic here'")
        raise SystemExit(1)
    build_index_for_topic(sys.argv[1])
