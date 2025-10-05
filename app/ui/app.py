# keep your path shim at the top if you added it
from pathlib import Path as _P
import sys as _sys
_ROOT = _P(__file__).resolve().parents[2]  # project root: ~/Desktop/literature-assistant
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from app.rag.retrieve import RAGStore
import app.agents.summarizer as summarizer

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from pathlib import Path
import sys

# PROJECT ROOT = <...>/literature-assistant
ROOT = Path(__file__).resolve().parents[2]   # go up from app/ui/ to project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# <<< end shim

from app.rag.retrieve import RAGStore
from app.agents.summarizer import llm_summary
import streamlit as st
from pathlib import Path as _Path  # to avoid name clash if you use Path later

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from pathlib import Path
import streamlit as st
from app.rag.retrieve import RAGStore
from app.agents.summarizer import llm_summary

st.title("Literature Assistant (MVP)")
topic = st.text_input("Topic (used when you built the index)", value="self-supervised learning in medical imaging")
question = st.text_area("Your question", value="What are dominant methods and open problems?")
k = st.slider("Top-K chunks", 4, 20, 10)

topic_slug = topic.strip().lower().replace(" ", "_")
index_faiss = Path("data/index") / f"{topic_slug}.faiss"

if not index_faiss.exists():
    st.warning("No index found for this topic. Build it first in Terminal:")
    st.code(f'python app/rag/build_index.py "{topic}"')
else:
    if st.button("Search & Summarize"):
        store = RAGStore(topic_slug)
        hits = store.search(question, k=k)
        with st.expander("Top hits (debug)"):
            for h in hits:
                st.markdown(f"**{h['title']} ({h['year']})** — [{h['pdf_url']}]({h['pdf_url']})")
                st.write(h["text"][:400] + "…")
        st.markdown("### Summary")
        st.write(llm_summary(hits, question))

