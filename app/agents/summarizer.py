# app/agents/summarizer.py
import os
from typing import List, Dict

def _fallback_cited_summary(hits: List[Dict], max_chars: int = 1500) -> str:
    """Fallback summarizer that stitches top chunks with citations."""
    parts, total = [], 0
    for h in hits:
        author = h["authors"][0].split()[-1] if h.get("authors") else "Author"
        seg = f"[{author}, {h.get('year','n.d.')}] {h.get('text','')[:300].replace('\\n',' ')} … (source: {h.get('pdf_url','')})"
        parts.append(seg)
        total += len(seg)
        if total >= max_chars:
            break
    return "\n\n".join(parts)

def llm_summary(hits: List[Dict], question: str) -> str:
    """
    Summarize using OpenAI if OPENAI_API_KEY is set; otherwise use the fallback.
    Returns markdown text with inline [Author, Year] citations and links.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return _fallback_cited_summary(hits)

    try:
        from openai import OpenAI
        client = OpenAI()
        context = "\n\n".join([
            f"### {h.get('title','')} ({h.get('year','')}) | {h.get('pdf_url','')}\n{h.get('text','')}"
            for h in hits
        ])
        prompt = f"""Use only the CONTEXT to answer the QUESTION in 5–8 concise bullets.
Every bullet must include a citation like [Author, Year] and the source link.

QUESTION: {question}

CONTEXT:
{context}"""
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # If anything fails, fall back gracefully so the app still works
        return _fallback_cited_summary(hits) + f"\n\n_(LLM fallback due to: {e})_"
