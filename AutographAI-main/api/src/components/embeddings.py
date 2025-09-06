# ─── src/semantic.py ────────────────────────────────────────────────
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

class SimpleEmbedder:
    """
    Tiny wrapper around Sentence‑Transformers to embed an arbitrary list of
    text fragments (records).  Stores   [{text, meta…}, …]   + a float32 matrix.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.records: List[Dict[str, Any]] = []
        self.emb: np.ndarray | None = None   # shape (N, dims)

    # ------------- building the index ---------------------------------
    def add_records(self, rows: List[Dict[str, Any]], text_key="text"):
        """
        rows – list of dicts, each MUST contain `text_key`
        """
        texts = [r[text_key] for r in rows]
        embeds = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
        self.records.extend(rows)
        self.emb = embeds if self.emb is None else np.vstack([self.emb, embeds])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, text_col: str):
        inst = cls()
        inst.add_records(df.to_dict("records"), text_key=text_col)
        return inst

    @classmethod
    def from_html(cls, html_path: str):
        soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "lxml")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all(["p", "li"]) if p.get_text(strip=True)]
        rows = [{"text": t} for t in paragraphs]
        return cls.from_dataframe(pd.DataFrame(rows), "text")

    # ------------- search ---------------------------------------------
    def search(self, query: str, top_k: int = 10, threshold: float = .75):
        q_emb = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        cos = util.cos_sim(q_emb, self.emb)[0].cpu().numpy()    # shape (N,)
        idx = np.argsort(-cos)[: top_k * 3]                     # rough cut
        hits = [
            {
                "name":     self.records[i].get("title") or self.records[i].get("text")[:60],
                "labels":   self.records[i].get("labels", []),
                "preview":  self.records[i].get("text")[:200] + ("…" if len(self.records[i].get("text")) > 200 else ""),
                "score":    float(cos[i]),
            }
            for i in idx
            if cos[i] >= threshold
        ]
        # return best `top_k`
        return sorted(hits, key=lambda h: -h["score"])[: top_k]
