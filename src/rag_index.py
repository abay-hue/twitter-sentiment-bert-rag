import os, faiss, numpy as np
from transformers import AutoTokenizer, AutoModel
import torch, json

EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _embed(texts, tok, enc):
    with torch.no_grad():
        embs = []
        for t in texts:
            x = tok(t, return_tensors="pt", truncation=True, padding=True)
            y = enc(**x).last_hidden_state.mean(dim=1)  # simple mean-pooled CLS-free embedding
            embs.append(y.squeeze(0).numpy())
        return np.vstack(embs)

def build_corpus_index(corpus_txt_path="data/processed/corpus.txt", out="models/faiss.index"):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    enc = AutoModel.from_pretrained(EMB_MODEL)
    texts = [l.strip() for l in open(corpus_txt_path, "r", encoding="utf-8") if l.strip()]
    X = _embed(texts, tok, enc).astype("float32")
    idx = faiss.IndexFlatIP(X.shape[1])
    faiss.normalize_L2(X)
    idx.add(X)
    faiss.write_index(idx, out)
    meta = {"texts": texts}
    json.dump(meta, open(out + ".meta.json", "w"))
    print("✅ Built index with", len(texts), "docs")

def query(q, idx_path="models/faiss.index", k=3):
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    enc = AutoModel.from_pretrained(EMB_MODEL)
    idx = faiss.read_index(idx_path)
    meta = json.load(open(idx_path + ".meta.json"))
    x = _embed([q], tok, enc).astype("float32")
    faiss.normalize_L2(x)
    D, I = idx.search(x, k)
    return [{"text": meta["texts"][i], "score": float(D[0][j])} for j, i in enumerate(I[0])]
