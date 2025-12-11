
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
import json
import re

INDEX_PATH = "../data/embeddings/all-MiniLM-L6-v2.faiss"
METADATA_PATH = "../data/processed/recipes_display.jsonl"
RECALL = 5 # How much to Recall

_SEARCHER = None  
_METADATA = None
_MODEL = None
_DEVICE = None

def light_normalize(input_str):
    if not isinstance(input_str, str):
        return ""

    input_str = input_str.lower()
    
    input_str = input_str.replace("&", " and ")
    input_str = input_str.replace("+", " and ")
    input_str = input_str.replace("/", " and ")
    input_str = input_str.replace("'", "")

    input_str = re.sub(r'[^a-z0-9\s\-]', ' ', input_str)
    input_str = re.sub(r'\s+', ' ', input_str).strip()

    return input_str

def embed_query(query, device, model):
    with torch.no_grad():
        emb = model.encode(
            [query],
            batch_size=1,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    emb = emb.cpu().numpy().astype("float32")
    return emb

def load_searcher(index_path):
    global _DEVICE
    global _MODEL
    _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                 device=_DEVICE)
    _MODEL.eval()

    index = faiss.read_index(index_path)
    print("Loaded FAISS index with ntotal =", index.ntotal)

    return index

def load_metadata(metadata_path):
    recipe_mdata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_object = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
                continue

            r_id = int(json_object["id"])
            recipe_mdata[r_id] = json_object

    return recipe_mdata

def run_query(searcher, metadata, query, device, model, k):
    query_emb = embed_query(light_normalize(query), device, model)
    scores, faiss_ids = searcher.search(query_emb, k)

    scores = scores[0]
    faiss_ids = faiss_ids[0]

    results = []

    for score, rid in zip(scores, faiss_ids):
        rid = int(rid)
        if rid == -1:
            continue

        obj = metadata.get(rid) 

        if obj is None:
            continue
        
        results.append({
            "id": rid,
            "score": float(score),
            "name": obj.get("name"),
            "description": obj.get("description"),
            "ingredients": obj.get("ingredients"),
            "tags": obj.get("tags"),
            })

    return results

def get_searcher(index_path=INDEX_PATH, metadata_path=METADATA_PATH, force_reload=False):
    global _SEARCHER
    global _METADATA
    if _SEARCHER is None or force_reload:
        searcher = load_searcher(index_path)
        _SEARCHER = searcher

    if _METADATA is None or force_reload:
        metadata = load_metadata(metadata_path)
        _METADATA = metadata

    return _SEARCHER, _METADATA

def search(query, recall=RECALL, searcher=None, metadata=None):
    if searcher is None or metadata is None:
        searcher, metadata = get_searcher()

    return run_query(searcher, metadata, query, _DEVICE, _MODEL, recall)

if __name__ == "__main__":
    q = "healthy quick meal"

    searcher, metadata = get_searcher()
    hits = search(q, searcher=searcher, metadata=metadata)

    for h in hits:
        print(f"{h['score']:.3f} | {h['id']} | {h['name']}")
