from pyserini.search.lucene import LuceneSearcher
import json
import re

INDEX_PATH = "../data/bm25/index"
BM25_K1 = 0.9
BM25_B = 0.4
RECALL = 5

_SEARCHER = None

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

def load_searcher(index_path=INDEX_PATH):
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    return searcher

def run_query(searcher, query, recall):
    norm_query = light_normalize(query)
    hits = searcher.search(norm_query, k=recall)

    results = []

    for rank, hit in enumerate(hits):
        lucene_doc = searcher.doc(hit.docid)
        obj = json.loads(lucene_doc.raw())
        raw = obj.get("raw", "")
    
        name = ""
        description = ""
        tags = []
        ingredients = []

        if raw != "":
            dsp_json = json.loads(raw)
            name = dsp_json.get("name", "")
            tags = dsp_json.get("tags", [])
            ingredients = dsp_json.get("ingredients", [])
            description = dsp_json.get("description", "")

        results.append({
            "rank": rank + 1,
            "id": int(hit.docid),
            "score": hit.score,
            "name": name,
            "description": description,
            "tags": tags,
            "ingredients": ingredients
        })

    return results

def get_searcher(index_path=INDEX_PATH, force_reload=False):
    global _SEARCHER
    if _SEARCHER is None or force_reload:
        searcher = load_searcher(index_path) 
        _SEARCHER = searcher
    
    return _SEARCHER

def search(query, recall=RECALL, searcher=None):
    if searcher is None:
        searcher = get_searcher()

    return run_query(searcher, query, recall)

if __name__ == "__main__":
    queries = ["baked salmon with lemon", "texas sheet cake", "low fat chicken",
             "molasses ginger cookies", "broccoli cheddar soup"]

    searcher = get_searcher()
    for query in queries:
        results = search(query, RECALL, searcher) 

        for r in results:
            print(f"{r['rank']:2d}. {r['score']:.3f} | {r['id']} | {r['name']}")

        print()

