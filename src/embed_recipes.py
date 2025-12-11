import os
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import textwrap
import json

FILE_PATH = "../data/processed/clean_recipes_input.jsonl"
EMBD_PATH = "../data/embeddings/all-MiniLM-L6-v2.data"
ID_PATH = "../data/embeddings/all-MiniLM-L6-v2.txt"

BATCH_SIZE = 128
CHUNK_SIZE = 5000
EMB_DIM = 384
N_RECIPES = 231_637

def get_input_data(file_path, chunk_size):
    chunk = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_object = json.loads(line)
                chunk.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
                continue

            if len(chunk) == chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk

def get_embed_input_str(json_object):
    name = json_object["name"]
    description = json_object["description"] 
    ingredients = ", ".join(json_object["ingredients"])
    tags = ", ".join(json_object["tags"])

    emb_str = f"""
    name: {name}
    description: {description}
    ingredients: {ingredients}
    tags: {tags}
    """ 

    return textwrap.dedent(emb_str).strip()

def get_id(json_object):
    rec_id = json_object["id"]

    return rec_id

def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"    

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    model.eval()

    ensure_parent_dir(EMBD_PATH)
    ensure_parent_dir(ID_PATH)

    emb_mem = np.memmap(
        EMBD_PATH,
        dtype="float32",
        mode="w+",
        shape=(N_RECIPES, EMB_DIM),
    )

    id_out = open(ID_PATH, "w", encoding="utf-8")

    offset = 0
    for json_objects in get_input_data(FILE_PATH, CHUNK_SIZE):
        id_list = [get_id(json_object) for json_object in json_objects]
        embed_input_str = [get_embed_input_str(json_object) for json_object in json_objects]

        with torch.no_grad():
            document_embeddings_chunk = model.encode(
                embed_input_str,
                batch_size=BATCH_SIZE,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
    
        document_embeddings_chunk_np = document_embeddings_chunk.cpu().numpy()
        n = len(document_embeddings_chunk_np)
        emb_mem[offset:offset+n, :] = document_embeddings_chunk_np
        emb_mem.flush()
    
        for recipe_id in id_list:
            id_out.write(str(recipe_id) + "\n")

        offset += n
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    id_out.close()
    print("Done, total valid recipes:", offset)
