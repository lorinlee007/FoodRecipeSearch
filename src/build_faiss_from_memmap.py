
import os
import numpy as np
import faiss

EMBD_PATH = "../data/embeddings/all-MiniLM-L6-v2.data"
ID_PATH = "../data/embeddings/all-MiniLM-L6-v2.txt"
OUT_PATH = "../data/embeddings/all-MiniLM-L6-v2.faiss"
EMB_DIM = 384

def build_faiss_index(embeddings, ids):
    # Base index: inner product
    base_index = faiss.IndexFlatIP(EMB_DIM)

    # Wrap with ID map so FAISS stores your recipe IDs
    index = faiss.IndexIDMap2(base_index)

    # Add all vectors with IDs
    index.add_with_ids(embeddings, ids)

    return index

if __name__ == "__main__":
    file_size = os.path.getsize(EMBD_PATH)  # bytes
    bytes_per_vec = 4 * EMB_DIM            # float32 = 4 bytes
    N = file_size // bytes_per_vec

    print("Detected N (from memmap):", N)

    # loading the embeddings

    embeddings = np.memmap(
        EMBD_PATH,
        dtype="float32",
        mode="r",
        shape=(N, EMB_DIM),
    )

    # loading the IDs

    ids = np.loadtxt(ID_PATH, dtype="int64")

    print("Embeddings shape:", embeddings.shape)
    print("IDs shape:", ids.shape)

    # fail if IDs not equal embeddings
    assert embeddings.shape[0] == ids.shape[0], "Mismatch between embeddings and IDs!"

    print("Building Embeddings.....")
    index = build_faiss_index(embeddings, ids)
    print("Index ntotal:", index.ntotal)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    faiss.write_index(index, OUT_PATH)

    print("FAISS index saved.")
