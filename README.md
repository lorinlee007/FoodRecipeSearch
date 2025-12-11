
# Food Recipe Search: Semantic vs. Keyword search

This project compares two retrieval approaches for recipe search: a traditional Lucene-based text index using BM25 scoring and a dense-embedding retrieval system built on a FAISS nearest-neighbor index. By running both systems on the same set of natural-language queries, we evaluate how effectively each method retrieves relevant recipes, with a particular focus on recall for complex, real-world query phrasing.

## Setup instructions 

1. Install Git Large File Storage (Git LFS) 
This project uses Git LFS because the repository includes large index files (FAISS index + Lucene index). Before cloning or pulling the repo, install Git LFS.

On macOS (Homebrew):

```
brew install git-lfs
git lfs install
```

You only need to run git lfs install once per machine.

2. Create the Conda Environment
The project includes an `environment.yml` file that defines all Python dependencies, including Pyserini, FAISS, Java21 and supporting libraries.

Create the environment using:

```
conda env create -f environment.yml
conda activate foodrec
```

3. Java Dependency (Required for Lucene / Pyserini)

Pyserini internally uses Lucene, which requires a matching Java runtime. For this project, you must use Java 21, because that is the version compatible with the Pyserini build used here.  

The environment file installs Java 21 via Conda (openjdk=21). However, if your system already has Java installed (e.g., via Homebrew), the shell may resolve to the system Java instead of the Conda one depending on your PATH / .zshrc configuration. 

3.1 Check what Java Conda installed

After activating the environment:

```
conda activate foodrec
echo "$CONDA_PREFIX"
ls "$CONDA_PREFIX/lib/jvm/bin/java"
"$CONDA_PREFIX/lib/jvm/bin/java" -version
```

You should see a Java 21 version string. If the ls command fails, search for the Java binary inside the environment:

```
find "$CONDA_PREFIX" -maxdepth 4 -name java
```

Pick the java path that lives inside this Conda environment (somewhere under $CONDA\_PREFIX).

3.2 Make sure Java 21 takes precedence

Check what your shell is currently using:

```
which java
java -version
```

If this does not show Java 21 from the Conda environment, your system Java is taking precedence (often from /usr/bin or /opt/homebrew).

To force the Conda JDK to be used whenever this environment is active:

```
export JAVA_HOME="$CONDA_PREFIX/lib/jvm"
export PATH="$JAVA_HOME/bin:$PATH"
```

Verify

```
java -version
```

## Running the Project

**Run all Python scripts from inside the src directory so their relative paths resolve**

### Step 1 [Data Processing]

First run `data_process.py` and `build_dsp_metadata.py` from the **src** directory (order doesn’t matter). Both scripts read `data/raw/RAW_recipes.csv` and emit cleaned JSONL files: `data_process.py` produces `data/processed/clean_recipes_input.jsonl`, which later feeds the embedding builder (embed_recipes.py) and the BM25 corpus/index (create_bm25_corpus.py, plus any Lucene build step). `build_dsp_metadata.py` writes `data/processed/recipes_display.jsonl`, which supplies the rich metadata shown in semantic and BM25 search results.

### Step 2 [Embedding Index Building]

Run python `embed_recipes.py` from **src** to build embeddings. It consumes `data/processed/clean_recipes_input.jsonl`, processes recipes in chunks of CHUNK\_SIZE = 5000, encodes each chunk in batches of BATCH_SIZE = 128 using the all-MiniLM-L6-v2 SentenceTransformer (embedding dimension EMB_DIM = 384), and writes N_RECIPES = 231,637 normalized vectors to `data/embeddings/all-MiniLM-L6-v2.data` (NumPy memmap) plus matching IDs to `data/embeddings/all-MiniLM-L6-v2.txt`. These outputs feed `build_faiss_from_memmap.py` to create the FAISS index.

Run python `build_faiss_from_memmap.py` from inside **src** after `embed_recipes.py` completes. The script reads the memmapped embeddings (`data/embeddings/all-MiniLM-L6-v2.data`) and matching ID list (`data/embeddings/all-MiniLM-L6-v2.txt1`), infers N from the file size (each 384‑dim vector stored as float32, so 4 * EMB_DIM bytes per vector), and asserts the ID count matches. It builds a FAISS IndexFlatIP wrapped in IndexIDMap2 so each vector retains its recipe ID, adds the entire dataset in one call (index.add_with_ids), and saves the resulting index to `data/embeddings/all-MiniLM-L6-v2.faiss` for semantic search.

Run python `search_faiss.py` from **src** once the FAISS index and display metadata exist. The script lazily loads `data/embeddings/all-MiniLM-L6-v2.faiss`, the metadata from `data/processed/recipes_display.jsonl`, and the same all-MiniLM-L6-v2 SentenceTransformer. For each query (e.g., the sample \"healthy quick meal\" in \_\_main\_\_), it normalizes the text, encodes it on the available device (MPS if present, otherwise CPU), searches the FAISS index with the configured recall (RECALL = 5), and prints a list of top hits with scores, recipe IDs, and names. This is the semantic search demo.

### Step 3 [BM25 Index Building]

Run python `create_bm25_corpus.py` from **src** after generating both `data/processed/clean_recipes_input.jsonl` and `data/processed/recipes_display.jsonl`. The script merges the normalized recipe fields with their display metadata, builds a contents string that intentionally repeats the recipe name three times to overweight exact title matches, then appends the description, tags, and ingredients. For each recipe it emits a JSONL document (id, contents, raw) under `data/bm25/corpus/recipes.jsonl`, where raw stores the pretty metadata (minus the ID) so Lucene search results can reconstruct names/descriptions/tags/ingredients for display. This corpus feeds your Lucene/BM25 indexing step.

Next build the Lucene Index, run the following command from the **src** directory. Please note that this is the step that requires Java21. Once the following command complets you will see your Lucene Index in `data/bm25/index/*`

```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ../data/bm25/corpus \
  --index ../data/bm25/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw
```

Run python `search_bm25.py` from **src** once you’ve built a Lucene index under `data/bm25/index`. The script loads that index with Pyserini, applies the same light_normalize as earlier, sets BM25 parameters (k1 = 0.9, b = 0.4), and executes queries (see the sample list in \_\_main\_\_). For each hit it retrieves the stored raw JSON, decodes the display metadata, and prints rank, score, ID, and name. This is the keyword/BM25 demo.

## Trained Model (Embeddings + FAISS Index)

This project does not train a neural network model. Instead, it relies on a pre-trained SentenceTransformer (`all-MiniLM-L6-v2`) to generate dense vector embeddings for all recipes. After encoding, each embedding is L2-normalized and stored in:

- `data/embeddings/all-MiniLM-L6-v2.data` — the normalized embedding matrix (NumPy memmap)

These embeddings serve as the semantic representation of each recipe.

To enable fast nearest-neighbor search, the embeddings are then added to a FAISS index:

- `data/embeddings/all-MiniLM-L6-v2.faiss` — the FAISS index used for semantic retrieval

In practice, the combination of:
1. **the embedding generator** (pre-trained SentenceTransformer), and  
2. **the FAISS index** (efficient similarity search)

functions as the semantic search model for this project.

No fine-tuning or supervised training is performed.

## Demo Notebook

The repository includes a `demo/` directory containing `demo.ipynb`.  
This notebook provides a quick sanity check for the project: it loads both the FAISS embedding index and the Lucene BM25 index, runs a set of five example queries, and displays the top results from each retrieval system. For each query, you should see the **score**, **recipe ID**, and **recipe name**, which confirms that both search engines are functioning correctly and returning meaningful results.

## Results Notebook

The `results/` directory contains `results.ipynb`, which performs the full evaluation of the retrieval systems. This notebook runs a set of 15 complex, natural-language recipe queries and compares the outputs of BM25 and FAISS. For each query, the notebook computes **MRR@5** and **Precision@5**, allowing you to quantitatively evaluate how well each search engine performs on realistic user-style queries.