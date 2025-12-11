import os
import pandas as pd

FILE_PATH = "../data/processed/clean_recipes_input.jsonl" 
OUTPUT_PATH = "../data/samples/sample_recipes.jsonl"

df = pd.read_json(FILE_PATH, lines=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.sample(50).to_json(OUTPUT_PATH, orient="records", lines=True)
