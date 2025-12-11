import json
import os
import re
import pandas as pd

def clean_list(lst_str):
    if not isinstance(lst_str, str):
        return ""

    lst_str = lst_str.replace("[", "")
    lst_str = lst_str.replace("]", "")
    lst_str = lst_str.replace("'", "")
    lst_str = lst_str.replace("\"", "")

    tok = lst_str.split(",")
    tok = list(map(str.strip, tok))

    return tok

def min_desc_clean(desc_str):
    if not isinstance(desc_str, str):
        return ""

    desc_str = desc_str.replace("\r\n", "\n")
    desc_str = desc_str.replace("\r", "\n")

    lines = [line.strip() for line in desc_str.split("\n") if line.strip()]
    paragraph = ". ".join(lines)

    paragraph = re.sub(r"\s+", " ", paragraph)

    return paragraph.strip()

def min_clean(in_str):
    if not isinstance(in_str, str):
        return ""

    in_str = re.sub(r"\s+", " ", in_str)

    return in_str.strip()

PATH = "../data/raw/RAW_recipes.csv"

df = pd.read_csv(PATH)

df = df[["name", "id", "tags", "description", "ingredients", "steps"]]

df["name"] = df["name"].map(min_clean)
df["description"] = df["description"].map(min_desc_clean)
df["ingredients"] = df["ingredients"].map(clean_list)
df["steps"] = df["steps"].map(clean_list)
df["tags"] = df["tags"].map(clean_list)

OUT_PATH = "../data/processed/recipes_display.jsonl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w") as f:
    for _, row in df.iterrows():
        record = {
            "id": str(row["id"]),
            "name": row["name"],
            "description": row["description"],
            "tags": row["tags"],
            "ingredients": row["ingredients"],
            "steps": row["steps"]
        }
        f.write(json.dumps(record) + "\n")
