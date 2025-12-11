import json
import os

INPUT_PATH = "../data/processed/clean_recipes_input.jsonl"
INPUT_DISPLAY_PATH = "../data/processed/recipes_display.jsonl"
OUTPUT_DIR = "../data/bm25/corpus"
OUTPUT_PATH =  OUTPUT_DIR + "/recipes.jsonl"

def get_raw():
    raw_map = {}
    
    with open(INPUT_DISPLAY_PATH, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            rid = str(obj["id"])
            
            display_json = obj.copy()
            display_json.pop("id", None)

            raw_map[rid] = display_json 

    return raw_map

def get_contents(obj):
    name = obj.get("name", "") 
    desc = obj.get("description", "") 
    tags = obj.get("tags", [])
    ingredients = obj.get("ingredients", [])

    contents = (
        (name + " ") * 3 + 
        desc + " " + 
        " ".join(tags) + " " +
        " ".join(ingredients)
    )

    return contents

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_map = get_raw()

    with open(INPUT_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
        for line in fin:
            line = line.strip()
            json_obj = json.loads(line)

            rid = str(json_obj["id"])

            contents = get_contents(json_obj)

            display_obj = raw_map.get(rid, {})
            
            doc = {
                "id": rid,
                "contents": contents,
                "raw": json.dumps(display_obj)
            }
            
            fout.write(json.dumps(doc) + "\n")
