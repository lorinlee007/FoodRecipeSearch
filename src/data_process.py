import os
import pandas as pd
import json
import re

MEASUREMENT_WORDS = {
# units
"cup", "cups", 
"teaspoon", "teaspoons", "tsp", "tsps", "tablespoon", "tablespoons", "tbsp", "tbsps",
"ml", "milliliter", "milliliters",
"liter", "liters", "litre", "litres", "l",
"ounce", "ounces", "oz",
"gram", "grams", "g",
"kilogram", "kilograms", "kg",
"pound", "pounds", "lb", "lbs",

# counts / packaging
"pinch", "pinches",
"dash", "dashes",
"slice","slices",
"clove","cloves",
"can","cans",
"jar","jars",
"packet", "packets",
"package", "packages",
"stick", "sticks",
"bunch", "bunches",
}

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

def clean_and_get_tags(tags):
    if not isinstance(tags, str):
        return []

    tags = tags.replace("[", "")
    tags = tags.replace("]", "")
    tags = tags.replace("'", "")
    tags = tags.replace("\"", "")

    tags = tags.split(",")

    tags = list(map(light_normalize, tags))
    tags = [item for item in tags if item]
    
    return tags

def normalize_ingredients(input_str):
    if not isinstance(input_str, str):
        return ""

    input_str = input_str.lower()
    
    input_str = input_str.replace("&", " and ")
    input_str = input_str.replace("+", " and ")

    input_str = re.sub(r'\d+', ' ', input_str)
    input_str = re.sub(r'[^a-z\s\-]', ' ', input_str)
    input_str = re.sub(r'\s+', ' ', input_str).strip()

    tokens = input_str.split()

    tokens = [t for t in tokens if t not in MEASUREMENT_WORDS]

    return " ".join(tokens) 

def clean_and_get_ingredients(ingredients):
    if not isinstance(ingredients, str):
        return []

    ingredients = ingredients.replace("[", "")
    ingredients = ingredients.replace("]", "")
    ingredients = ingredients.replace("'", "")
    ingredients = ingredients.replace("\"", "")

    ingredients = ingredients.split(",")

    ingredients = list(map(light_normalize, ingredients))
    ingredients = [item for item in ingredients if item]
    
    return ingredients

PATH = "../data/raw/RAW_recipes.csv"

df = pd.read_csv(PATH)

df = df[["name", "id", "tags", "description", "ingredients"]]

df["name"] = df["name"].map(light_normalize)
df["tags"] = df["tags"].map(clean_and_get_tags)
df["description"] = df["description"].map(light_normalize)
df["ingredients"] = df["ingredients"].map(clean_and_get_ingredients)

OUT_PATH = "../data/processed/clean_recipes_input.jsonl"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w") as f:
    for _, row in df.iterrows():
        record = {
            "id": str(row["id"]),
            "name": row["name"],
            "description": row["description"],
            "tags": row["tags"],
            "ingredients": row["ingredients"],
        }
        f.write(json.dumps(record) + "\n")
