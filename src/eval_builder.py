import pandas as pd
import json
import os

INPUT_PATH = "../data/processed/clean_recipes_input.jsonl"
EVAL_DIR = "../data/eval"

query_specs = [
    # ---------- Keyword-y queries ----------
    {
        "qid": 1,
        "query": "chocolate chip cookies",
        "type": "keyword",
        # cookies live under desserts + cookies-and-brownies in your sample
        "must_tags": ["desserts"],
        "any_tags": ["cookies-and-brownies"],
        "must_not_tags": [],
        "must_ingredients": ["chocolate", "chip"],
        "must_not_ingredients": []
    },
    {
        "qid": 2,
        "query": "chicken curry",
        "type": "keyword",
        # "curries" tag appears in your data (indonesian prawns)
        "must_tags": ["curries"],
        "any_tags": ["main-dish"],
        "must_not_tags": [],
        "must_ingredients": ["chicken"],
        "must_not_ingredients": []
    },
    {
        "qid": 3,
        "query": "banana bread",
        "type": "keyword",
        "must_tags": ["breads"],
        "any_tags": ["breakfast"],
        "must_not_tags": [],
        "must_ingredients": ["banana"],
        "must_not_ingredients": []
    },
    {
        "qid": 4,
        "query": "beef stew slow cooker",
        "type": "keyword",
        # slow cooker + soups/stews style
        "must_tags": ["crock-pot-slow-cooker"],
        "any_tags": ["soups-stews", "stews"],
        "must_not_tags": [],
        "must_ingredients": ["beef"],
        "must_not_ingredients": []
    },
    {
        "qid": 5,
        "query": "tomato basil soup",
        "type": "keyword",
        "must_tags": ["soups-stews"],
        "any_tags": ["healthy-2", "vegetables"],
        "must_not_tags": [],
        "must_ingredients": ["tomato"],
        "must_not_ingredients": []
    },
    {
        "qid": 6,
        "query": "garlic butter shrimp",
        "type": "keyword",
        "must_tags": ["seafood"],
        "any_tags": ["main-dish", "shrimp"],
        "must_not_tags": [],
        "must_ingredients": ["shrimp", "garlic"],
        "must_not_ingredients": []
    },
    {
        "qid": 7,
        "query": "lemon pepper chicken wings",
        "type": "keyword",
        # wings are often appetizers / lunch / chicken / wings
        "must_tags": ["poultry"],
        "any_tags": ["appetizers", "lunch"],
        "must_not_tags": [],
        "must_ingredients": ["chicken", "wing"],
        "must_not_ingredients": []
    },
    {
        "qid": 8,
        "query": "spaghetti carbonara",
        "type": "keyword",
        "must_tags": ["pasta"],
        "any_tags": ["main-dish", "italian"],
        "must_not_tags": [],
        "must_ingredients": ["spaghetti"],
        "must_not_ingredients": []
    },
    {
        "qid": 9,
        "query": "blueberry muffins",
        "type": "keyword",
        "must_tags": ["muffins"],
        "any_tags": ["breakfast", "breads"],
        "must_not_tags": [],
        # "blueber" to catch blueberry / blueberries
        "must_ingredients": ["blueber"],
        "must_not_ingredients": []
    },
    {
        "qid": 10,
        "query": "avocado toast",
        "type": "keyword",
        "must_tags": ["breakfast"],
        "any_tags": ["brunch", "snacks"],
        "must_not_tags": [],
        "must_ingredients": ["avocado"],
        "must_not_ingredients": []
    },

    # ---------- Semantic / complex queries ----------
    {
        "qid": 11,
        "query": "high-protein vegetarian dinner without cheese",
        "type": "semantic",
        # we saw "vegetarian" tag in your data
        "must_tags": ["vegetarian", "main-dish"],
        # proxy for healthy / protein-ish
        "any_tags": ["healthy", "healthy-2", "low-carb"],
        "must_not_tags": [],
        "must_ingredients": [],
        # remove obvious cheese / dairy
        "must_not_ingredients": [
            "cheese", "mozzarella", "parmesan", "feta", "cheddar",
            "cream cheese", "sour cream", "half-and-half", "buttermilk"
        ]
    },
    {
        "qid": 12,
        "query": "quick weeknight meal for kids who are picky eaters",
        "type": "semantic",
        "must_tags": ["kid-friendly"],
        "any_tags": ["30-minutes-or-less", "60-minutes-or-less", "easy", "weeknight"],
        "must_not_tags": [],
        "must_ingredients": [],
        # don't block actual spice words too hard, just an example
        "must_not_ingredients": []
    },
    {
        "qid": 13,
        "query": "healthy breakfast that keeps you full for long",
        "type": "semantic",
        "must_tags": ["breakfast"],
        "any_tags": ["healthy", "healthy-2", "high-fiber", "high-in-something"],
        "must_not_tags": ["desserts"],
        "must_ingredients": [],
        "must_not_ingredients": []
    },
    {
        "qid": 14,
        "query": "low-sodium soup for cold weather",
        "type": "semantic",
        "must_tags": ["soups-stews"],
        "any_tags": ["low-sodium", "low-in-something", "healthy-2", "fall", "winter"],
        "must_not_tags": [],
        "must_ingredients": [],
        "must_not_ingredients": []
    },
    {
        "qid": 15,
        "query": "cheap meals to make with pantry staples only",
        "type": "semantic",
        "must_tags": [],
        "any_tags": ["inexpensive", "easy", "5-ingredients-or-less", "3-steps-or-less"],
        "must_not_tags": [],
        "must_ingredients": [],
        "must_not_ingredients": []
    },
    {
        "qid": 16,
        "query": "dessert that is light and not too sweet",
        "type": "semantic",
        "must_tags": ["desserts"],
        "any_tags": ["fruit", "berries", "low-in-something"],
        "must_not_tags": [],
        "must_ingredients": [],
        # block heavy sugary toppings a bit
        "must_not_ingredients": ["frosting", "icing"]
    },
    {
        "qid": 17,
        "query": "low-carb dinner for someone trying to lose weight",
        "type": "semantic",
        "must_tags": [],
        "any_tags": ["low-carb", "very-low-carbs", "healthy-2", "main-dish"],
        # avoid pasta/rice-heavy things via tags and ingredients
        "must_not_tags": ["pasta", "rice"],
        "must_ingredients": [],
        "must_not_ingredients": ["pasta", "spaghetti", "macaroni", "rice", "bread", "tortilla"]
    },
    {
        "qid": 18,
        "query": "vegan comfort food similar to mac and cheese",
        "type": "semantic",
        "must_tags": ["vegan"],
        "any_tags": ["comfort-food", "pasta"],
        "must_not_tags": [],
        "must_ingredients": [],
        "must_not_ingredients": [
            "cheese", "milk", "butter", "cream", "sour cream", "cream cheese"
        ]
    },
    {
        "qid": 19,
        "query": "easy lunch I can pack for work without reheating",
        "type": "semantic",
        "must_tags": ["lunch"],
        "any_tags": ["no-cook", "salads", "easy", "to-go", "brown-bag"],
        # avoid soups/stews that basically require reheating
        "must_not_tags": ["soups-stews"],
        "must_ingredients": [],
        "must_not_ingredients": ["soup"]
    },
    {
        "qid": 20,
        "query": "gluten-free snack that toddlers will actually eat",
        "type": "semantic",
        "must_tags": ["snacks"],
        "any_tags": ["gluten-free", "kid-friendly", "toddler-friendly"],
        "must_not_tags": [],
        "must_ingredients": [],
        "must_not_ingredients": []
    }
]

def auto_relevant_ids(df, spec, max_n=50):
    must_tags       = set(spec.get("must_tags", []))
    any_tags        = set(spec.get("any_tags", []))
    must_not_tags   = set(spec.get("must_not_tags", []))
    must_ings       = [s.lower() for s in spec.get("must_ingredients", [])]
    must_not_ings   = [s.lower() for s in spec.get("must_not_ingredients", [])]

    def match(row):
        tags = set(row["tags"]) if isinstance(row["tags"], list) else set()
        ing_text = " ".join(row["ingredients"]).lower() if isinstance(row["ingredients"], list) else ""

        # must_tags: all must be present
        if must_tags and not must_tags.issubset(tags):
            return False

        # any_tags: at least one must match (if non-empty)
        if any_tags and not (tags & any_tags):
            return False

        # must_not_tags: none allowed
        if must_not_tags and (tags & must_not_tags):
            return False

        # must_ingredients: all substrings must appear
        for kw in must_ings:
            if kw and kw not in ing_text:
                return False

        # must_not_ingredients: none may appear
        for kw in must_not_ings:
            if kw and kw in ing_text:
                return False

        return True

    mask = df.apply(match, axis=1)
    # return df.loc[mask, "id"].head(max_n).tolist()
    return df.loc[mask, "id"].tolist()


df = pd.read_json(INPUT_PATH, lines=True)

for spec in query_specs:
    spec["relevant_ids"] = auto_relevant_ids(df, spec, max_n=50)

for spec in query_specs:
    print(spec["qid"], spec["type"], "=>", len(spec["relevant_ids"]), "matches")

os.makedirs(EVAL_DIR, exist_ok=True)

with open(EVAL_DIR + "/queries.json", "w") as f:
    json.dump(query_specs, f, indent=2)
