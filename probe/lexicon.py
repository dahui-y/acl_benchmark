"""Lexical resources for building aspect-controlled stimuli.

The taxonomy in ``action_object_taxonomy/`` stores actions as gerunds and objects
as bare nouns. Building the aspect conditions needs base and past forms of each
verb, plurals of each object, and a note of which objects are mass nouns (they
cannot carry the bare-plural atelic contrast).
"""

# gerund -> (base, past). Hardcoded rather than rule-derived: the list is small
# and closed, and the rules would get "sauteing"/"zesting" wrong.
VERB_FORMS = {
    "chopping": ("chop", "chopped"),
    "mincing": ("mince", "minced"),
    "shredding": ("shred", "shredded"),
    "slicing": ("slice", "sliced"),
    "frying": ("fry", "fried"),
    "sauteing": ("saute", "sauteed"),
    "grilling": ("grill", "grilled"),
    "roasting": ("roast", "roasted"),
    "browning": ("brown", "browned"),
    "crushing": ("crush", "crushed"),
    "mashing": ("mash", "mashed"),
    "squeezing": ("squeeze", "squeezed"),
    "grating": ("grate", "grated"),
    "zesting": ("zest", "zested"),
    "whipping": ("whip", "whipped"),
    "blending": ("blend", "blended"),
    "coating": ("coat", "coated"),
    "rolling": ("roll", "rolled"),
    "peeling": ("peel", "peeled"),
    "melting": ("melt", "melted"),
}

# Objects that resist the bare-plural atelic contrast. Items in the taxonomy that
# are mass nouns ("spinach", "meat") or that are already category labels
# ("vegetable", "citrus") are excluded from the main matrix so that the
# telic/atelic contrast stays a clean count-noun alternation.
MASS_OR_GENERIC = {
    "vegetable", "leaf", "meat", "seafood", "fish", "citrus", "berry",
    "spinach", "kale", "lettuce", "broccoli", "cauliflower", "cabbage",
    "asparagus", "corn", "ginger", "garlic", "butter", "cheese", "milk",
    "cream", "chocolate", "dough", "bread", "rice", "pasta", "beef", "pork",
    "chicken", "turkey", "ham", "bacon", "shrimp", "salmon", "tuna",
}

IRREGULAR_PLURALS = {
    "potato": "potatoes",
    "tomato": "tomatoes",
    "mango": "mangoes",
    "avocado": "avocados",
    "leaf": "leaves",
    "loaf": "loaves",
}

SUBJECTS = ["a man", "a woman", "a chef", "a cook"]
SCENES = ["in the kitchen", "at a market stall", "in a home kitchen", "in a cafe kitchen"]

# Minimal meaning-preserving rewordings of each scene: a one-token substitution
# that leaves aspect and event untouched. This gives a *tight* noise floor. The
# looser control (fronting the whole scene phrase) moves many tokens and would
# inflate the floor, which would bias the analysis toward concluding that aspect
# is invisible -- i.e. toward the expected result. The tight floor is the
# conservative one, so it is the primary reference.
SCENE_SYNONYMS = {
    "in the kitchen": "inside the kitchen",
    "at a market stall": "at a market booth",
    "in a home kitchen": "in a domestic kitchen",
    "in a cafe kitchen": "in a bistro kitchen",
}


def pluralize(noun: str) -> str:
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def indefinite(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def is_usable_object(noun: str) -> bool:
    """Count nouns only, so the bare-plural atelic contrast is well formed."""
    return noun not in MASS_OR_GENERIC and " " not in noun
