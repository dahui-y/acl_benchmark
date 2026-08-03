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

# Aspectual class of each verb. Telicity works differently across these, so the
# analysis has to treat class as an item-level factor rather than pooling over it:
#
#   incremental_theme  culmination is defined by the object being used up
#                      ("slice the lime" ends when the lime is sliced)
#   degree_achievement gradable property with no inherent endpoint
#                      ("brown the onion" -- how brown counts as done?)
#   activity           no endpoint at all ("roll the dough")
#
# NOTE: these assignments are the authors' first pass and MUST be checked by a
# second annotator with formal-semantics training before they go in a paper.
# The standard diagnostics are the "in an hour" / "for an hour" adverbial test
# and the entailment from progressive to perfect.
VERB_ASPECTUAL_CLASS = {
    "chopping": "incremental_theme", "mincing": "incremental_theme",
    "shredding": "incremental_theme", "slicing": "incremental_theme",
    "grating": "incremental_theme", "zesting": "incremental_theme",
    "peeling": "incremental_theme", "coating": "incremental_theme",
    "melting": "degree_achievement", "browning": "degree_achievement",
    "crushing": "degree_achievement", "mashing": "degree_achievement",
    "squeezing": "degree_achievement", "whipping": "degree_achievement",
    "blending": "degree_achievement",
    "frying": "activity", "sauteing": "activity", "grilling": "activity",
    "roasting": "activity", "rolling": "activity",
}

# Which object subcategories each action category can plausibly apply to.
#
# Without this the by-verb stratified sampler happily produces "melting a
# scallion", which no model can render sensibly and which would confound the
# aspect measurement with plausibility. OSCBench solved the same problem by
# pairing action categories with compatible object subcategories and then having
# humans check the result; this table is the first half of that, and the review
# sheet is the second.
#
# Conservative on purpose: a pair left out costs coverage, a bad pair left in
# costs an uninterpretable video.
ACTION_OBJECT_COMPATIBILITY = {
    "Cutting": {"Leafy", "Root", "Bulb", "Stem_Stalk", "Fruiting", "Cruciferous",
                "Mushroom", "Citrus", "Berries", "Tropical_Melon", "Pome", "Stone",
                "Meat", "Processed_Meat", "Seafood", "Plant", "Cheeses",
                "Carb_Foods", "Herbs_Spices"},
    "Heating": {"Leafy", "Root", "Bulb", "Stem_Stalk", "Fruiting", "Cruciferous",
                "Mushroom", "Meat", "Processed_Meat", "Seafood", "Plant", "Eggs",
                "Carb_Foods", "Grains", "Nuts_Seeds"},
    "Pressing": {"Root", "Bulb", "Fruiting", "Citrus", "Berries", "Tropical_Melon",
                 "Pome", "Stone", "Plant", "Nuts_Seeds", "Herbs_Spices"},
    "Grating": {"Root", "Bulb", "Fruiting", "Cruciferous", "Citrus", "Cheeses",
                "Nuts_Seeds", "Herbs_Spices"},
    "Mixing": {"Milk_Derivatives", "Eggs", "Sweeteners", "Dessert_Bases",
               "Berries", "Tropical_Melon", "Fats_Oils", "Condiments_Misc"},
    "Coating": {"Meat", "Processed_Meat", "Seafood", "Plant", "Carb_Foods",
                "Dessert_Bases", "Snacks", "Nuts_Seeds", "Fruiting"},
    "Rolling": {"Carb_Foods", "Dessert_Bases", "Grains"},
    "Peeling": {"Root", "Bulb", "Citrus", "Tropical_Melon", "Pome", "Stone",
                "Fruiting", "Seafood", "Eggs", "Nuts_Seeds"},
    "Melting": {"Milk_Derivatives", "Cheeses", "Fats_Oils", "Sweeteners",
                "Dessert_Bases", "Snacks"},
}


def is_compatible(action_category: str, object_sub: str) -> bool:
    return object_sub in ACTION_OBJECT_COMPATIBILITY.get(action_category, set())


# Objects that resist the bare-plural atelic contrast. Items in the taxonomy that
# are mass nouns ("spinach", "meat") or that are already category labels
# ("vegetable", "citrus") are excluded from the main matrix so that the
# telic/atelic contrast stays a clean count-noun alternation.
# Mass nouns and category labels. Both break the design: the telicity axis
# alternates "a lime" / "limes" / "the limes", which needs a count noun, and a
# generic label ("vegetable") is not an object anyone can render.
#
# Compiled by going through the taxonomy subcategory by subcategory rather than
# guessing -- the earlier guessed list let through "chopping a dill" and
# "blending a caramel".
MASS_OR_GENERIC = {
    # category labels, not objects
    "vegetable", "leaf", "meat", "seafood", "fish", "citrus", "berry", "nut",
    "herb", "cheese",
    # leafy greens
    "spinach", "kale", "lettuce", "cabbage",
    # cured / uncountable proteins
    "ham", "bacon", "salmon", "tuna", "beef", "pork", "chicken", "turkey", "tofu",
    # dairy and fats
    "milk", "cream", "yogurt", "mozzarella", "paneer", "parmesan", "mascarpone",
    "butter", "ghee", "margarine", "shortening",
    # grains and doughs
    "rice", "pasta", "bread", "dough", "batter",
    # sweeteners and confection bases
    "sugar", "jaggery", "honey", "caramel", "ganache", "buttercream", "frosting",
    "fondant", "gelatin", "chocolate",
    # herbs and spices (mass in the relevant sense)
    "basil", "cilantro", "parsley", "mint", "thyme", "dill", "rosemary",
    "coriander", "nutmeg", "chilies",
    # condiments and non-food
    "sauce", "ice", "clay",
    # already-plural or awkward forms
    "corn", "ginger", "garlic", "asparagus", "broccoli", "cauliflower", "shrimp",
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
