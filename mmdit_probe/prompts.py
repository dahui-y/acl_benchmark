"""The six (style, content) pairs the probe runs on.

Every style branch carries its OWN object -- sunflower, wave, teapot, ... --
and every content branch a different one. That is deliberate: without a
distinct object in the style branch there is no way to tell "the style came
across" from "the style image came across", and content leakage is the failure
mode that kills attention-swap style transfer in practice.

The six styles are picked from different families (impasto painting, woodblock
print, flat-shaded 3D, graphite drawing, stained glass, pixel art) so a null
result cannot be explained by "we only tried painterly styles". Content targets
are all photographic, so any non-photographic look in the output came from the
injection and not from the content prompt.
"""

PAIRS = [
    ("vangogh",
     "an oil painting of a sunflower in thick impasto brushstrokes, "
     "swirling texture, Van Gogh style",
     "a photograph of a golden retriever sitting on grass"),
    ("ukiyoe",
     "a Ukiyo-e Japanese woodblock print of a breaking wave, flat colour "
     "areas, bold outlines",
     "a photograph of a red sports car on a mountain road"),
    ("lowpoly",
     "a low-poly 3D render of a teapot, flat shading, visible triangular "
     "facets, pastel palette",
     "a photograph of a wooden chair in an empty room"),
    ("pencil",
     "a graphite pencil sketch of a bicycle on textured paper, cross-hatching, "
     "no colour",
     "a photograph of a lighthouse on a cliff at sunset"),
    ("stainedglass",
     "a stained glass window depicting a lion, bold black leading, saturated "
     "translucent colour",
     "a photograph of a bowl of strawberries on a table"),
    ("pixelart",
     "a 16-bit pixel art sprite of a castle, hard aliased edges, limited "
     "palette",
     "a photograph of a city street with pedestrians"),
]

# What object the style branch contains. Only used by the report, to name the
# thing to look for when judging leakage by eye.
LEAK_OBJECT = {
    "vangogh": "sunflower", "ukiyoe": "wave", "lowpoly": "teapot",
    "pencil": "bicycle", "stainedglass": "lion", "pixelart": "castle",
}

# `none` is not a control that gets thrown away: it is the run that produces
# both the content baseline and the style reference, from the same batch and
# the same noise as every swapped condition.
CONDITIONS = ["none", "img", "txt", "both"]
