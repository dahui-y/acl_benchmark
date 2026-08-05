"""What the judge is asked, and what counts as the target state for each verb.

Two properties make this an automatic evaluator we can defend without human
labels, and both are design choices rather than conveniences.

**The judge never sees the sentence.** It is shown frames, the verb in its bare
form, and the object -- never the aspect-marked prompt that produced the video.
So it cannot infer from "has grated" that it ought to see a finished state. Every
condition of an item is judged against exactly the same question, which is what
makes a difference between conditions attributable to the video rather than to
the judge's expectations. OSCBench's evaluator is shown the prompt; ours cannot
be, because our conditions differ precisely in the prompt.

**The questions are binary and anchored to a stated end state.** Not a 1-5
Likert on eight dimensions -- three yes/no questions against a written
definition of what "done" looks like for this verb. Two Likert scores "ought to
match" is not a checkable claim; two yes/no answers are, which is what lets
conditions carry answers derived from meaning rather than from annotators.

What this establishes, and what it does not: reliability (the same video judged
twice) and specificity (a video judged against a target state it cannot have
reached) are both measurable with no labels at all. Sensitivity is not -- see
KNOWN_NEGATIVE at the bottom of this file.
"""

# The state the object is in once the event culminates. Written to be checkable
# from a single frame -- if a definition needs the whole video to verify, the
# judge cannot answer it consistently and neither could a person.
#
# The five `activity` verbs (frying, sauteing, grilling, roasting, rolling) are
# the interesting case: they have no inherent endpoint, so their "target state"
# is a conventional stopping point rather than a semantic one. That is not a
# flaw in the definitions -- it is the property the aspectual-class variable
# tracks, and the analysis reads these rows against the other two classes.
TARGET_STATES = {
    # incremental theme: culmination is the object being used up
    "chop": "the object has been cut into rough pieces",
    "mince": "the object has been cut into very fine pieces",
    "shred": "the object has been reduced to thin strands",
    "slice": "the object has been cut into flat slices",
    "grate": "the object has been reduced to fine shreds against a grater",
    "zest": "the coloured outer peel has been removed as fine shavings",
    "peel": "the outer skin has been removed from the object",
    "coat": "the surface of the object is covered with a layer of another substance",
    # degree achievement: a gradable property with no inherent endpoint
    "crush": "the object has been broken or flattened by pressure",
    "mash": "the object has been reduced to a soft pulp",
    "squeeze": "the object has been compressed hard enough to express juice",
    "whip": "the object has been beaten until visibly aerated or frothy",
    "blend": "the object has been reduced to a smooth liquid or puree",
    "brown": "the surface of the object has turned brown",
    # activity: no endpoint at all; the state below is conventional
    "fry": "the object has been cooked in hot fat and its surface has browned",
    "saute": "the object has been cooked in a little fat and has softened or browned",
    "grill": "the object has been cooked over direct heat and shows char marks",
    "roast": "the object has been cooked by dry heat and its surface has browned",
    "roll": "the object has been flattened or extended into a sheet",
}

# id -> (what the analysis calls it, the question the judge answers)
QUESTIONS = {
    "action": "Does any frame show the action being performed on the object?",
    "initial": "In the FIRST frame, is the object in its original, unaffected state?",
    "final": "In the LAST frame, has the object reached the target state?",
}

PROMPT = """You are evaluating frames sampled from a short video, in chronological order.

The video was generated from a description involving one action and one object.
You are told the action and the object, but NOT the exact sentence.

  Action: {verb}
  Object: {noun}
  Target state: {target_state}

You are given {n_frames} frames, evenly spaced from the start of the video to \
the end. Frame 1 is the first frame of the video and frame {n_frames} is the last.

Answer three questions independently. For each one, first state the factual \
evidence you see in the frames, then answer.

  action:  {q_action}
           Answer "yes" only if the action is visibly being carried out on the \
object -- a hand or tool actually working on it, or the object visibly changing. \
Answer "no" if the object is merely present, held, displayed, or nearby while \
nothing is done to it.

  initial: {q_initial}
           "yes" means the object appears whole and untouched by this action in \
frame 1. "no" means frame 1 already shows it partly or fully in the target state.

  final:   {q_final}
           Judge frame {n_frames} against the target state above, and only \
against it. An object that has been transformed in some OTHER way has not \
reached this target state.

Answer "unclear" only when the object is not visible enough to judge.

Return strictly this JSON and nothing else:
{{
  "action":  {{"evidence": "...", "answer": "yes" | "no" | "unclear"}},
  "initial": {{"evidence": "...", "answer": "yes" | "no" | "unclear"}},
  "final":   {{"evidence": "...", "answer": "yes" | "no" | "unclear"}}
}}"""


def build_prompt(verb_base, noun, n_frames):
    if verb_base not in TARGET_STATES:
        raise KeyError(f"no target state defined for {verb_base!r}")
    return PROMPT.format(
        verb=verb_base, noun=noun, target_state=TARGET_STATES[verb_base],
        n_frames=n_frames, q_action=QUESTIONS["action"],
        q_initial=QUESTIONS["initial"], q_final=QUESTIONS["final"],
    )


# Pairs of conditions whose judgments MUST match, and why -- derived from what
# the sentences mean rather than from what anyone was asked.
#
# Careful about what this measures. These are different videos: the pilot showed
# one preposition visibly moving the scene. So when the judge answers them
# differently it may be answering correctly. What this rate measures is
# therefore VIDEO-MODEL INSTABILITY, once judge noise (test-retest) is
# subtracted -- it is not a validity check on the judge. Judge validity comes
# from test-retest (reliability) and from the known negatives below
# (specificity).
MUST_AGREE = [
    ("prog", "paraphrase_min", "one preposition swapped; truth-conditionally identical"),
    ("prog", "filler", "a length-matched clause added; says nothing about the object"),
    ("atelic", "telic_plural", "bare plural vs definite plural; neither bounds the event"),
    ("atelic", "atelic_some", "bare plural vs 'some'; neither bounds the event"),
]

# The condition that must come out differently, on `final`: the video shows a
# different action, so the item's target state cannot have been reached. Without
# this the agreement rates above would be satisfied by a judge that answers
# "yes" to everything.
MUST_DIFFER = [
    ("prog", "other_verb", "final", "a different action cannot reach this target state"),
]

# Known negatives, and the honest limit of what this design can establish.
#
# Ask a video about a target state it cannot have reached -- a grating video
# judged against "cut into very fine pieces" -- and the answer is no, known in
# advance, for every video we already have. That turns specificity into a
# measurement over dozens of cells instead of the single MUST_DIFFER cell, at
# the cost of one extra call per video and no extra generation.
#
# The asymmetry is real and belongs in the limitations: there is no source of
# known POSITIVES here. Nothing in the design certifies that a particular video
# does reach its target state, so sensitivity cannot be established without
# either external labelled video or a manually verified subsample. What follows
# is that the defensible claims are about differences between conditions under a
# judge of known reliability and specificity -- not about absolute rates.
KNOWN_NEGATIVE = (
    "judge each video against the OTHER verb's target state; the answer is "
    "no by construction"
)
