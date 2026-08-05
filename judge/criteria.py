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
definition of what "done" looks like for this verb. That is what lets the
control conditions carry ground truth: `paraphrase_min` must get the same
answers as `prog`, and any judge that disagrees with itself there is measurably
unreliable, with no annotator involved.
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


# Pairs of conditions whose judgments MUST match, and why. This is where the
# ground truth comes from: it is derived from what the sentences mean, not from
# what anyone was asked. A judge that answers these pairs differently is
# unreliable by construction -- but so is a video model that renders them
# differently, so the two have to be separated (see validate.py).
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
