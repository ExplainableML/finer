SYSTEM_MSG_REPHRASE = (
    "You are an information extraction assistant.\n"
    "Select ONE main object from the caption that clearly participates in at least one relation with another object.\n"
    "Extract relations ONLY if they are explicitly stated in the caption—never infer or guess.\n"
    "Allowed relation types:\n"
    "• Spatial: e.g., 'behind X', 'in front of Y', 'on Z', 'under W', 'next to Q', 'between A and B', 'near C', 'inside D', 'at E'.\n"
    "• Action with a target: verb phrases that take an object, e.g., 'holding a cup', 'biting a bone', 'looking at the door', 'sitting on the chair'.\n"
    "Constraints:\n"
    "• Every relation must involve the chosen object.\n"
    "• Refer to other objects with plain nouns; add attributes only to disambiguate same-named objects.\n"
    "• Use ONLY what the caption states; do NOT invent relations.\n"
    "• List 1–5 relations; if only one is present, output just that one.\n"
    "Compose ONE fluent phrase that starts with the object and then lists the relations.\n"
    "Prefer: 'The <object> is <relation1>, is <relation2>, ... and is <relationN>'.\n"
    "Return EXACTLY one line:\n"
    "PHRASE=<your single phrase with relations>\n"
    "No trailing period. No extra text."
)

FEW_SHOT_REPHRASE = [
    (
        "A warmly lit dining scene shows a table draped in white, red, and pink cloth. "
        "In front, two plates hold juicy hamburgers with bacon, cheese, lettuce, and tomato, "
        "with glasses of wine nearby. The nearer hamburger, set before the photographer, sits on a plate; "
        "the other hamburger is across the table for a companion.",
        "PHRASE=The near hamburger is on a plate, is in front of the other hamburger, and is beside a glass of wine"
    )
]

SYSTEM_MSG_NEGREL = (
    "You are a negative relation editor.\n"
    "Input format:\n"
    "  CLAUSE_INDEX=<1-based index to edit>\n"
    "  PHRASE=The <HEAD> <clause1>, <clause2>, ... and <clauseN>\n"
    "Each clause is a relation expressed as a verb + complement, e.g., "
    "'is on a table', 'are between two cars', 'has a transparent faceplate', "
    "'holds a bottle', 'wears a red jersey', 'faces left', 'shows a temperature above 50 degrees'.\n"
    "\n"
    "Task:\n"
    "Select and edit EXACTLY the clause with the given CLAUSE_INDEX (1-based) to make it a clearly different (ideally opposite) NEGATIVE relation.\n"
    "\n"
    "Style guidance (choose ONE option to edit the selected clause):\n"
    "  (A) If the selected clause encodes a spatial relation via a preposition or comparator "
    "(e.g., in/on/inside/outside/under/over/above/below/behind/in front of/"
    "to the left of/to the right of/between/near/at/surrounding/is surrounded by/on top of/at the bottom of, etc.), replace that spatial term with an "
    "opposite or distinctly different spatial relation (e.g., on→inside, in→out of, left→right, above→below, beside→inside). "
    "  (B) If the clause describes an action of the HEAD, replace this action with one distinctly different or opposite. Change the clause’s main lexical verb "
    "(e.g., holds→drops, wears→removes, shows→hides, opens→closes, runs→stands). You may also adjust adverbs or prepositions if any "
    "('is standing on'→'is running away from', 'is driving slowly to'→'is flying high from'). Preserve tense/number/aspect and auxiliaries "
    "(e.g., 'is holding'→'is dropping', 'has opened'→'has closed').\n"
    "  (C) If the selected clause describes possession or properties of the HEAD "
    "(e.g., has/have…, is/are made of…, shows/displays/reads/contains/wears…), "
    "replace the complement with something clearly different or opposite (e.g., 'contains two plastic bags'→'contains three paper bags').\n"
    "\n"
    "Hard constraints (must follow):\n"
    "• If the CLAUSE_INDEX is larger than the number of clauses you see, edit the LAST clause.\n"
    "• Keep the HEAD EXACTLY as in the input.\n"
    "• Keep ALL other clauses unchanged; preserve separators (commas and the final 'and').\n"
    "• Do NOT reorder clauses.\n"
    "• Edit ONLY the selected clause; do NOT add/remove clauses; the edited clause MUST be distinctly different from the original clause.\n"
    "• Avoid merely inserting 'not'; prefer concrete lexical or complement changes.\n"
    "• The new clause must not duplicate another clause and should remain grammatical (tense/number agreement intact).\n"
    "\n"
    "Output EXACTLY one line:\n"
    "PHRASE=<rewritten phrase>\n"
    "No extra text. No trailing period."
)

FEW_SHOT_NEGREL = [
    (
        "CLAUSE_INDEX=2\nPHRASE=The hamburger is on a plate, is in front of a glass of wine, and is beside napkins",
        "PHRASE=The hamburger is on a plate, is behind a glass of wine, and is beside napkins"
    ),
    (
        "CLAUSE_INDEX=4\nPHRASE=The armor is in front of Pepper Potts, is facing the camera, is on a display stand, and has a transparent faceplate",
        "PHRASE=The armor is in front of Pepper Potts, is facing the camera, is on a display stand, and has an opaque faceplate"
    ),
    (
        "CLAUSE_INDEX=5\nPHRASE=The drone is above a field, is facing north, has four propellers, is carrying a small package, and is moving slowly",
        "PHRASE=The drone is above a field, is facing north, has four propellers, is carrying a small package, and is hovering motionless"
    ),
]


SYSTEM_MSG_CANON_REL = (
    "You will be given ONE relation sentence like:\n"
    "  The <HEAD> is <clause1>, is <clause2>, ... and is/has <clauseN>\n"
    "\n"
    "Goal: convert it into a noun phrase that can slot into 'Can you see ___ in this image?'\n"
    "\n"
    "Make only these minimal edits:\n"
    "• Lowercase the very first letter unless the first word is a proper noun/brand (e.g., IKEA, iPhone, NASA).\n"
    "• Insert the single word 'that' exactly once after the head noun phrase and before the first verb "
    "(e.g., 'the <HEAD> that is <clause1>, is <clause2>, ...'). If a correct 'that' is already there, do not add another.\n"
    "\n"
    "Do NOT change any other words, numbers, or punctuation. Preserve quotes and capitalization elsewhere. "
    "Ensure there are spaces on both sides of 'that' (i.e., ' that '). The spaces should be the same as the spaces among other words.\n"
    "\n"
    "Return exactly the noun phrase on one line — no extra text."
)

# Few-shot examples (no PHRASE=, just raw input -> raw output)
FEW_SHOT_CANON_REL = [
    ("The apple is on a table, is next to a cup, and has a short stem",
     "the apple that is on a table, is next to a cup, and has a short stem"),
    ("IKEA bag is on the floor, is blue, and has long handles",
     "IKEA bag that is on the floor, is blue, and has long handles"),
    ("An orange is under a lamp and is touching a plate",
     "an orange that is under a lamp and is touching a plate"),
]

