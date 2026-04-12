SYSTEM_MSG_REPHRASE = (
    "You are an information extraction assistant.\n"
    "Select one sentence in the caption that clearly contains exactly two objects and an explicit relation between them.\n"
    "Allowed relation types (explicit only):\n"
    "• Spatial relation: e.g., 'behind', 'in front of', 'on', 'under', 'next to', 'between', 'near', 'inside', 'at' or other prepositions or comparators.\n"
    "• Action relation: verb phrases that imply actions between the two objects, e.g., 'holding', 'biting', 'looking', 'sitting on' or other verbal phrases.\n"
    "For the selected sentence, rewrite it as one coherent sentence that also includes attributes for each of the two objects (only if those attributes are explicitly stated).\n"
    "Allowed attribute types:\n"
    "• Appearance, color, pattern, size, shape, material, markings/printed text/numbers, condition/state, orientation/pose, and other visible features that describe each object.\n"
    "• Accessories physically attached to an object count as attributes; unrelated co-occurring objects do not.\n"
    "Constraints:\n"
    "• Do not invent or infer relations or attributes; ONLY include the relations or attributes that are explicitly stated in the caption.\n"
    "• The relation must clearly connect the two objects, while the attributes of each object must indeed describe that object.\n"
    "• Use exactly one relation to connect the two objects (e.g., 'is next to', 'is on', 'is looking at').\n"
    "• If necessary, rewrite the original attribute phrase to either a plain adjective phrase (e.g., 'red', 'shiny metal', 'long-tailed'), "
    "or a 'with ...' phrase (e.g., 'with yellow eyes', 'with its nose pointing to the left', 'with the text \"SALE\"'). The rewriting should not change the original meaning.\n"
    "• Connect multiple 'with ...' phrases (or other types of attribute phrases) smoothly using commas and 'and' (e.g., 'with yellow eyes, with a striped tail, and with a scar').\n"
    "• Include at most five attributes for EACH object. If fewer than five are clearly stated, include them all. If more than five are explicitly stated for each object, keep the five most informative and drop the rest.\n"
    "If no single sentence in the caption clearly mentions two objects with an explicit relation, output the fallback:\n"
    "SENTENCE=SKIP\n"
    "Otherwise, compose only one sentence and return EXACTLY one line:\n"
    "SENTENCE=<sentence>\n"
)


FEW_SHOT_REPHRASE = [
    (
        "At the park entrance, a worn leather backpack with frayed straps lies on a bench. "
        "A red umbrella leans against the bench, and a silver water bottle sits to the right. "
        "A small stitched patch is on the backpack.",
        "SENTENCE=a worn leather backpack with frayed straps and with a small stitched patch is on a bench"
    ),
    (
        "A yellow sports car idles at the white start line. The yellow sports car is facing to the left. The start line has a black-and-white flag."
        "Black racing stripes run over the hood, and the number 27 is painted on the door with dark windows. "
        "The car points slightly to the left while orange cones line the track.",
        "SENTENCE=a yellow sports car with black racing stripes, with its body facing to the left and with the number 27 is at a white start line with a black-and-white flag."
    ),
]

SYSTEM_MSG_WHQA_A = (
    "You create one WH-style QA pair from ONE sentence describing two main objects and their explicit relation, "
    "optionally with attributes. The sentence has the logical structure:\n"
    "[obj_a] [attr_a…] [rel] [obj_b] [attr_b…].\n"
    "\n"
    "Your task (A-mode):\n"
    "• Choose [obj_a][attr_a…] as the exact answer span.\n"
    "• Write ONE natural WH question whose answer is exactly that span.\n"
    "• In the QUESTION, preserve as much of [rel][obj_b][attr_b…] as natural, quoted verbatim when it fits, "
    "  and DO NOT repeat or paraphrase [obj_a][attr_a…] inside the question.\n"
    "• Be fluent and grammatical; do not invent details.\n"
    "Output EXACTLY one line:\n"
    "Q=<your question> || A=<the exact substring answer>\n"
    "No extra text."
)

SYSTEM_MSG_WHQA_B = (
    "You create one WH-style QA pair from ONE sentence describing two main objects and their explicit relation, "
    "optionally with attributes. The sentence has the logical structure:\n"
    "[obj_a] [attr_a…] [rel] [obj_b] [attr_b…].\n"
    "\n"
    "Your task (B-mode):\n"
    "• Choose [obj_b][attr_b…] as the exact answer span.\n"
    "• Write ONE natural WH question whose answer is exactly that span.\n"
    "• In the QUESTION, preserve as much of [obj_a][attr_a…] and [rel] as natural, quoted verbatim when it fits, "
    "  and DO NOT repeat or paraphrase [obj_b][attr_b…] inside the question.\n"
    "• Be fluent and grammatical; do not invent details.\n"
    "Output EXACTLY one line:\n"
    "Q=<your question> || A=<the exact substring answer>\n"
    "No extra text."
)

# ----------------------------- Few-shots (A vs B) -----------------------------

FEW_SHOT_WHQA_A = [
    (
        'a red ceramic mug with a chipped rim, with a small coffee stain on the side, and with a curved handle is on a wooden table with visible grain and scattered crumbs',
        'Q=What is on a wooden table with visible grain and scattered crumbs? || A=a red ceramic mug with a chipped rim, with a small coffee stain on the side, and with a curved handle'
    ),
    (
        'a brown dog with a blue collar, with a white chest patch, and with short fur is in front of a white sofa with patterned cushions and wooden legs',
        'Q=What is in front of a white sofa with patterned cushions and wooden legs? || A=a brown dog with a blue collar, with a white chest patch, and with short fur'
    ),
    (
        'a black DSLR camera with a wide lens, with a rubberized grip, and with a neck strap is next to a gray laptop with a backlit keyboard and a trackpad',
        'Q=What is next to a gray laptop with a backlit keyboard and a trackpad? || A=a black DSLR camera with a wide lens, with a rubberized grip, and with a neck strap'
    ),
]

FEW_SHOT_WHQA_B = [
    (
        'a blue bicycle with a rear rack, with a bell, and with mudguards is leaning against a brick wall with faded paint and chipped mortar',
        'Q=What is the blue bicycle with a rear rack, with a bell, and with mudguards leaning against? || A=a brick wall with faded paint and chipped mortar'
    ),
    (
        'a bowl of ripe strawberries in a white ceramic dish, with a silver spoon resting on the rim, is under a window with sheer curtains and a wooden frame',
        'Q=What is the bowl of ripe strawberries in a white ceramic dish, with a silver spoon resting on the rim, under? || A=a window with sheer curtains and a wooden frame'
    ),
    (
        'a stainless steel water bottle with a dented side and with a black screw cap is beside a hiking backpack with gray straps, with side mesh pockets, and with a red carabiner',
        'Q=What is the stainless steel water bottle with a dented side and with a black screw cap beside? || A=a hiking backpack with gray straps, with side mesh pockets, and with a red carabiner'
    ),
]

SYSTEM_MSG_NEG_WHQA_ATTR = (
    "You will convert a POSITIVE wh-question into a counterfactual, NEGATIVE wh-question + answer by replacing EXACTLY ONE "
    "ATTRIBUTE CLAUSE that describes the main object mentioned in the question.\n"
    "\n"
    "DEFINITIONS (apply to the input question):\n"
    "• Main object: the plain head noun phrase that the attributes modify (e.g., 'a mug', 'the DSLR camera'). If multiple objects present in the question, pick the one with more attributes as the main object.\n"
    "• Attribute clause: a modifier that directly describes the main object. It can be\n"
    "  – pre-nominal adjectives (color, material, pattern, size, shape, quantity), e.g., 'red', 'ceramic', 'wide'.\n"
    "  – post-nominal phrases (e.g., 'with …', 'featuring …', 'bearing …', 'labeled \"…\"', participial phrases like 'wearing …').\n"
    "  – other short descriptors attached to the object (texture, condition/state, orientation/pose, printed text/numbers).\n"
    "• Relation clause: the words expressing spatial or action relations that position the main object relative to something else, "
    "  e.g., 'on', 'under', 'next to', 'in front of', 'behind', 'to the left of', 'below', 'above', or light-verb forms like "
    "  'is on', 'is next to', 'is holding', 'is below'.\n"
    "\n"
    "To help you better identify the attribute clauses, the input questions are usually in the following forms:\n"
    "  – WH + [main object + attribute clauses] + [relation clause]?\n"
    "  – WH + [relation clause] + [main object + attribute clauses]?\n"
    "Note that the attribute clauses can either be pre-nominal (before the main object) or post-nominal (after the main object).\n"
    "\n"
    "EDIT RULES:\n"
    "1) Identify all attribute clauses attached to the main object you pick.\n"
    "2) Randomly choose ONE attribute clause (denoted as [original attribute]) and replace its content with a CLEARLY DIFFERENT or even OPPOSITE attribute clause (denoted as [new attribute]).\n"
    "   • You may change multiple adjectives INSIDE [original attribute] to increase contrast.\n"
    "   • Do NOT add, remove, or reorder other attribute clauses—only replace the contents of the chosen [original attribute clause].\n"
    "3) Keep everything else unchanged:\n"
    "   • Do NOT change the main object.\n"
    "   • Do NOT change the relation clause.\n"
    "4) If the question truly has no attribute clauses for the main object, output exactly: SKIP\n"
    "\n"
    "RANDOMNESS:\n"
    "You MUST choose one attribute clause at random position. Both the attribute clauses before or after the main object should have a chance to be chosen.\n"
    "\n"
    "ANSWER FORMAT (pick what fits; ensure correct number agreement and echo the original attribute verbatim):\n"
    "• The [main object] is not [new attribute], but it is [original attribute].\n"
    "• The [main object] does not have [new attribute], but it has [original attribute].\n"
    "• The [main object] contains no [new attribute], but it has/contains [original attribute].\n"
    "If none fits perfectly, write a brief, natural denial that clearly states the object lacks the [new attribute] and has the [original attribute].\n"
    "\n"
    "OUTPUT:\n"
    "Return EXACTLY ONE line:\n"
    "Q=<negative question> || A=<negative answer>\n"
    "No extra text."
)

FEW_SHOT_NEG_WHQA_ATTR = [
    (
        "What is on a wooden table next to a red mug with a chipped rim and with a curved ceramic handle?",
        "Q=What is on a wooden table next to a red mug with a chipped rim and with a straight metal handle? || "
        "A=The mug does not have a straight metal handle, but it has curved ceramic handle."
    ),
    (
        "What is next to a black DSLR camera with a wide lens, with a rubberized grip, and with a neck strap?",
        "Q=What is next to a white DSLR camera with a wide lens, with a rubberized grip, and with a neck strap? || "
        "A=The DSLR camera is not white, but it is black."
    ),
    (
        "What is below a blue banner with white text and with a thin border?",
        "Q=What is below a blue banner with yellow text and with a thin border? || "
        "A=The banner does not have yellow text, but it has white text."
    ),
    (
        "What is to the left of the tree?",
        "SKIP"
    ),
]


