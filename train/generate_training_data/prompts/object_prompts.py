SYSTEM_MSG_OBJECTS = (
    "You are an information extraction assistant.\n"
    "From the caption, select up to FIVE main objects. A main object must have at least one descriptive attribute in the caption "
    "(e.g., color, size, material, possession/with-phrase, appositive, relative clause, or an explicit number).\n"
    "Rules:\n"
    "• Output only object names that appear in the caption and are part of the described scene. Never invent or infer.\n"
    "• Prefer plain object names in the output (omit adjectives), but KEEP explicit numbers/quantifiers if the caption states them "
    "(e.g., 'two cats', 'five chickens', 'a pair of skis').\n"
    "• If multiple mentions share the same name and no explicit number is given, output the plural form (e.g., 'dogs'). \n"
    "• List 1–5 main objects; if only one is present, output just that one.\n"
    "Do not add quantifiers like 'some' unless present in the caption.\n"
    "Format:\n"
    "Return EXACTLY one line:\n"
    "PHRASE=<comma-separated list with 'and' before the last item, e.g., 'a dog, a cat and two birds'>\n"
    "No trailing period. No extra text."
)

FEW_SHOT_OBJECTS = [
    (
        "A bustling street scene with three blue buses at the curb, two red cars in traffic, "
        "a tall glass office building behind them, and a bronze statue with a raised arm in the plaza.",
        "PHRASE=three buses, two cars, an office building and a statue"
    ),
    (
        "On a wooden table, a cast-iron skillet sits beside a blue ceramic vase of sunflowers, "
        "with red apples and green pears piled high in a stainless steel bowl.",
        "PHRASE=a skillet, a vase, apples, pears and a bowl"
    ),
]

SYSTEM_MSG_NEGOBJ = (
    "You are a negative object creator.\n"
    "You will receive a caption, an object list as PHRASE=..., and REPLACE_INDEX=k (1-based).\n"
    "Replace EXACTLY the k-th object in PHRASE with a distinctly different NEGATIVE object.\n"
    "\n"
    "Keep ALL other objects unchanged and preserve their order and punctuation. Keep the same quantifier/determiner "
    "for the replaced slot (e.g., 'two cats' -> 'two bicycles').\n"
    "\n"
    "Constraints for the NEGATIVE object:\n"
    "• It must be distinctly different from the replaced object (not a synonym; not just singular/plural).\n"
    "• It must NOT be a synonym or near-equivalent of ANY object that appears in the caption.\n"
    "• It must NOT appear anywhere in the caption (as a whole word, singular or plural).\n"
    "• Do not modify any other items; do not reorder items; do not add or remove items.\n"
    "\n"
    "Edge cases (must follow):\n"
    "• If REPLACE_INDEX is greater than the number of objects in PHRASE, replace the LAST object.\n"
    "• If REPLACE_INDEX is less than 1, replace the FIRST object.\n"
    "\n"
    "Self-check (must hold):\n"
    "• Same number of items as input; exactly one item (the k-th per the rule above) differs.\n"
    "\n"
    "Output EXACTLY one line:\n"
    "PHRASE=<the new list, same format>\n"
    "No extra text. No quotes. No trailing period."
)

# Few-shots cover different list lengths and positions; each uses REPLACE_INDEX to avoid a last-item bias.
FEW_SHOT_NEGOBJ = [
    (
        # Replace FIRST of three
        "Caption:\nA small yellow school bus is parked next to two red cars in front of a brick office building.\n"
        "PHRASE=a bus, two cars and an office building\n"
        "REPLACE_INDEX=1",
        "PHRASE=a boat, two cars and an office building"
    ),
    (
        "Caption:\nTwo hikers stand on a wooden bridge with their dog beside them over a calm river.\n"
        "PHRASE=two hikers, a dog, a bridge and a river\n"
        "REPLACE_INDEX=2",
        "PHRASE=two hikers, a bicycle, a bridge and a river"
    ),
    (
        "Caption:\nOn a side table there is a bowl of fruit, a ceramic vase, and a brass lamp.\n"
        "PHRASE=a bowl of fruit, a vase and a lamp\n"
        "REPLACE_INDEX=3",
        "PHRASE=a bowl of fruit, a vase and a clock"
    ),
    (
        "Caption:\nA dashboard card labeled 'Latency' shows a graph with a menu icon and a legend for several lines.\n"
        "PHRASE=a card, a graph, a menu icon, a legend and lines\n"
        "REPLACE_INDEX=3",
        "PHRASE=a card, a graph, a slider, a legend and lines"
    ),
    (
        "Caption:\nA booking screen shows a highlighted 12:05–12:30 time slot, a search box, an entry for Animal Kingdom Villas, a restaurant icon, and a market link.\n"
        "PHRASE=a time slot, a box, an entry, a restaurant and a market\n"
        "REPLACE_INDEX=4",
        "PHRASE=a time slot, a box, an entry, a ticket counter and a market"
    ),
]

CATEGORIES = ["natural_image", "chart_graph", "screenshot_ui", "document_text"]

SYSTEM_MSG_CATEGORY = (
    "You are a strict classifier. Based ONLY on the caption text, assign EXACTLY ONE label.\n"
    "Allowed labels:\n"
    "  - natural_image\n"
    "  - chart_graph   (includes diagrams, flowcharts, schematics, illustrations)\n"
    "  - screenshot_ui (UI/app/website/IDE/terminal screenshots)\n"
    "  - document_text (scans, printed pages, receipts, invoices, whiteboards, handwriting)\n"
    "\n"
    "Rules:\n"
    "• Return ONLY the label string, nothing else.\n"
    "• If multiple seem plausible, pick the single best one.\n"
    "• If unsure, choose natural_image.\n"
)

FEW_SHOT_CATEGORY = [
    ("A line chart comparing quarterly revenue for three products with a legend and axes labels.",
     "chart_graph"),
    ("A screenshot of a mobile banking app showing the account balance and a transfer button.",
     "screenshot_ui"),
    ("A scanned receipt from a grocery store listing items and totals.",
     "document_text"),
    ("A photo of a dog running on a beach at sunset.",
     "natural_image"),
    ("A flowchart illustrating an authentication pipeline with decision nodes and arrows.",
     "chart_graph"),
]

