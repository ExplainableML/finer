SYSTEM_MSG_PHRASE = (
    "You are an information extraction assistant.\n"
    "Select ONE main object from the caption that has at least one described attribute.\n"
    "Extract attribute phrases ONLY if they are explicitly stated and are used to describe the chosen main object—never infer or guess.\n"
    "Then compose a SINGLE noun phrase describing that object with the extracted attribute phrases.\n"
    "Use ONLY evidence from the caption. Never invent attributes.\n"
    "Allowed attribute types:\n"
    "• Appearance, color, pattern, size, shape, material, markings/printed text/numbers, "
    "condition/state, orientation/pose, and other visible features that describe the main object.\n"
    "• Accessories physically attached to the main object (e.g., a collar on a dog) count as attributes; unrelated co-occurring objects do not.\n"
    "Constraints:\n"
    "• Do NOT include spatial relations to other main objects (e.g., 'to the left of the bus').\n"
    "• Do NOT include actions involving other main objects (e.g., 'holding a cup').\n"
    "• The extracted attributes must clearly describe the chosen main object. NEVER invent attributes. NEVER extract attributes for the other objects.\n"
    "• Extract 1–5 attributes for the chosen main object. If fewer than five are stated, extract fewer. If only one is present, use that one. NEVER invent attributes.\n"
    "• If necessary, rewrite the original attribute phrase to either a plain adjective phrase (e.g., 'red', 'shiny metal', 'long-tailed'), "
    "or a 'with ...' phrase (e.g., 'with yellow eyes', 'with its nose pointing to the left', 'with the text \"SALE\"'). The rewriting should not change the original meaning.\n"
    "• Connect multiple 'with ...' phrases smoothly using commas and 'and' (e.g., 'with yellow eyes, with a striped tail, and with a scar').\n"
    "Return EXACTLY one line:\n"
    "PHRASE=<your noun phrase with attributes>\n"
    "No trailing period. No extra text."
)

FEW_SHOT_ATTRIBUTES = [
    (
        "At the park entrance, a worn leather backpack with frayed straps lies on a bench. "
        "A red umbrella leans against the bench, and a silver water bottle sits to the right. "
        "A small stitched patch on the backpack reads \"MT-42\".",
        "PHRASE=a worn leather backpack with frayed straps and with the text \"MT-42\""
    ),
    (
        "A yellow sports car idles at the start line. "
        "Black racing stripes run over the hood, and the number 27 is painted on the door. "
        "The car points slightly to the left while orange cones line the track.",
        "PHRASE=a yellow sports car with black racing stripes, with the number 27, and with its nose pointing to the left"
    ),
]

SYSTEM_MSG_NEGATTR = (
    "You are a negative attribute editor.\n"
    "You will receive an ATTRIBUTE PHRASE: a single noun phrase describing one main object with 1–5 attributes.\n"
    "Each attribute is one replaceable unit: either (a) a pre-nominal adjective group (e.g., 'long-sleeved red') "
    "or (b) one entire 'with ...' clause or other forms of clause separated by commas or 'and'.\n"
    "\n"
    "Task:\n"
    "Pick exactly ONE attribute unit at random and replace it with a distinctly different NEGATIVE attribute.\n"
    "\n"
    "Randomness:\n"
    "• Replace the attribute unit at random position. Both pre-nominal adjective group or 'with ...' clause should have a chance to be replaced. \n"
    "\n"
    "Definitions & scope of attributes that can be changed:\n"
    "• Appearance, color, pattern, size, shape, material, texture, markings/printed text/numbers, "
    "condition/state, orientation/pose, and accessories physically attached to the main object.\n"
    "\n"
    "Constraints for the replacement:\n"
    "• Keep the object head and all other attributes unchanged; preserve order, punctuation, articles, quotes, units, and capitalization.\n"
    "• Keep the grammatical shape of the replaced unit (adjective group stays an adjective group; a 'with ...' clause stays a 'with ...' clause).\n"
    "• The replacement must be distinctly different from the original and NOT a synonym, near-synonym, or morphological variant of any attribute in the phrase (e.g., red↛crimson; striped↛banded).\n"
    "• Do not duplicate any existing attribute already present in the phrase.\n"
    "• Avoid always changing the same type of attribute; consider changing any types of attributes stated in the definitions above.\n"
    "\n"
    "Self-check before answering (must be satisfied):\n"
    "• Exactly one attribute unit differs; all other attribute units are identical.\n"
    "\n"
    "Output EXACTLY one line:\n"
    "PHRASE=<the rewritten noun phrase>\n"
    "No extra text. No quotes. No trailing period."
)


FEW_SHOT_NEGATTR = [
    (
        'PHRASE=a soccer player with a vibrant, long-sleeved red jersey with a collar, with yellow text reading "Standard Chartered", with gold-colored logos, and with short, dark black hair',
        'PHRASE=a soccer player with a vibrant, long-sleeved red jersey with a collar, with pink caption reading "Standard Charity", with gold-colored logos, and with long, dark black hair'
    ),
    (
        'PHRASE=a card with a latency graph, with the word "Latency" labeled, and with the measurement unit "milliseconds" noted below it',
        'PHRASE=a card with a latency graph, with the word "Latency" labeled, and with the measurement unit "milliseconds" displayed above it'
    ),
    (
        'PHRASE=a square, white ceramic plate with a serving of dark tan, very thin noodles, with chopped vegetables including mushrooms, carrots, onions, and peppers, and with sesame seeds scattered throughout',
        'PHRASE=a square, white ceramic plate with a serving of snow white, very thick noodles, with chopped vegetables including mushrooms, carrots, onions, and peppers, and with sesame seeds scattered throughout'
    ),
    (
        'PHRASE=a blue taxi with rectangular plate, with shiny wheels and with transparent windscreen',
        'PHRASE=a yellow taxi with rectangular plate, with shiny wheels and with transparent windscreen'
    ),
    (
        'PHRASE=a lightweight wooden chair with a curved backrest',
        'PHRASE=a lightweight metal chair with a curved backrest'
    ),
]

SYSTEM_MSG_NEGATTR_IDX = (
    "You are a negative attribute editor.\n"
    "Input format:\n"
    "  ATTRIBUTE_INDEX=<1-based index to edit>\n"
    "  PHRASE=<a single noun phrase describing one main object with 1–5 attributes>\n"
    "\n"
    "Definition of attribute units (counting left→right):\n"
    "• Unit 1 = the entire pre-nominal adjective group that modifies the head noun (e.g., 'long-sleeved red'), if present; "
    "  otherwise, the first 'with ...' clause counts as unit 1.\n"
    "• Units 2..N = each full 'with ...' clause (or other forms of attribute clauses) in order of appearance. Clauses separated by commas or 'and' each count as one unit.\n"
    "\n"
    "Task:\n"
    "Edit EXACTLY the attribute unit with the given ATTRIBUTE_INDEX to make it a clearly different (ideally opposite) NEGATIVE attribute.\n"
    "\n"
    "Hard constraints (must follow):\n"
    "• If ATTRIBUTE_INDEX is larger than the number of attribute units you see, edit the LAST unit.\n"
    "• Keep the object head and all other attribute units unchanged; preserve order, punctuation, articles, quotes, units, and capitalization.\n"
    "• Keep the grammatical shape of the replaced unit (adjective group stays an adjective group; a 'with ...' clause stays a 'with ...' clause).\n"
    "• The new attribute must be distinctly different from the original and NOT a synonym/near-synonym/morphological variant of any attribute in the phrase.\n"
    "• Do not duplicate any existing attribute already present in the phrase. Do not simply insert 'not'.\n"
    "\n"
    "Output EXACTLY one line:\n"
    "PHRASE=<the rewritten noun phrase>\n"
    "No extra text. No quotes. No trailing period.\n"
    "Important: The few-shot samples below are for formatting and behavior ONLY. "
    "Do NOT copy their attribute phrases. Only edit the final PHRASE after the few-shot examples."
)

FEW_SHOT_NEGATTR_IDX = [
    (
        'ATTRIBUTE_INDEX=1\nPHRASE=a lightweight wooden chair with a curved backrest',
        'PHRASE=a heavyweight metal chair with a curved backrest'
    ),
    (
        'ATTRIBUTE_INDEX=3\nPHRASE=a blue taxi with a rectangular plate, with shiny wheels, and with a transparent windscreen',
        'PHRASE=a blue taxi with a rectangular plate, with shiny wheels, and with an opaque windscreen'
    ),
    (
        'ATTRIBUTE_INDEX=2\nPHRASE=a square, white ceramic plate with a serving of dark tan, very thin noodles, with chopped vegetables including mushrooms, carrots, onions, and peppers, and with sesame seeds scattered throughout',
        'PHRASE=a square, white ceramic plate with a serving of snow white, very thick noodles, with chopped vegetables including mushrooms, carrots, onions, and peppers, and with sesame seeds scattered throughout'
    ),
    (
        'ATTRIBUTE_INDEX=5\nPHRASE=a clear plastic bottle with a blue cap',
        'PHRASE=a clear plastic bottle with a red cap'
    ),
]

DIFF_SYSTEM_MSG = (
    "You will receive TWO answers about the SAME head object phrase. They share most attributes but differ on ONE.\n"
    "An attribute can either be pre-nominal adjectives (e.g., 'orange', 'metal'), a post-nominal 'with..' clause (e.g. 'with long legs') or any other phrases that describe the head object phrase (e.g. 'showing a...', 'that has...').\n"
    "Goal:\n"
    "1) Find the differing attribute.\n"
    "2) Write TWO MINIMAL FULL-SENTENCE answers that keep each answer’s surface format "
    "(e.g., keep 'Yes,' vs 'No, but', and keep wording like 'can be seen in this image' if present).\n"
    "   • Each output should mention ONLY the head object + the differing attribute.\n"
    "   • If the difference is a pre-nominal adjective, keep ONLY that adjective + the head noun (no extra attributes).\n"
    "   • The head object text must be identical in both outputs.\n"
    "Rules: use ONLY the given answers; do not invent; If you can't find the differing attribute, just repeat the two input sentences; If you find multiple differing attributes, keep them all in your rewritten sentences.\n"
    "Output EXACTLY two lines (no extra text):\n"
    "ACC_PHRASE=<full sentence>\n"
    "REJ_PHRASE=<full sentence>"
    "The few-shot examples below are only for demonstration, DO NOT COPY attribute phrases from them:\n"
)

FEW_SHOT_DIFF = [
        (
            "Example:\n"
            "ACC:\nYes, the image shows a lightweight wooden chair with a curved backrest.\n"
            "REJ:\nYes, the image shows a lightweight metal chair with a curved backrest.",
            "ACC_PHRASE=Yes, the image shows a lightweight wooden chair.\n"
            "REJ_PHRASE=Yes, the image shows a lightweight metal chair."
        ),
        (
            "Example:\n"
            "ACC:\nYes, a striking black bird with iridescent feathers and with a piercing yellow eye is visible in this image.\n"
            "REJ:\nNo, but a striking black bird with iridescent feathers and with a dull brown eye is present in this image.",
            "ACC_PHRASE=Yes, a striking black bird with a piercing yellow eye is visible in this image.\n"
            "REJ_PHRASE=No, but a striking black bird with a dull brown eye is present in this image."
        ),
        (
            "Example:\n"
            "ACC:\nYes, this image shows a black vintage motorcycle with a circular speedometer.\n"
            "REJ:\nNo, but this image shows a white modern motorcycle with a circular speedometer.",
            "ACC_PHRASE=Yes, this image shows a vintage motorcycle with a black finish.\n"
            "REJ_PHRASE=No, but this image shows a modern motorcycle with a white finish."
        ),
    ]



