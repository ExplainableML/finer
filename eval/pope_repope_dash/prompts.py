# ash_eval/prompts.py

def yesno_prompt(question: str) -> str:
    # for llava_next models, we found that an explicit prompt like this works better
    # therefore, we only use this to evaluate llava_next (base and FINER version), not using it should yield similar results
    return (
        "Answer the following question about the image with a single word: 'yes' or 'no'.\n\n"
        f"Question: {question}\n\n"
        "Answer (yes/no):"
    )


def build_messages(pil_image, model_name, question: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question},
        ],
    }]


def parse_yesno(text: str) -> int:
    """
    Returns:
      1  -> yes
      0  -> no
     -1  -> unknown / unparsable
    """
    s = (text or "").lower().strip()

    yi = s.find("yes")
    ni = s.find("no")

    yes_present = yi != -1
    no_present = ni != -1

    if yes_present and not no_present:
        return 1
    if no_present and not yes_present:
        return 0
    if yes_present and no_present:
        return 1 if yi < ni else 0
    return -1