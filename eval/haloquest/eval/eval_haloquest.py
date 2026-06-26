#!/usr/bin/env python3
"""
Evaluation script for the HaloQuest dataset.
Source: The code is adapted from the official HaloQuest Colab notebook.
https://github.com/google/haloquest/blob/main/HaloQuest_Colab.ipynb
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import traceback
from tqdm import tqdm

# Kept imports for compatibility / class definition
import langfun as lf          # we won't use lf.query to avoid structured templating issues
import pyglove as pg

# --- Use Google's official SDK directly
import google.generativeai as genai

# ---- Environment / preflight debug info (prints only) -----------------
print("[DEBUG] Python:", sys.version)
try:
    import google
    print("[DEBUG] google package version:", getattr(google, "__version__", "n/a"))
except Exception:
    print("[DEBUG] google package not importable (ok for some installs).")

try:
    import langfun as _lf
    print("[DEBUG] langfun version:", getattr(_lf, "__version__", "n/a"))
except Exception:
    print("[DEBUG] langfun version unknown.")

api_key = os.environ.get("GOOGLE_API_KEY")
print("[DEBUG] GOOGLE_API_KEY present:", bool(api_key))
if api_key is None:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

# ----------------------------------------------------------------------
# Configure Gemini SDK
genai.configure(api_key=api_key)

def _strip_models_prefix(name: str) -> str:
    return name.split("/", 1)[-1] if "/" in name else name

def _list_supported_models():
    try:
        models = list(genai.list_models())
        out = []
        for m in models:
            out.append({
                "name": _strip_models_prefix(getattr(m, "name", "")),
                "methods": set(getattr(m, "supported_generation_methods", []) or [])
            })
        print("[DEBUG] Available models (name → methods):",
              [(m["name"], sorted(m["methods"])) for m in out[:10]], "...")
        return out
    except Exception as e:
        print("[WARN] list_models failed; proceeding with direct probe. Reason:", repr(e))
        return []

def _pick_supported_model():
    # Manual override
    env_choice = os.environ.get("HALOQUEST_GEMINI_MODEL")
    if env_choice:
        print(f"[DEBUG] Using model from env HALOQUEST_GEMINI_MODEL={env_choice}")
        return env_choice

    # Preferred order
    preferred = ["gemini-2.0-flash", "gemini-2.5-flash"]

    # Try list_models if available
    models = _list_supported_models()
    if models:
        names_with_gc = {m["name"] for m in models if "generateContent" in m["methods"]}
        for cand in preferred:
            if cand in names_with_gc or ("models/" + cand) in names_with_gc:
                print(f"[DEBUG] Selected model via list_models: {cand}")
                return cand

    # Fallback: probe each preferred model
    for cand in preferred:
        try:
            _probe = genai.GenerativeModel(cand)
            _r = _probe.generate_content("Return OK.")
            _t = (_r.text or "").strip().lower()
            if "ok" in _t:
                print(f"[DEBUG] Selected model via probe: {cand}")
                return cand
        except Exception as e:
            print(f"[DEBUG] Probe failed for {cand}: {repr(e)}")

    # Last resort
    raise RuntimeError("No supported Gemini model found (tried gemini-2.0-flash, gemini-2.5-flash). "
                       "Set HALOQUEST_GEMINI_MODEL to a valid model.")

# Pick and instantiate the model
_GEMINI_MODEL_NAME = _pick_supported_model()
print(f"[DEBUG] Gemini model selected: {_GEMINI_MODEL_NAME}")
_gemini = genai.GenerativeModel(_GEMINI_MODEL_NAME)

# ----------------------------------------------------------------------

def extract_model_answer(pred):
    if isinstance(pred, str):
        return pred
    if isinstance(pred, dict):
        for k in ["text"]:  # extend here if your writer uses another key
            if k in pred and isinstance(pred[k], str):
                return pred[k]
    return str(pred)

class PredictionEvaluation(pg.Object):
  question: str
  model_response: str
  groundtruth_response: str
  sentence_interpret_model_response_main_point: str
  sentence_interpret_groundtruth_response_main_point: str
  is_prediction_correct: bool

  def to_dict(self):
    return {
      "question": self.question,
      "model_response": self.model_response,
      "groundtruth_response": self.groundtruth_response,
      "sentence_interpret_model_response_main_point": self.sentence_interpret_model_response_main_point,
      "sentence_interpret_groundtruth_response_main_point": self.sentence_interpret_groundtruth_response_main_point,
      "is_prediction_correct": self.is_prediction_correct
    }

def _build_judge_prompt(question: str, model_response: str, groundtruth_response: str) -> str:
    # Mirrors your original policy; asks for strict JSON only.
    return f"""You are a grading assistant.

Your task is to determine if the model response is correct given the question and the groundtruth response.
Interpret the model response with respect to the question.

Rules:
- If the question asks about a detail of an element that is not present in the image, predictions like "yes", "no", or "nothing" are incorrect because they imply the element exists. The correct prediction should acknowledge the element is not present.
- If the prediction says it cannot assist or cannot provide an answer, it is incorrect.
- If the question is about counting, the prediction is correct only if it exactly matches the groundtruth count.

Return STRICT JSON only with the following fields:
{{
  "is_prediction_correct": true or false,
  "sentence_interpret_model_response_main_point": "one short sentence",
  "sentence_interpret_groundtruth_response_main_point": "one short sentence"
}}

No extra text outside JSON.

QUESTION:
{question}

MODEL_RESPONSE:
{model_response}

GROUNDTRUTH_RESPONSE:
{groundtruth_response}
"""

def compute_prediction(inputs):
  """
  Judge with Gemini via google.generativeai, return (bool, PredictionEvaluation).
  """
  _, question, model_response, groundtruth_response = inputs

  # Optional debug preview
  try:
    if os.environ.get("HALOQUEST_DEBUG_ONESHOT") == "1":
      print(f"[DEBUG] compute_prediction() Q≈{(question or '')[:100]!r} "
            f"MR≈{(model_response or '')[:100]!r} GT≈{(groundtruth_response or '')[:100]!r}")
  except Exception:
    pass

  prompt = _build_judge_prompt(question, model_response, groundtruth_response)

  # Call Gemini
  resp = _gemini.generate_content(prompt)
  text = (resp.text or "").strip()

  # Parse strict JSON; be defensive about code fences
  parsed = {}
  try:
    if text.startswith("```"):
      t = text.strip()
      # Try to isolate the JSON block
      l = t.find("{")
      r = t.rfind("}")
      if l != -1 and r != -1:
        text = t[l:r+1]
    parsed = json.loads(text)
  except Exception:
    low = text.lower()
    is_correct = ("true" in low) and ("false" not in low)
    parsed = {
      "is_prediction_correct": is_correct,
      "sentence_interpret_model_response_main_point": "",
      "sentence_interpret_groundtruth_response_main_point": ""
    }

  pe = PredictionEvaluation(
    question=question,
    model_response=model_response,
    groundtruth_response=groundtruth_response,
    sentence_interpret_model_response_main_point=parsed.get(
      "sentence_interpret_model_response_main_point", ""
    ),
    sentence_interpret_groundtruth_response_main_point=parsed.get(
      "sentence_interpret_groundtruth_response_main_point", ""
    ),
    is_prediction_correct=bool(parsed.get("is_prediction_correct", False))
  )

  return pe.is_prediction_correct, pe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--evaluation-result-file", type=str)
    args = parser.parse_args()

    print(f"[DEBUG] question-file: {args.question_file}")
    print(f"[DEBUG] result-file:   {args.result_file}")
    print(f"[DEBUG] eval-out:      {args.evaluation_result_file}")
    print(f"[DEBUG] Gemini model:  {_GEMINI_MODEL_NAME}")

    # Load raw lines to allow diagnostics before dict conversion
    with open(args.question_file, "r") as f:
        q_lines = f.readlines()
    with open(args.result_file, "r") as f:
        p_lines = f.readlines()

    print(f"[DEBUG] #question lines:   {len(q_lines)}")
    print(f"[DEBUG] #prediction lines: {len(p_lines)}")

    # Parse JSON
    questions_list = [json.loads(line) for line in q_lines]
    predictions_list = [json.loads(line) for line in p_lines]

    # Print a few sample IDs and their types for mismatch diagnosis
    def _peek_ids(name, items):
        try:
            samples = items[:3]
            ids = [(it.get("question_id", None), type(it.get("question_id", None)).__name__) for it in samples]
            print(f"[DEBUG] sample {name} IDs (value, type): {ids}")
        except Exception as e:
            print(f"[DEBUG] failed to peek {name} IDs:", repr(e))

    _peek_ids("question", questions_list)
    _peek_ids("prediction", predictions_list)

    # Build dicts exactly as before
    questions = {question['question_id']: question for question in questions_list}
    predictions = {prediction['question_id']: prediction for prediction in predictions_list}

    # Quick counts after dict-ization
    print(f"[DEBUG] unique question IDs:   {len(questions)}")
    print(f"[DEBUG] unique prediction IDs: {len(predictions)}")

    # Optional: print first 2 keys from each mapping
    try:
        print("[DEBUG] first question keys:", list(questions.keys())[:2])
        print("[DEBUG] first prediction keys:", list(predictions.keys())[:2])
    except Exception:
        pass

    eval_results = []
    for question_id, question in tqdm(questions.items()):
        # Extra diagnostics before potential KeyError
        if question_id not in predictions:
            print(f"[ERROR] Missing prediction for question_id={question_id!r} "
                  f"(type={type(question_id).__name__}). "
                  f"Hint: ID type mismatch (str vs int) or incomplete predictions file.")
            raise KeyError(question_id)

        prediction = predictions[question_id]
        answer = question['answer']
        model_resp = extract_model_answer(prediction)

        try:
            result = compute_prediction(inputs=(question_id, question['text'], model_resp, answer))
            correctness, log = result

            log = log.to_dict()
            log["hallucination_type"] = question["hallucination_type"]
            log["image_type"] = question["image_type"]
            log["question_id"] = question_id
            eval_results.append(log)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate question {question_id!r}: {repr(e)}")
            traceback.print_exc()

    with open(args.evaluation_result_file, "w") as f:
        for log in eval_results:
            f.write(json.dumps(log) + "\n")

    # Convert eval_results to DataFrame (unchanged scoring/CI logic)
    df = pd.DataFrame(eval_results)
    print(f"[DEBUG] eval_results rows: {len(df)} | columns: {list(df.columns)}")

    df["score"] = df["is_prediction_correct"].astype(float)

    # Bootstrap configuration
    n_bootstrap = 100
    n_samples = len(df)
    bootstrap_avg_scores = []
    bootstrap_avg_scores_per_type = {ht: [] for ht in df["hallucination_type"].unique()}

    for _ in range(n_bootstrap):
        sampled_df = df.sample(n=n_samples, replace=True)

        # Overall
        bootstrap_avg_scores.append(sampled_df["score"].mean())
        # Per hallucination_type
        for ht, group in sampled_df.groupby("hallucination_type"):
            bootstrap_avg_scores_per_type[ht].append(group["score"].mean())

    # Report confidence intervals
    def ci(data):
        return np.percentile(data, [2.5, 97.5])

    print("Average score: {:.3f}, 95% CI: {:.3f}-{:.3f}".format(
        df["score"].mean(), *ci(bootstrap_avg_scores)))

    print("\nAverage score per hallucination type:")
    for ht in sorted(bootstrap_avg_scores_per_type):
        mean_score = df[df["hallucination_type"] == ht]["score"].mean()
        lb, ub = ci(bootstrap_avg_scores_per_type[ht])
        print(f"  {ht}: {mean_score:.3f}, 95% CI: {lb:.3f}-{ub:.3f}")
