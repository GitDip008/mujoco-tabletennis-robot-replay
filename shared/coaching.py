import pathlib
import yaml
import json
from openai import OpenAI
from summarizer import run_session, format_summary_for_prompt


# ######################### Stroke label mapping #########################
STROKE_LABEL_MAP = {
    "Stroke Type 1": "Forehand Topspin",
    "Stroke Type 2": "Backhand Drive",
    "Stroke Type 3": "Forehand Smash",
    "No Stroke"    : "No Stroke / Rest",
}


# ######################### System prompt #########################
SYSTEM_PROMPT = """You are an expert table tennis coach with 20 years of \
competitive and training experience. You analyze IMU sensor data collected \
from a player's racket wrist during a training session.

Your role is to:
1. Interpret the session statistics provided.
2. Identify the player's dominant patterns, weaknesses, and inconsistencies.
3. Give 3 specific, actionable coaching recommendations ranked by priority.
4. Use biomechanical language where relevant (wrist snap, elbow lead, \
follow-through, hip rotation, contact point, stroke arc).
5. Keep feedback constructive, specific, and evidence-based from the data.
6. Be concise — no more than 250 words total.

Important context:
- Confidence score reflects how cleanly the IMU signal matches a known \
stroke pattern. Low confidence = inconsistent or poorly executed technique.
- Irregular tempo means the player's stroke rhythm is unstable.
- Stroke distribution imbalance may indicate over-reliance on one stroke.
"""


def build_user_prompt(summary_text: str, subject_id: int) -> str:
    """
    Injects the session summary into a structured user prompt.
    Translates internal class names to real stroke names.
    """
    for internal, real in STROKE_LABEL_MAP.items():
        summary_text = summary_text.replace(internal, real)

    return f"""Analyze the following table tennis training session for Player {subject_id}.

{summary_text}

Based on this data, provide your response in EXACTLY this format — no extra text:

ASSESSMENT: <one sentence overall assessment>

RECOMMENDATIONS:
1. [PRIORITY: HIGH] <recommendation + specific drill>
2. [PRIORITY: MED] <recommendation + specific drill>
3. [PRIORITY: LOW] <recommendation + specific drill>

NEXT SESSION FOCUS: <one sentence>
"""


def get_coaching_feedback(
    summary: dict,
    subject_id: int,
    cfg: dict = None,
) -> dict:
    """
    Sends session summary to local LM Studio (Mistral-7B) and returns
    structured coaching output.

    Args:
        summary    : dict from summarizer.run_session()
        subject_id : player ID
        cfg        : config dict (loads from config.yaml if None)

    Returns:
        {
            "subject_id"      : int,
            "raw_response"    : str,
            "assessment"      : str,
            "recommendations" : list[dict],
            "next_focus"      : str,
            "prompt_tokens"   : int,
            "output_tokens"   : int,
        }
    """
    if cfg is None:
        root = pathlib.Path(__file__).resolve().parent.parent
        with open(root / "config.yaml") as f:
            cfg = yaml.safe_load(f)

    summary_text = format_summary_for_prompt(summary)
    user_prompt  = build_user_prompt(summary_text, subject_id)

    # LM Studio part
    client = OpenAI(
        base_url = cfg["llm"]["base_url"],
        api_key  = "lm-studio",             # required by openai client, value ignored by LM Studio
    )

    response = client.chat.completions.create(
        model=cfg["llm"]["model"],
        max_tokens=cfg["llm"]["max_tokens"],
        temperature=cfg["llm"]["temperature"],
        messages=[
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            },
        ],
    )

    raw_text = response.choices[0].message.content
    parsed   = _parse_response(raw_text)

    return {
        "subject_id"      : subject_id,
        "raw_response"    : raw_text,
        "assessment"      : parsed["assessment"],
        "recommendations" : parsed["recommendations"],
        "next_focus"      : parsed["next_focus"],
        "prompt_tokens"   : response.usage.prompt_tokens,
        "output_tokens"   : response.usage.completion_tokens,
    }


def _parse_response(text: str) -> dict:
    """
    Parses the structured LLM response into individual fields.
    Gracefully handles partial or malformed output.
    """
    result = {
        "assessment"      : "",
        "recommendations" : [],
        "next_focus"      : "",
    }

    for line in text.strip().splitlines():
        line_s = line.strip()

        if line_s.startswith("ASSESSMENT:"):
            result["assessment"] = line_s.replace("ASSESSMENT:", "").strip()

        elif line_s.startswith("NEXT SESSION FOCUS:"):
            result["next_focus"] = line_s.replace("NEXT SESSION FOCUS:", "").strip()

        elif line_s.startswith(("1.", "2.", "3.")) and "[PRIORITY:" in line_s:
            try:
                p_start  = line_s.index("[PRIORITY:") + len("[PRIORITY:")
                p_end    = line_s.index("]", p_start)
                priority = line_s[p_start:p_end].strip()
                rec_text = line_s[p_end + 1:].strip()
                result["recommendations"].append({
                    "rank"    : len(result["recommendations"]) + 1,
                    "priority": priority,
                    "text"    : rec_text,
                })
            except (ValueError, IndexError):
                result["recommendations"].append({
                    "rank"    : len(result["recommendations"]) + 1,
                    "priority": "UNKNOWN",
                    "text"    : line_s,
                })

    # Fallback: if parsing failed, surface the raw text so nothing is lost
    if not result["assessment"] and not result["recommendations"]:
        result["assessment"] = "[Parse failed — see raw_response]"

    return result


def print_coaching_report(feedback: dict):
    """Pretty-print the coaching feedback to terminal."""
    print("\n" + "═" * 55)
    print(f"  COACHING REPORT — Player {feedback['subject_id']}")
    print(f"  Model : Mistral-7B (LM Studio local)")
    print("═" * 55)
    print(f"\nASSESSMENT:\n  {feedback['assessment']}")
    print("\nRECOMMENDATIONS:")
    for rec in feedback["recommendations"]:
        print(f"\n  [{rec['priority']}] #{rec['rank']}")
        print(f"  {rec['text']}")
    print(f"\nNEXT SESSION FOCUS:\n  {feedback['next_focus']}")
    print(f"\n── Token usage: "
          f"{feedback['prompt_tokens']} in / {feedback['output_tokens']} out ──")
    print("═" * 55)
    print(f"\n── Raw response (for debug) ──\n{feedback['raw_response']}\n")


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from inference import StrokePredictor

    root       = pathlib.Path(__file__).resolve().parent.parent
    SUBJECT_ID = 10

    df         = pd.read_csv(root / "data/raw/TTSWING.csv")
    session    = df[df["id"] == SUBJECT_ID].copy()

    predictor  = StrokePredictor.from_subject(subject_id=SUBJECT_ID)
    summary    = run_session(predictor, session)
    feedback   = get_coaching_feedback(summary, subject_id=SUBJECT_ID)

    print_coaching_report(feedback)

    out_path = root / "checkpoints" / f"coaching_subj{SUBJECT_ID:03d}.json"
    with open(out_path, "w") as f:
        json.dump(feedback, f, indent=2)
    print(f"Saved to {out_path}")