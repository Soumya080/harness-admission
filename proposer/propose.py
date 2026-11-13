import json
from typing import Any
from harness.core import Harness

PROPOSER_SYSTEM_PROMPT = """You are an expert prompt engineer. Your job is to improve an LLM agent harness by inspecting failure cases from an evaluation pool.
You will be provided with:
1. The current system prompt (incumbent harness).
2. Several failure cases where the model failed to produce the correct answer.

Generate an edited, improved system prompt that directly addresses these systematic errors without regressing on general behavior.
Respond in JSON format:
{
  "system_prompt": "<improved system prompt>",
  "edit_note": "<short explanation of changes made>"
}
"""


def build_proposer_prompt(incumbent: Harness, failures: list[dict[str, Any]]) -> list[dict[str, str]]:
    failure_descriptions = []
    for idx, f in enumerate(failures[:5], 1):
        q = f.get("question", "")
        gold = f.get("gold", "")
        gen = f.get("generation", "")
        failure_descriptions.append(
            f"--- Failure Example {idx} ---\nQuestion: {q}\nGold Answer: {gold}\nModel Output:\n{gen}\n"
        )

    user_content = (
        f"### Current Incumbent System Prompt:\n{incumbent.system_prompt}\n\n"
        f"### Sample Failures on Proposal Pool:\n"
        + "\n".join(failure_descriptions)
        + "\nPropose a refined system prompt that fixes these reasoning failures."
    )

    return [
        {"role": "system", "content": PROPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_proposer_response(raw_text: str, incumbent: Harness) -> Harness:
    try:
        # Extract json block if surrounded by markdown code fences
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[-1].split("```")[0].strip()
        data = json.loads(cleaned)
        new_sp = data.get("system_prompt", incumbent.system_prompt)
        note = data.get("edit_note", "Prompt refinement based on failure cases")
    except Exception:
        # Fallback if json parsing fails
        new_sp = raw_text.strip()
        note = "Raw proposer prompt proposal"

    return Harness(
        system_prompt=new_sp,
        parent=incumbent.id,
        tag="proposer_edit",
        note=note,
    )
