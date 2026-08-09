"""
action_items.py

Step 6 (Action Items part) of the Corporate Meeting Intelligence Pipeline.
---------------------------------------------------------------------------
Takes the ROLE + IDENTITY annotated transcript produced by Step 5
(step5_identity_role_resolution.py) and extracts structured action items:

    Task | Owner | Deadline | Priority

This is the LLM half of Step 6. Summarization (Executive Summary, Key
Points) is expected to be handled separately by your BART model — this
script does NOT do summarization.

Input: the Step-5 merged output, e.g. "step6_input.json", where each
transcript segment carries "speaker", "role", and "identified_name"
(no stored "speaker_label" - the display label is built on the fly here
from those three fields so there's no redundant/duplicated field to go
stale).

Usage:
    export GROQ_API_KEY="your_key_here"

    python action_items.py --input step6_input.json
    python action_items.py --input step6_input.json --output action_items.json

Requires:
    pip install groq
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List

from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"

# Folder action item results get saved into automatically. Created on
# first use if it doesn't exist yet - never needs to be made by hand.
ACTION_ITEMS_OUTPUT_FOLDER = "action_items_outputs"

SYSTEM_PROMPT = """You are an AI specialized in corporate meeting analysis.
You will receive a ROLE- AND IDENTITY-ANNOTATED meeting transcript. Each
speaker turn is labeled like "Speaker 3 (Software Engineer, Deba):" or
"Speaker 8 (DevOps Engineer):" if no name was resolved for that speaker.
Trust these labels; do not re-infer or change anyone's role or identity.

YOUR ONLY TASK
Extract concrete ACTION ITEMS from the transcript — commitments, assigned
tasks, and follow-ups that someone is responsible for doing after the
meeting.

WHAT COUNTS AS AN ACTION ITEM
- A speaker commits to doing something ("I'll send the updated schema",
  "I'll post the PR in Teams").
- A speaker is directly assigned something by someone else ("Rahul, can
  you share an update", "please prepare the final presentation").
- A concrete next step is agreed on for the group ("we'll take that in
  the next sprint" — capture as an item, owner "Unassigned" if no single
  person is named).

WHAT DOES NOT COUNT
- General discussion, opinions, or status updates with no forward-looking
  commitment ("the campaign drove a 12% increase" is NOT an action item).
- Questions that are answered and resolved within the meeting itself with
  no follow-up work created.
- Small talk, acknowledgements ("got it", "sounds good", "thanks").

FOR EACH ACTION ITEM PROVIDE
1. "task"        - a short, clear description of WHAT needs to be done, in
                   your own words (do not quote the transcript verbatim).
2. "owner"       - WHO is responsible. Use the exact "Speaker N (Role)"
                   or "Speaker N (Role, Name)" label from the transcript.
                   If genuinely unclear or a group commitment with no
                   single owner, use "Unassigned".
3. "owner_name"  - the real first name for the owner, taken directly from
                   the "(Role, Name)" portion of that Speaker's label if
                   present. If the label has no name (just a role), use
                   null - do not guess a name yourself.
4. "deadline"    - WHEN it's due, using the speaker's own words/timeframe
                   (e.g. "before Friday", "within an hour", "tomorrow
                   evening"). If no timeframe was mentioned, use
                   "Not specified". Do NOT invent or infer a calendar date
                   that wasn't stated.
5. "priority"    - "High", "Medium", or "Low", inferred from urgency
                   language and context:
                     High   = blocking other work, explicit urgency, tied
                              to a deployment/deadline under discussion
                     Medium = a normal task with no urgency signal
                              (default if unclear)
                     Low    = explicitly minor, optional, or "nice to have"
6. "evidence"    - a short (under 15 words) paraphrase of the transcript
                   moment that supports this item, in your own words. Do
                   not quote the transcript directly.

RULES
- Do not invent action items that aren't supported by the transcript.
- Do not merge two distinct commitments from different speakers into one
  item, even if they're related (e.g. "push fix" and "test after fix" by
  two different speakers are two separate items with two owners).
- If the same speaker restates the same commitment more than once, only
  include it once.
- Output ONLY valid JSON. No preamble, no explanation, no markdown fences.

Output format:
{
  "action_items": [
    {
      "task": "Push the fix for the duplicate records bug",
      "owner": "Speaker 3 (Software Engineer, Deba)",
      "owner_name": "Deba",
      "deadline": "Within an hour, pending regression tests passing",
      "priority": "High",
      "evidence": "Committed to pushing fix once regression passes"
    },
    {
      "task": "Confirm deployment window has no changes",
      "owner": "Speaker 1 (Host)",
      "owner_name": null,
      "deadline": "Not specified",
      "priority": "Low",
      "evidence": "Confirmed deployment timing unchanged"
    }
  ]
}
"""


def get_next_sequential_path(folder: str, prefix: str, ext: str) -> str:
    """
    Scan `folder` for files named "<prefix>_0001<ext>", "<prefix>_0002<ext>",
    etc., and return the path for the NEXT number in that sequence. Scans
    the folder fresh each call (rather than keeping a counter in memory),
    so numbering correctly continues across separate runs of the script,
    even from a different process or a different day.
    """
    os.makedirs(folder, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(ext)}$")
    numbers = [
        int(m.group(1))
        for f in os.listdir(folder)
        if (m := pattern.match(f))
    ]
    next_num = max(numbers, default=0) + 1
    return os.path.join(folder, f"{prefix}_{next_num:04d}{ext}")


def default_output_path() -> str:
    # Action items are always saved as JSON regardless of the input format.
    return get_next_sequential_path(ACTION_ITEMS_OUTPUT_FOLDER, "action_items", ".json")


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: GROQ_API_KEY environment variable not set.\n"
            "Run: export GROQ_API_KEY='your_key_here'"
        )
    return Groq(api_key=api_key)


def clean_json_response(raw: str) -> str:
    """Strip stray markdown fences/preamble some models add despite instructions."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    first_brace = raw.find("{")
    if first_brace > 0:
        raw = raw[first_brace:]
    return raw


def build_speaker_label(seg: Dict) -> str:
    """
    Build the display label for a segment from its raw fields, since
    Step 5 no longer stores a precomputed "speaker_label" string.
        speaker + role + real name -> "Speaker 3 (Software Engineer, Deba)"
        speaker + role only        -> "Speaker 3 (Software Engineer)"
        speaker + real name only   -> "Speaker 3 (Deba)"
        speaker only               -> "Speaker 3"
    "identified_name" equal to "speaker" itself (e.g. "Speaker 8") means
    no real name was ever resolved - it's a stand-in, not a name - so it
    is NOT appended to the label; only role shows in that case.
    Falls back to a legacy "speaker_label" field if present, for
    compatibility with older Step-5 output that still included it.
    """
    if seg.get("speaker_label"):
        return seg["speaker_label"]

    speaker = seg.get("speaker", "Unknown Speaker")
    role = seg.get("role")
    name = seg.get("identified_name")
    has_real_name = bool(name) and name != speaker

    if role and has_real_name:
        return f"{speaker} ({role}, {name})"
    if role:
        return f"{speaker} ({role})"
    if has_real_name:
        return f"{speaker} ({name})"
    return speaker


def json_segments_to_labeled_text(segments: List[Dict]) -> str:
    """
    Convert role+identity-annotated JSON segments into plain text using
    the label built by build_speaker_label(), merging consecutive
    same-speaker turns.
    """
    lines = []
    last_label = None
    buffer = []

    def flush():
        if last_label is not None and buffer:
            lines.append(f"{last_label}:")
            lines.append(" ".join(buffer))

    for seg in segments:
        label = build_speaker_label(seg)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if label != last_label:
            flush()
            buffer = [text]
            last_label = label
        else:
            buffer.append(text)
    flush()

    return "\n".join(lines)


def load_role_identity_annotated_transcript(path: str) -> str:
    """
    Load the role+identity-annotated transcript produced by Step 5.
    Supports both the .json form (segments with 'role'/'identified_name',
    or a legacy 'speaker_label') and the .txt form (already
    'Speaker N (Role, Name):' lines).
    """
    ext = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext == ".json":
        data = json.loads(raw)
        segments = data.get("transcript", data if isinstance(data, list) else [])
        if not segments:
            sys.exit(f"ERROR: No 'transcript' segments found in {path}")
        return json_segments_to_labeled_text(segments)

    # .txt — expected to already have "Speaker N (Role, Name):" labels
    if "(" not in raw:
        print(
            "WARNING: input .txt doesn't look role/identity-annotated (no '(' found). "
            "Did you pass the Step 5 output?",
            file=sys.stderr,
        )
    return raw


def extract_action_items(transcript: str, client: Groq = None) -> Dict:
    """
    Send a role+identity-annotated transcript to the LLM and return
    extracted action items as a Python dict: {"action_items": [...]}.
    """
    if client is None:
        client = get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,  # deterministic extraction, not creative writing
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    )

    raw_output = response.choices[0].message.content
    cleaned = clean_json_response(raw_output)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON.\nError: {e}\nRaw output:\n{raw_output}"
        )

    if "action_items" not in result:
        raise ValueError(f"Response missing 'action_items' key. Got: {result}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract action items (who/what/when/priority) from a Step-5 annotated transcript."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the Step-5 role+identity annotated transcript (.json or .txt)",
    )
    parser.add_argument(
        "--output",
        help=f"Path to save action items JSON. If omitted, auto-saves into "
             f"./{ACTION_ITEMS_OUTPUT_FOLDER}/ with sequential numbering "
             f"(action_items_0001.json, _0002.json, ...) - no filename "
             f"needed, just run the script again on your next transcript.",
    )
    args = parser.parse_args()

    transcript = load_role_identity_annotated_transcript(args.input)

    client = get_client()
    result = extract_action_items(transcript, client=client)

    output_json = json.dumps(result, indent=2)
    print(output_json)

    output_path = args.output or default_output_path()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"\nSaved action items to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()