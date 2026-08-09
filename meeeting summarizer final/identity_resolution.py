"""
identity_resolution.py

Step 5 (Identity part) of the Corporate Meeting Intelligence Pipeline.
-----------------------------------------------------------------------
Takes a speaker-labeled transcript (Step 4 output, optionally already
ROLE-annotated by role_resolution.py) and resolves each Speaker ID to a
real first name, where the transcript gives evidence for one.

This module does ONLY identity resolution. It does not:
  - rewrite/summarize the transcript
  - generate action items
  - infer or change anyone's role (it preserves "role"/"speaker_label"
    fields if they're already present from role_resolution.py)

HYBRID APPROACH (per project discussion):
  1. SELF-INTRODUCTION PASS (strong signal, highest confidence)
     Looks for a speaker naming themselves near the start of their own
     turn, e.g. "Hi, this is Rahul", "Nitya here", "I'm Aman from design".
     If a meeting adopts a norm of everyone introducing themselves (and
     late joiners doing the same when they join), this pass alone
     resolves most speakers with High confidence.
  2. VOCATIVE / ADDRESS PASS (fallback, same logic already proven out in
     action_items.py's owner_name field)
     For any Speaker ID the intro pass didn't cover — someone skipped
     their intro, or joined late without announcing themselves — falls
     back to inferring identity from being addressed by name right
     before/after their turn ("Rahul, are you joining?" -> next turn by
     that Speaker ID is likely Rahul; "Thanks, Aman" right after a turn
     -> that speaker is likely Aman).
  Both passes run in a SINGLE LLM call per transcript so the model can
  weigh intro evidence against address evidence together rather than
  the second pass blindly overwriting the first.

LATE JOINERS:
  This module resolves identity from whatever the transcript contains.
  It does NOT control whether a late joiner actually introduces
  themselves — that has to be enforced upstream (e.g. a meeting-bot
  prompt on participant-join, or a team norm). If a late joiner never
  says their name and is never addressed by name either, their Speaker
  ID is left unresolved (name: null) rather than guessed.

Two input formats are supported (auto-detected from file extension),
matching role_resolution.py:

1) PLAIN TEXT (.txt) — "Speaker N:" or already role-annotated
   "Speaker N (Role):" lines.

2) JSON (.json) — timestamped segment list, e.g.:

    {
      "created_at": "15-07-2026 10:00:00",
      "transcript": [
        {"start": 0, "end": 12, "speaker": "Speaker 1", "text": "..."},
        ...
      ]
    }

   If segments already carry "role" / "speaker_label" (i.e. this file is
   the output of role_resolution.py), those fields are preserved and
   "speaker_label" is upgraded to include the name, e.g.:
      "Speaker 3 (Software Engineer)" -> "Speaker 3 (Software Engineer, Deba)"

Usage:
    export GROQ_API_KEY="your_key_here"

    python identity_resolution.py --input sample_meeting_new_transcript.json
    python identity_resolution.py --input step6_input.json --output identities.json

    # Save the annotated transcript (ready for Step 6) to a specific path:
    python identity_resolution.py --input step6_input.json --annotated-output step6_input_with_identity.json

Requires:
    pip install groq
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an AI specialized in corporate meeting analysis.
You will receive a speaker-labeled transcript. Speakers are labeled either
as "Speaker N:" or "Speaker N (Role):" (role already resolved in an
earlier step, if present — do not touch or re-infer roles).

YOUR ONLY TASK
Resolve each Speaker ID to a real first name, where the transcript gives
clear evidence for one. Do not resolve roles, tasks, or anything else.

USE TWO KINDS OF EVIDENCE, IN THIS PRIORITY ORDER:

1. SELF-INTRODUCTION (strongest signal)
   The speaker names themselves, typically near the start of their own
   turn or the meeting. Examples of the PATTERN (not exhaustive):
     - stating their own name directly ("this is X", "X here")
     - naming themselves while explaining a late arrival or connection
       issue
   If a speaker clearly self-identifies at any point in the transcript,
   that is High confidence regardless of when it happens.

2. VOCATIVE / ADDRESS (fallback, use only if no self-introduction exists
   for that Speaker ID)
   Valid patterns:
     - Someone addresses a person by name right before that Speaker ID's
       next turn, and the reply fits (answers the question / acknowledges
       being called on) - e.g. "Rahul, are you joining?" followed by that
       Speaker ID saying "Yeah, I'm here."
     - A speaker is thanked by name immediately after their turn - e.g.
       Speaker X speaks, then someone replies "Thanks, Aman" - Speaker X
       is likely Aman.
     - A speaker is called out by name and responds in a way consistent
       with being that person (e.g. "you're on mute, Priya" followed by
       that Speaker ID un-muting and continuing their point).
   Do NOT use a name just because it's mentioned in passing about a
   topic (e.g. "because Rahul requested it" does NOT make the current
   speaker Rahul - Rahul is being talked ABOUT, not addressed). Also do
   not assume a name mentioned as being "on mute" or "joining" belongs to
   the CURRENT speaker unless the next/adjacent turn from that Speaker ID
   is consistent with it being a reply from that named person.

CONFIDENCE:
- High   = self-introduction, OR two or more independent address cues
           pointing to the same name for the same Speaker ID
- Medium = exactly one clear address cue and no contradicting evidence
- Low    = weak/indirect address evidence only
If there is no self-introduction and no valid address evidence at all
for a Speaker ID, set "name" to null and confidence to "Low" - do NOT
guess.

CONSISTENCY RULE:
- If two different pieces of evidence point to two different names for
  the SAME Speaker ID, prefer the self-introduction if one exists;
  otherwise prefer whichever address evidence is stronger (more direct,
  more immediate) and note the conflict in "evidence".
- A name should not be assigned to two different Speaker IDs unless the
  transcript truly supports two different people sharing that name (rare
  - treat this as a signal to double check, and lower confidence to Low
  if genuinely ambiguous).

RULES
- Include an entry for EVERY unique Speaker ID present in the input,
  even those with no resolvable name (use "name": null, confidence
  "Low", evidence explaining there was no self-intro or address cue).
- Do not invent names not supported by the transcript.
- Output ONLY valid JSON. No preamble, no explanation, no markdown fences.

Output format:
{
  "speakers": [
    {
      "speaker": "Speaker 6",
      "name": "Rahul",
      "confidence": "High",
      "evidence": [
        "Addressed as Rahul before this speaker's next turn and replied consistently",
        "Thanked as Rahul later in the meeting"
      ]
    },
    {
      "speaker": "Speaker 8",
      "name": null,
      "confidence": "Low",
      "evidence": [
        "No self-introduction or address cue found for this speaker"
      ]
    }
  ]
}
"""


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: GROQ_API_KEY environment variable not set.\n"
            "Run: export GROQ_API_KEY='your_key_here'"
        )
    return Groq(api_key=api_key)


def json_segments_to_text(segments: List[Dict]) -> str:
    """
    Convert a list of timestamped segments into the plain
    "Speaker N:" / "Speaker N (Role):" text format the prompt expects,
    merging consecutive same-speaker turns. Uses "speaker_label" if
    already present (i.e. role-annotated input from role_resolution.py),
    otherwise falls back to plain "Speaker N".
    """
    lines = []
    last_label = None
    buffer = []

    def flush():
        if last_label is not None and buffer:
            lines.append(f"{last_label}:")
            lines.append(" ".join(buffer))

    for seg in segments:
        label = seg.get("speaker_label") or seg.get("speaker", "Unknown Speaker")
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


def load_transcript(path: str) -> str:
    """
    Load a transcript file (.txt or .json) and normalize it to plain
    text for the prompt, matching role_resolution.py's conventions.
    """
    ext = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext == ".json":
        data = json.loads(raw)
        segments = data.get("transcript", data if isinstance(data, list) else [])
        if not segments:
            sys.exit(f"ERROR: No 'transcript' segments found in {path}")
        return json_segments_to_text(segments)

    # default: plain text (.txt or anything else) - already labeled
    return raw


def extract_speaker_ids(transcript: str) -> List[str]:
    """Pull out unique 'Speaker N' labels in order of first appearance."""
    ids = re.findall(r"^(Speaker\s+\d+)\s*(?:\([^)]*\))?\s*:", transcript, flags=re.MULTILINE)
    seen = []
    for s in ids:
        if s not in seen:
            seen.append(s)
    return seen


def clean_json_response(raw: str) -> str:
    """Strip stray markdown fences/preamble some models add despite instructions."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    first_brace = raw.find("{")
    if first_brace > 0:
        raw = raw[first_brace:]
    return raw


def resolve_identities(transcript: str, client: Optional[Groq] = None) -> Dict:
    """
    Send a speaker-labeled transcript to the LLM and get back
    name predictions as a Python dict: {"speakers": [...]}.
    """
    if client is None:
        client = get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,  # deterministic - identity resolution shouldn't be "creative"
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

    expected_ids = set(extract_speaker_ids(transcript))
    returned_ids = {s.get("speaker") for s in result.get("speakers", [])}
    missing = expected_ids - returned_ids
    if missing:
        print(f"WARNING: model omitted speakers: {missing}", file=sys.stderr)

    return result


def build_name_map(identity_result: Dict) -> Dict[str, str]:
    """
    Turn the identity-resolution output into {'Speaker 1': 'Rahul', ...}.

    The LLM itself is instructed to return "name": null when there's no
    self-intro or address evidence at all - that's the honest, ungenerous
    signal we want out of the model, and it's preserved as-is in the raw
    identity_result / --save-raw output.

    Here, when building the map that everything downstream actually uses,
    a null gets replaced with the Speaker ID itself (e.g. "Speaker 8")
    rather than left null. That keeps every downstream field populated
    with *something* displayable, while "name == speaker" is still the
    unambiguous signal (checked by annotate_text_transcript /
    annotate_json_transcript / build_speaker_label) that no real name was
    ever resolved for that speaker - it isn't a guessed name, just the
    same label the speaker already had.
    """
    return {
        s.get("speaker"): (s.get("name") or s.get("speaker"))
        for s in identity_result.get("speakers", [])
        if s.get("speaker")
    }


def annotate_text_transcript(transcript_text: str, name_map: Dict[str, str]) -> str:
    """
    Rewrite speaker labels to include the resolved name, preserving any
    existing role annotation:
        Speaker 6:                      -> Speaker 6 (Rahul):
        Speaker 3 (Software Engineer):  -> Speaker 3 (Software Engineer, Deba):
    Speakers with no resolved name (name_map value == the speaker's own
    ID, per build_name_map's fallback) are left unchanged - no name is
    tacked on, since there's nothing real to add.
    """
    def repl(match):
        speaker = match.group(1)
        role = match.group(2)  # may be None
        name = name_map.get(speaker)
        if not name or name == speaker:
            return match.group(0)
        if role:
            return f"{speaker} ({role}, {name}):"
        return f"{speaker} ({name}):"

    return re.sub(
        r"^(Speaker\s+\d+)\s*(?:\(([^)]*)\))?\s*:",
        repl,
        transcript_text,
        flags=re.MULTILINE,
    )


def annotate_json_transcript(raw_data: dict, name_map: Dict[str, str]) -> dict:
    """
    Add identity info onto each segment of a JSON transcript, preserving
    any existing fields (start/end/speaker/text/role/speaker_label) so
    downstream steps (e.g. action_items.py) still work. Adds/updates:
        "identified_name" -> "Rahul" if resolved, or the speaker's own
                              ID (e.g. "Speaker 8") as a stand-in if no
                              self-intro/address evidence exists at all -
                              never null.
        "speaker_label"    -> upgraded to include the name only when a
                              real name was resolved (identified_name !=
                              speaker), e.g.
                              "Speaker 6 (Software Engineer, Rahul)" or
                              "Speaker 6 (Rahul)" if no role was set.
                              Left unchanged when no real name exists.
    """
    annotated = json.loads(json.dumps(raw_data))  # deep copy
    segments = annotated.get("transcript", [])
    for seg in segments:
        speaker = seg.get("speaker")
        name = name_map.get(speaker) or speaker
        seg["identified_name"] = name
        role = seg.get("role")
        if name != speaker and role:
            seg["speaker_label"] = f"{speaker} ({role}, {name})"
        elif name != speaker:
            seg["speaker_label"] = f"{speaker} ({name})"
        # if no real name resolved (name == speaker), leave any existing
        # speaker_label as-is - nothing real to add
    return annotated


def default_annotated_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_with_identity{ext or '.txt'}"


def main():
    parser = argparse.ArgumentParser(
        description="Identity resolution on a speaker-labeled (optionally role-annotated) transcript."
    )
    parser.add_argument("--input", required=True, help="Path to transcript file (.txt or .json)")
    parser.add_argument("--output", help="Path to save raw identity-resolution JSON (optional)")
    parser.add_argument(
        "--annotated-output",
        help="Path to save the transcript merged with identities, ready for Step 6. "
             "Defaults to '<input>_with_identity.<ext>'.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip producing the identity-annotated transcript; only output the raw identity JSON.",
    )
    args = parser.parse_args()

    transcript = load_transcript(args.input)
    ext = os.path.splitext(args.input)[1].lower()

    client = get_client()
    result = resolve_identities(transcript, client=client)

    output_json = json.dumps(result, indent=2)
    print(output_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nSaved identity predictions to {args.output}", file=sys.stderr)

    if args.no_annotate:
        return

    name_map = build_name_map(result)
    annotated_path = args.annotated_output or default_annotated_path(args.input)

    if ext == ".json":
        with open(args.input, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        annotated = annotate_json_transcript(raw_data, name_map)
        with open(annotated_path, "w", encoding="utf-8") as f:
            json.dump(annotated, f, indent=2, ensure_ascii=False)
    else:
        annotated_text = annotate_text_transcript(transcript, name_map)
        with open(annotated_path, "w", encoding="utf-8") as f:
            f.write(annotated_text)

    print(f"Saved identity-annotated transcript (Step 6 input) to {annotated_path}", file=sys.stderr)


if __name__ == "__main__":
    main()