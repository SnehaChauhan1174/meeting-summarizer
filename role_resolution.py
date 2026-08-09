"""
role_resolution.py

Step 5 (partial) of the Corporate Meeting Intelligence Pipeline.
-----------------------------------------------------------------
Takes a CLEAN, SPEAKER-LABELED transcript (the output of Step 4:
"Merge & Clean") and infers the ROLE of each speaker using an LLM.

This module does ONLY role inference. It does not:
  - rewrite/summarize the transcript
  - generate action items
  - resolve real names/identities (that is a separate, later step)

Two input formats are supported (auto-detected from file extension):

1) PLAIN TEXT (.txt) — matches earlier Step 4 output:

    Speaker 1:
    Good morning everyone. Let's start the quarterly review.
    Speaker 2:
    I'll present the Q2 revenue numbers and financial forecast.

2) JSON (.json) — timestamped segment list, e.g.:

    {
      "created_at": "15-07-2026 10:00:00",
      "transcript": [
        {"start": 0, "end": 12, "speaker": "Speaker 1", "text": "Morning everyone..."},
        {"start": 12, "end": 21, "speaker": "Speaker 6", "text": "Yeah, I'm here..."}
      ]
    }

Usage:
    export GROQ_API_KEY="your_key_here"

    # Run on a transcript file (.txt or .json):
    python role_resolution.py --input sample_transcript.txt
    python role_resolution.py --input sample_meeting_new_transcript.json

    # Save output to a file:
    python role_resolution.py --input sample_meeting_new_transcript.json --output roles.json

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

SYSTEM_PROMPT = """You are an AI specialized in corporate meeting analysis.
You will receive a clean, speaker-labeled transcript produced after speaker diarization.

YOUR TASK
Your ONLY responsibility is to infer the ROLE of each speaker.

DO NOT:
- rewrite the transcript
- summarize the meeting
- generate action items
- identify real names
- invent information not supported by the transcript

Use only contextual clues from what each speaker says.

IMPORTANT — EVALUATE SPEAKERS INDEPENDENTLY:
Do not assume every speaker shares the same functional background just
because the overall meeting sounds technical/engineering-heavy (or sounds
dominated by any other single function). Meetings routinely mix people
from different functions — e.g. a Product Manager or Business Analyst
sitting in on an engineering standup, a Designer joining a technical
review, a Client or Executive dropping into a sprint call. Judge each
speaker strictly on their OWN lines, not on the theme of the meeting as
a whole or the roles you've already assigned to other speakers.
Cues like scoping/prioritization decisions ("let's take that in the next
sprint", "that's out of scope for this release"), asking about timelines
or documentation, or approving budget/requirements point toward
Product/Program/Business roles even inside an otherwise technical
conversation — don't default such speakers to an engineering role just
because their neighbors are engineers.

Possible roles include (but are not limited to):

Leadership / Executive:
- CEO
- Managing Director
- Executive (VP / Director-level, function unspecified)
- Founder / Co-founder
- Board Member

Program & Product:
- Program Manager
- Project Manager
- Product Manager
- Product Owner
- Scrum Master

Engineering & Technical:
- Engineering Manager
- Software Engineer
- DevOps Engineer
- QA Engineer
- Frontend Engineer
- Backend Engineer
- Data Engineer
- Data Scientist
- Solutions Architect / Technical Architect

Data & Business:
- Data Analyst
- Business Analyst
- Finance Lead
- Researcher / Analyst
- Legal / Compliance
- Operations Manager

Design:
- Designer / UX Lead

Documentation:
- Technical Writer

Marketing & Sales:
- Marketing Lead
- Sales Lead
- Account Manager
- Customer Success Manager

People & Support:
- HR
- Recruiter

External Parties:
- Client
- Customer
- Vendor / Partner
- Consultant

Meeting-specific / Generic:
- Host
- Meeting Organizer
- Team Member
- Unknown

ROLE DISAMBIGUATION NOTES:
- Only use "CEO" / "Managing Director" / "Founder" if there is a strong,
  explicit cue (e.g. others deferring to them on company-wide decisions,
  budget/strategy authority, being addressed with that title). Do not
  assume seniority just because someone speaks first or leads the call —
  that alone points to "Host" or "Meeting Organizer", not a C-level role.
- "Executive" is a fallback for clear senior/leadership signals when a
  more specific title (CEO, Managing Director, VP of X) isn't evident.
- "Engineering Manager" = leads/coordinates engineers but doesn't write
  code themselves in the transcript (assigns work, tracks status, removes
  blockers). "Software Engineer" = discusses writing/fixing/shipping code
  directly.

ROLE DISAMBIGUATION NOTES:
- "DevOps Engineer" = discusses deployment windows, infrastructure stability,
  deployment configs, CI/CD, releases going out. Use this instead of
  "Operations Manager" when the cues are about deploying/running software
  systems (technical), not business/office operations.
- "Operations Manager" = business/day-to-day operations, not infrastructure
  or deployments. Only use this when the cues are clearly non-technical.
- "Technical Writer" vs "Product Manager" / "Business Analyst": merely
  saying "I'll update the documentation" is weak, passive evidence — many
  roles maintain docs as a side task. Weigh it lower than DECISION-MAKING
  language from the same speaker. If a speaker also makes scope or
  prioritization calls (e.g. "let's take that as a separate story next
  sprint", "that's out of scope for this release", deciding what ships
  now vs. later), treat that as the stronger signal and prefer
  "Product Manager" or "Business Analyst" over "Technical Writer" — actual
  technical writers document what already exists, they don't decide what
  gets built or deferred.

If the transcript does not provide enough evidence, assign "Unknown".

CONFIDENCE CRITERIA (apply strictly):
- High   = role is supported by 2 or more independent behavioral/content cues
- Medium = role is supported by exactly 1 clear cue
- Low    = role is inferred mostly by elimination or very weak/indirect cues

IMPORTANT:
- Include an entry for EVERY unique Speaker ID present in the input,
  even if there is little or no evidence for that speaker (use "Unknown"
  with Low confidence in that case).
- Output ONLY valid JSON. No preamble, no explanation, no markdown fences.

Output format:
{
  "speakers": [
    {
      "speaker": "Speaker 1",
      "role": "Host",
      "confidence": "High",
      "evidence": [
        "Started the meeting",
        "Directed discussion",
        "Assigned follow-up work"
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
    Convert a list of timestamped segments:
        {"start": 0, "end": 12, "speaker": "Speaker 1", "text": "..."}
    into the plain "Speaker N:\\ntext" format the prompt expects.

    Consecutive segments from the same speaker are merged into one block
    so the transcript reads naturally (matches how Step 4 output looks).
    """
    lines = []
    last_speaker = None
    buffer = []

    def flush():
        if last_speaker is not None and buffer:
            lines.append(f"{last_speaker}:")
            lines.append(" ".join(buffer))

    for seg in segments:
        speaker = seg.get("speaker", "Unknown Speaker")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if speaker != last_speaker:
            flush()
            buffer = [text]
            last_speaker = speaker
        else:
            buffer.append(text)
    flush()

    return "\n".join(lines)


def load_transcript(path: str) -> str:
    """
    Load a transcript file and normalize it to the plain-text
    "Speaker N:\\ntext" format used by the prompt, regardless of
    whether the source file is .txt or .json.
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

    # default: plain text (.txt or anything else)
    return raw


def build_role_map(roles_result: Dict) -> Dict[str, str]:
    """Turn the role-resolution output into {'Speaker 1': 'Host', ...}."""
    return {
        s.get("speaker"): s.get("role", "Unknown")
        for s in roles_result.get("speakers", [])
        if s.get("speaker")
    }


def annotate_text_transcript(transcript_text: str, role_map: Dict[str, str]) -> str:
    """
    Rewrite 'Speaker N:' labels as 'Speaker N (Role):' in a plain-text
    transcript, e.g.:
        Speaker 1:            ->   Speaker 1 (Host):
        Good morning...            Good morning...
    """
    def repl(match):
        speaker = match.group(1)
        role = role_map.get(speaker, "Unknown")
        return f"{speaker} ({role}):"

    return re.sub(r"^(Speaker\s+\d+)\s*:", repl, transcript_text, flags=re.MULTILINE)


def annotate_json_transcript(raw_data: dict, role_map: Dict[str, str]) -> dict:
    """
    Add role info onto each segment of a JSON transcript, preserving the
    original structure (start/end/speaker/text) so downstream steps that
    need timestamps still work. Adds two new fields per segment:
        "role"          -> "Host"
        "speaker_label" -> "Speaker 1 (Host)"
    """
    annotated = json.loads(json.dumps(raw_data))  # deep copy
    segments = annotated.get("transcript", [])
    for seg in segments:
        speaker = seg.get("speaker")
        role = role_map.get(speaker, "Unknown")
        seg["role"] = role
        seg["speaker_label"] = f"{speaker} ({role})"
    return annotated


def default_annotated_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_with_roles{ext or '.txt'}"


def extract_speaker_ids(transcript: str) -> List[str]:
    """Pull out unique 'Speaker N' labels in order of first appearance."""
    ids = re.findall(r"^(Speaker\s+\d+)\s*:", transcript, flags=re.MULTILINE)
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
    # If there's leading text before the first '{', drop it
    first_brace = raw.find("{")
    if first_brace > 0:
        raw = raw[first_brace:]
    return raw


def resolve_roles(transcript: str, client: Groq = None) -> Dict:
    """
    Send a speaker-labeled transcript to the LLM and get back
    role predictions as a Python dict.
    """
    if client is None:
        client = get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,  # deterministic — role inference shouldn't be "creative"
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

    # Sanity check: make sure every speaker in the transcript got an entry
    expected_ids = set(extract_speaker_ids(transcript))
    returned_ids = {s.get("speaker") for s in result.get("speakers", [])}
    missing = expected_ids - returned_ids
    if missing:
        print(f"WARNING: model omitted speakers: {missing}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="Role resolution on a speaker-labeled transcript.")
    parser.add_argument("--input", required=True, help="Path to transcript file (.txt or .json)")
    parser.add_argument("--output", help="Path to save raw role-resolution JSON (optional)")
    parser.add_argument(
        "--annotated-output",
        help="Path to save the Step-4 transcript merged with roles, ready for Step 6. "
             "Defaults to '<input>_with_roles.<ext>'.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip producing the role-annotated transcript; only output the raw role JSON.",
    )
    args = parser.parse_args()

    transcript = load_transcript(args.input)
    ext = os.path.splitext(args.input)[1].lower()

    client = get_client()
    result = resolve_roles(transcript, client=client)

    output_json = json.dumps(result, indent=2)
    print(output_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nSaved role predictions to {args.output}", file=sys.stderr)

    if args.no_annotate:
        return

    role_map = build_role_map(result)
    annotated_path = args.annotated_output or default_annotated_path(args.input)

    if ext == ".json":
        with open(args.input, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        annotated = annotate_json_transcript(raw_data, role_map)
        with open(annotated_path, "w", encoding="utf-8") as f:
            json.dump(annotated, f, indent=2, ensure_ascii=False)
    else:
        annotated_text = annotate_text_transcript(transcript, role_map)
        with open(annotated_path, "w", encoding="utf-8") as f:
            f.write(annotated_text)

    print(f"Saved role-annotated transcript (Step 6 input) to {annotated_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
