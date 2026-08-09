"""
step5_identity_role_resolution.py

Step 5 of the Corporate Meeting Intelligence Pipeline (unified).
------------------------------------------------------------------
This is the single entry point for Step 5 in the pipeline diagram:
"IDENTITY & ROLE RESOLUTION -> Named Speakers with Roles".

It takes the CLEAN, SPEAKER-LABELED transcript produced by Step 4
("Merge & Clean") and runs BOTH:
    - role_resolution.resolve_roles()       (role_resolution.py)
    - identity_resolution.resolve_identities() (identity_resolution.py)
as two independent LLM calls against the same Step-4 input, then MERGES
the two results into a single annotated transcript, ready to be handed
straight to Step 6 (action_items.py).

Why two independent calls instead of one combined prompt:
    Role inference and identity inference use different, sometimes
    unrelated evidence (behavioral/content cues vs. self-intro/address
    cues). Keeping them separate means each task gets a prompt fully
    focused on its own job, and a mistake in one doesn't skew the other.
    This script is what hides that internal split — from the outside,
    and to every step downstream, Step 5 looks like ONE stage with ONE
    output, exactly as in the pipeline diagram.

OUTPUT
    A single annotated transcript (same input format, .txt or .json)
    where every speaker turn carries:
        - role             (e.g. "Software Engineer")
        - identified_name  (e.g. "Deba", or null if unresolved)
        - speaker_label    (e.g. "Speaker 3 (Software Engineer, Deba)")
    This is the ONE file Step 6 needs — no separate role/identity files
    to juggle.

Usage:
    export GROQ_API_KEY="your_key_here"

    python step5_identity_role_resolution.py --input step4_output.json
    python step5_identity_role_resolution.py --input step4_output.json --output step6_input.json

Requires:
    pip install groq
    role_resolution.py and identity_resolution.py in the same directory
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Optional

from groq import Groq

import role_resolution as rr
import identity_resolution as ir


# Folder Step 5's merged transcripts get saved into automatically. Created
# on first use if it doesn't exist yet - never needs to be made by hand.
STEP5_OUTPUT_FOLDER = "step5_outputs"


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


def default_output_path(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1] or ".txt"
    return get_next_sequential_path(STEP5_OUTPUT_FOLDER, "step6_input", ext)


def merge_text_transcript(plain_transcript: str, role_map: Dict[str, str],
                           name_map: Dict[str, str]) -> str:
    """
    Rewrite plain 'Speaker N:' labels into the single combined label
    'Speaker N (Role, Name):'. If no real name was resolved for a
    speaker, name_map's value for that speaker equals the speaker's own
    ID (per identity_resolution.build_name_map's fallback) - in that
    case nothing is tacked on and the label stays 'Speaker N (Role):'.
    """
    import re

    def repl(match):
        speaker = match.group(1)
        role = role_map.get(speaker, "Unknown")
        name = name_map.get(speaker)
        if name and name != speaker:
            return f"{speaker} ({role}, {name}):"
        return f"{speaker} ({role}):"

    return re.sub(r"^(Speaker\s+\d+)\s*:", repl, plain_transcript, flags=re.MULTILINE)


def merge_json_transcript(raw_data: dict, role_map: Dict[str, str],
                           name_map: Dict[str, str]) -> dict:
    """
    Add role + identity fields onto each segment of the Step-4 JSON
    transcript in one pass. Only two new fields are added per segment:
        "role"            -> "Software Engineer" (or "Unknown")
        "identified_name" -> "Deba" if resolved, or the speaker's own ID
                              (e.g. "Speaker 8") as a stand-in if no
                              self-intro/address evidence exists at all.
                              Never null - "identified_name == speaker"
                              is the unambiguous "unresolved" signal for
                              anything reading this field downstream.
    No combined "speaker_label" string is stored - it would just be a
    redundant restatement of "speaker" + "role" + "identified_name" that
    could drift out of sync with them. Anything downstream (e.g.
    action_items.py) builds the display label on the fly from these
    three fields instead.
    """
    merged = json.loads(json.dumps(raw_data))  # deep copy
    segments = merged.get("transcript", [])
    for seg in segments:
        speaker = seg.get("speaker")
        role = role_map.get(speaker, "Unknown")
        name = name_map.get(speaker) or speaker
        seg["role"] = role
        seg["identified_name"] = name
    return merged


def run_step5(input_path: str, client: Optional[Groq] = None):
    """
    Run role resolution + identity resolution against the same Step-4
    input and return (merged_transcript, roles_result, identity_result).
    merged_transcript is a str for .txt input, or a dict for .json input.
    """
    if client is None:
        client = rr.get_client()

    ext = os.path.splitext(input_path)[1].lower()

    # Both resolvers read the SAME raw Step-4 input independently.
    plain_transcript = rr.load_transcript(input_path)

    print("Running role resolution...", file=sys.stderr)
    roles_result = rr.resolve_roles(plain_transcript, client=client)
    role_map = rr.build_role_map(roles_result)

    print("Running identity resolution...", file=sys.stderr)
    identity_result = ir.resolve_identities(plain_transcript, client=client)
    name_map = ir.build_name_map(identity_result)

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        merged_transcript = merge_json_transcript(raw_data, role_map, name_map)
    else:
        merged_transcript = merge_text_transcript(plain_transcript, role_map, name_map)

    return merged_transcript, roles_result, identity_result


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: run role + identity resolution on a Step-4 transcript and "
                    "produce ONE merged, annotated transcript for Step 6."
    )
    parser.add_argument("--input", required=True, help="Path to the Step-4 transcript (.txt or .json)")
    parser.add_argument(
        "--output",
        help=f"Path to save the merged Step-6-ready transcript. If omitted, "
             f"auto-saves into ./{STEP5_OUTPUT_FOLDER}/ with sequential "
             f"numbering (step6_input_0001.json, _0002.json, ...) - no "
             f"filename needed, just run the script again on your next transcript.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save the raw role-resolution and identity-resolution JSON "
             "(<output>.roles.json / <output>.identity.json) for debugging.",
    )
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    client = rr.get_client()

    merged_transcript, roles_result, identity_result = run_step5(args.input, client=client)

    output_path = args.output or default_output_path(args.input)

    if ext == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_transcript, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged_transcript)

    print(f"\nSaved Step 6 input (merged role + identity) to {output_path}", file=sys.stderr)

    if args.save_raw:
        with open(f"{output_path}.roles.json", "w", encoding="utf-8") as f:
            json.dump(roles_result, f, indent=2)
        with open(f"{output_path}.identity.json", "w", encoding="utf-8") as f:
            json.dump(identity_result, f, indent=2)
        print(f"Saved raw role/identity JSON alongside {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()