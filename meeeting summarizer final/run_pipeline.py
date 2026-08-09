"""
run_pipeline.py

Full Step 5 + Step 6 runner for the Corporate Meeting Intelligence Pipeline.
-----------------------------------------------------------------------------
One command: raw Step-4 transcript in, final action items JSON out.

Internally chains:
    1. role_resolution.resolve_roles()          (role_resolution.py)
    2. identity_resolution.resolve_identities()  (identity_resolution.py)
    3. step5_identity_role_resolution merge      (step5_identity_role_resolution.py)
    4. action_items.extract_action_items()       (action_items.py)

All four steps share ONE Groq client (one API key setup, four calls under
the hood: role, identity, then action-item extraction). By default only
the FINAL action items JSON is written - nothing else - since that's the
only thing you actually need at the end. Pass --save-intermediate if you
also want the merged Step-5 transcript written out for debugging/BART.

Usage:
    export GROQ_API_KEY="your_key_here"

    python run_pipeline.py --input sample_meeting_new_transcript.json
    python run_pipeline.py --input sample_meeting_new_transcript.json --output action_items.json

    # Also keep the intermediate Step-5 merged transcript:
    python run_pipeline.py --input sample_meeting_new_transcript.json --save-intermediate

Requires:
    pip install groq
    role_resolution.py, identity_resolution.py, step5_identity_role_resolution.py,
    and action_items.py all present in the same directory.
"""

import argparse
import json
import os
import sys

import role_resolution as rr
import identity_resolution as ir
import step5_identity_role_resolution as step5
import action_items as ai


def transcript_to_labeled_text(merged_transcript, ext: str) -> str:
    """
    Turn the Step-5 merged transcript (a dict for .json input, or an
    already-labeled str for .txt input) into the flat "Speaker N (Role,
    Name): text" text that action_items.extract_action_items() expects.
    """
    if ext == ".json":
        segments = merged_transcript.get("transcript", [])
        return ai.json_segments_to_labeled_text(segments)
    # .txt path: step5's merge_text_transcript already produced fully
    # labeled "Speaker N (Role, Name):" text - nothing further to do.
    return merged_transcript


def run_full_pipeline(input_path: str, client=None):
    """
    Run the whole Step 5 + Step 6 chain and return
    (action_items_result, merged_transcript, roles_result, identity_result).
    """
    if client is None:
        client = rr.get_client()

    ext = os.path.splitext(input_path)[1].lower()

    print("Step 5a: role resolution...", file=sys.stderr)
    print("Step 5b: identity resolution...", file=sys.stderr)
    merged_transcript, roles_result, identity_result = step5.run_step5(input_path, client=client)

    labeled_text = transcript_to_labeled_text(merged_transcript, ext)

    print("Step 6: action item extraction...", file=sys.stderr)
    action_items_result = ai.extract_action_items(labeled_text, client=client)

    return action_items_result, merged_transcript, roles_result, identity_result


def main():
    parser = argparse.ArgumentParser(
        description="Run role resolution + identity resolution + action item extraction "
                    "in one command: raw Step-4 transcript in, action items JSON out."
    )
    parser.add_argument("--input", required=True, help="Path to the Step-4 transcript (.txt or .json)")
    parser.add_argument(
        "--output",
        help=f"Path to save the final action items JSON. If omitted, "
             f"auto-saves into ./{ai.ACTION_ITEMS_OUTPUT_FOLDER}/ with "
             f"sequential numbering (action_items_0001.json, _0002.json, ...) "
             f"- no filename needed, just run the script again on your next transcript.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help=f"Also save the merged Step-5 transcript into "
             f"./{step5.STEP5_OUTPUT_FOLDER}/ (step6_input_0001.json, "
             f"_0002.json, ...), same auto-numbering as running "
             f"step5_identity_role_resolution.py on its own.",
    )
    args = parser.parse_args()

    client = rr.get_client()

    action_items_result, merged_transcript, roles_result, identity_result = run_full_pipeline(
        args.input, client=client
    )

    output_json = json.dumps(action_items_result, indent=2)
    print(output_json)

    output_path = args.output or ai.default_output_path()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"\nSaved final action items to {output_path}", file=sys.stderr)

    if args.save_intermediate:
        ext = os.path.splitext(args.input)[1].lower()
        intermediate_path = step5.get_next_sequential_path(
            step5.STEP5_OUTPUT_FOLDER, "step6_input", ext or ".txt"
        )
        if ext == ".json":
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(merged_transcript, f, indent=2, ensure_ascii=False)
        else:
            with open(intermediate_path, "w", encoding="utf-8") as f:
                f.write(merged_transcript)
        print(f"Saved intermediate Step-5 transcript to {intermediate_path}", file=sys.stderr)


if __name__ == "__main__":
    main()