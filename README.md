# Corporate Meeting Intelligence Pipeline (CMIP)

Turn a raw meeting recording into a diarized transcript, resolved speaker
roles and identities, an abstractive summary, and a structured action-item
list — no manual transcription or note-taking.

Built by Sneha Chauhan (123108044) and Anshika Sharma (123108049), under
the supervision of Dr. Himansu Sekhar Pattnayak, Department of Computer
Engineering, National Institute of Technology, Kurukshetra.

## What it does

Given an audio file, the pipeline produces:

- A time-aligned, speaker-labeled transcript
- Each speaker's inferred **role** (e.g. Host, Software Engineer) and, where
  the conversation gives evidence, their **real name**
- A short abstractive **summary**, from a BART model fine-tuned specifically
  on meeting transcripts (not a general news summarizer)
- A structured **action item** list — task, owner, deadline, priority

## Architecture

Whisper, pyannote, and the fine-tuned BART model all need a GPU; the
pipeline is split so nothing GPU-heavy has to run on your own machine.

```
 Your machine (CPU)                    Google Colab (GPU, behind ngrok)
┌─────────────────────┐                ┌──────────────────────────────┐
│ 1. Upload audio      │──POST /transcribe──▶│ faster-whisper (large-v2) │
│ 2. Role + identity    │◀── diarized transcript │ + pyannote diarization  │
│    resolution (Groq)  │                │        + merge               │
│ 4. Action items (Groq)│──POST /summarize──▶│ fine-tuned BART           │
│                        │◀── summary          │                          │
└─────────────────────┘                └──────────────────────────────┘
```

Transcription, diarization, and BART summarization run on Colab, reached
over an `ngrok` tunnel (Colab has no stable public IP of its own). Role
resolution, identity resolution, and action-item extraction are plain
Llama 3.3 70B chat-completion calls via the Groq API — no model weights to
load — so they run directly on your machine.

The full six-stage design, dataset, fine-tuning setup, and evaluation
results are written up in the project report (`report/CMIP_report.pdf`,
if included in your copy of this repo).

## Repo structure

```
route_MS.ipynb                       Colab notebook: FastAPI + ngrok, serves
                                      /transcribe and /summarize
faster_whisper_batch_largev2__MS.ipynb   Transcription + diarization, %run by route_MS.ipynb
bart_summarization.ipynb             Loads the fine-tuned BART checkpoint, %run by route_MS.ipynb
data_preprocessing.ipynb             AMI Meeting Corpus cleaning pipeline used before fine-tuning

role_resolution.py                   Step 5a: infers each speaker's role (Groq)
identity_resolution.py               Step 5b: resolves speaker IDs to real names (Groq)
step5_identity_role_resolution.py    Merges role + identity onto the transcript
action_items.py                      Step 6b: extracts task/owner/deadline/priority (Groq)
run_pipeline.py                      CLI: transcript in -> action items out (chains all four above)
main.py                              Local FastAPI server, calls the Colab endpoints

sample_meeting_new_transcript.json   Example input transcript
transcription_results.json           Example transcription+diarization output
step5_outputs/                       Example role+identity annotated transcripts
action_items_outputs/                Example extracted action items
```

## Setup

**1. Colab side** — open `route_MS.ipynb` in Colab:

```python
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")        # pyannote access
os.environ["NGROK_TOKEN"] = userdata.get("NGROK_TOKEN")  # from ngrok.com
os.environ["BART_MODEL_PATH"] = "<path to your fine-tuned checkpoint on Drive>"
```

Set these in Colab's Secrets panel (key icon, left sidebar), then run the
notebook. It prints a public URL like `https://xxxx.ngrok-free.app` — this
only stays valid while that cell keeps running; a runtime restart means a
new URL.

**2. Local side:**

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_key"
export COLAB_PIPELINE_URL="https://xxxx.ngrok-free.app"   # from step 1
```

## Running it

**End to end, via the local API:**

```bash
uvicorn main:app --reload
curl -X POST http://localhost:8000/api/summarize -F "file=@meeting.wav"
```

**Just Steps 5–6, from an existing transcript** (useful for testing
role/identity/action-item resolution without re-running Whisper):

```bash
python run_pipeline.py --input sample_meeting_new_transcript.json
python run_pipeline.py --input sample_meeting_new_transcript.json --save-intermediate
```

## Results

Fine-tuned BART summarizer, evaluated on the held-out test split:

| Metric | Trainer-reported (10-sample) | Independent eval (5-sample) |
|---|---|---|
| ROUGE-1 | 33.23% | 38.08% |
| ROUGE-2 | 7.82% | 9.27% |
| ROUGE-L | 18.74% | 21.33% |
| BLEU | — | 4.45 |
| BERTScore F1 | — | 0.8665 |

Comparable to Golia & Kalita (2023)'s action-item-driven BART pipeline
despite roughly a tenth of their fine-tuning data and no topic
segmentation. Full comparison table and methodology in the report.

## Known issues

- `main.py` reads the ngrok URL from the `COLAB_PIPELINE_URL` env var but
  then references an undefined `COLAB_PIPELINE_URL` name (instead of the
  `COLAB_URL` variable it was assigned to) when making the request —
  fix that line before relying on `/api/summarize`.

## Future work

- Front-end: live per-stage progress, tabbed Transcript/Diarization/Summary/Action-Item output
- LLM comparator — benchmark multiple LLMs on role/identity/action-item resolution against a human-annotated set
- Human evaluation of summary and action-item quality
- Robustness to overlapping speech in the diarization merge step
- Expand the fine-tuning corpus beyond 100 transcripts
