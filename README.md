# Meeting Summarizer

# Data peprocessing
Data preprocessing pipeline applied to the AMI dataset before fine-tuning the summarization model.

# Dataset
Source: AMI Meeting Corpus via HuggingFace
Config: ihm (individual headset microphone)
Splits: train, validation, test
Key columns used: meeting_id, speaker_id, text, begin_time, end_time, summary

# Cleaning Pipeline
The full cleaning is applied as a single function clean_ami_dataframe() which processes each split independently.

## Cleaning Pipeline

| Step | Operation | Description |
|-----|-----------|-------------|
| 1 | Drop NULL text | Removes rows where `text` is `None` |
| 2 | Drop empty / whitespace | Removes rows where text is blank or spaces only |
| 3 | Normalize casing | Converts all text to lowercase |
| 4 | Drop pure backchannels | Drops utterances made entirely of reaction words like `yeah`, `okay`, `mm` |
| 5 | Remove filler words | Removes `um`, `uh`, `er`, `hmm` from inside sentences, keeps the rest |
| 6 | Remove broken fragments | Removes incomplete words ending with hyphen like `the-`, `should-` |
| 7 | Drop short utterances | Drops any utterance with fewer than **3 words** after all cleaning |



