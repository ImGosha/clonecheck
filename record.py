"""Record your own voice for the "my voice against a clone of my voice" test.

Run:  python record.py     then open the local address it prints.

The page shows one sentence at a time. Press the microphone, read the
sentence in your normal voice, stop, and press "Save and next". (If the
page cannot find a microphone, open the same address in Chrome or Safari,
or record with any app and upload the file.) Recordings are saved in
data/audio/own/ and stay on this computer (that folder is not uploaded to
GitHub).
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
import soundfile as sf

from clonecheck.audio import load
from clonecheck.config import AUDIO, DATA, RATE

OUT = AUDIO / "own"
SENTENCES = pd.read_csv(DATA / "own_sentences.csv")


def first_missing() -> int:
    for i, sentence_id in enumerate(SENTENCES["id"]):
        if not (OUT / f"{sentence_id}.wav").exists():
            return i
    return len(SENTENCES)


def show(i: int):
    done = sum((OUT / f"{s}.wav").exists() for s in SENTENCES["id"])
    progress = f"**{done} of {len(SENTENCES)} recorded.**"
    if i >= len(SENTENCES):
        return i, "## All done. Thank you!", progress + " You can close this page.", None
    row = SENTENCES.iloc[i]
    return i, f"## {row['text']}", f"{progress} Sentence {i + 1} ({row['lang']})", None


def save(i: int, recording: str | None):
    if recording is None or i >= len(SENTENCES):
        return show(i)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / f"{SENTENCES.iloc[i]['id']}.wav", load(recording), RATE, subtype="PCM_16")
    return show(i + 1)


with gr.Blocks(title="CloneCheck recorder") as page:
    gr.Markdown("# Record your voice\nRead each sentence in your normal speaking voice, "
                "the way you would say it on the phone.")
    index = gr.State(0)
    sentence = gr.Markdown()
    status = gr.Markdown()
    mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your recording")
    with gr.Row():
        back = gr.Button("Back")
        keep = gr.Button("Save and next", variant="primary")
    outputs = [index, sentence, status, mic]
    page.load(lambda: show(first_missing()), outputs=outputs)
    keep.click(save, inputs=[index, mic], outputs=outputs)
    back.click(lambda i: show(max(i - 1, 0)), inputs=index, outputs=outputs)

if __name__ == "__main__":
    page.launch(server_port=7861)
