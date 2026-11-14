"""Write CloneCheck.ipynb from the cells below.

Keeping the notebook's cells in a script makes them easy to review and
rebuild. After building, run it so the outputs are saved in the file:

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace CloneCheck.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# CloneCheck: is that really your grandson on the phone?

**Veritas AI International High School Hackathon 2026 · AI + Finance · Gosha**

*This project is a prototype and is not financial advice.*

CloneCheck helps a grandparent who gets a call from "a grandchild in trouble" by taking a few seconds of the caller's voice and saying whether it sounds like a real person or an AI-generated voice.

This notebook shows the data, trains the detector, and reports every result, including the ones that went badly. It runs in about a minute, because the slow part (turning 14,315 audio clips into numbers) was done in advance and saved in the repository.
""")

md("## 0. Setup")
code("""
import os, subprocess
in_project = os.path.exists("clonecheck/config.py")
in_project or subprocess.run("git clone -q https://github.com/ImGosha/clonecheck.git", shell=True)
in_project or os.chdir("clonecheck")
""")
code("""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from IPython.display import Audio, Image, display
from clonecheck import detect, report
from clonecheck.config import TRAIN_SOURCES
pd.set_option("display.max_colwidth", 100)
""")

md("""
## 1. The data

Real voices come from ten public datasets and my own recordings. Fake voices come from eight AI voice generators, which read out the same texts as the real clips.

The key design choice is the **role** column. A generator or dataset marked `train` is one the detector learns from. `dev` ones are never trained on and were used to choose settings. `test` ones were never trained on and never used to choose anything: they were scored once, at the end.
""")
code("report.clip_counts()")

md("""
### Listen

One sentence from the test set, said by a real person and by four AI voice generators. The last two columns are CloneCheck's AI-voice score for each clip (0 = sure it is real, 1 = sure it is AI; the alarm goes off above 0.08).
""")
code("""
examples = pd.read_csv("examples/examples.csv")
print(examples["text"][0])
examples.drop(columns=["text", "clip_id"])
""")
code("""
for row in examples.itertuples():
    print(row.what)
    display(Audio(f"examples/{row.file}"))
""")
md("The same real clip through the simulated phone line, which is how every clip is also tested:")
code('Audio("examples/real_phone.wav")')

md("""
## 2. Turning sound into numbers, and the model

Each clip is passed through **XLS-R**, a speech model Meta pretrained on 436,000 hours of speech in 128 languages. We do not train XLS-R. We keep the average of its layer-10 activations: 1,024 numbers per clip. The baseline uses 80 classic audio numbers (MFCCs) instead.

Those numbers were computed in advance (`scripts/features.py`). Here they are loaded:
""")
code("""
features = detect.load_features(layers=[10])
clips = detect.load_clips()
train_clips = clips[clips["split"] == "train"]
dev_clips = clips[clips["split"] == "dev"]
test_clips = clips[clips["split"] == "test"]
print(len(train_clips), "train clips,", len(dev_clips), "dev clips,", len(test_clips), "test clips")
print("numbers per clip:", detect.matrix(features, train_clips[:1], "clean", "xlsr", 10).shape[1])
""")
md("""
The detector itself is a **logistic regression**: it learns one weight for each of the 1,024 numbers and outputs the probability that the voice is AI-generated. `detect.train` is a thin wrapper around scikit-learn's `StandardScaler` and `LogisticRegression`.
""")

md("""
## 3. Version 1: real voices from one dataset

The first version learned from real voices in a single dataset (FLEURS) and fakes from three generators.
""")
code("""
fleurs_only = train_clips[train_clips["text_source"] == "fleurs"]
version1 = detect.train(features, fleurs_only, "xlsr", 10, conditions=("clean",))
real_dev_fleurs = dev_clips[(dev_clips["label"] == "real") & (dev_clips["source"] == "fleurs")]
detect.set_threshold(version1, features, real_dev_fleurs)   # alarm level: at most 5% of real dev clips flagged
report.flag_table(version1, features, dev_clips)
""")
md("""
Read the `fake` rows first: **every fake is caught**, including Silero, a generator it never trained on. That looked too good to be true.

Now the `real` rows. FLEURS voices are fine. But it flags most real people from every other dataset: audiobook readers (LibriSpeech, RuLibriSpeech), politicians (VoxPopuli), people with regional accents (English Dialects).

Version 1 did not learn what a human voice sounds like. It learned what a *FLEURS recording* sounds like, and calls everything else fake. That is also why it "catches" every fake.
""")

md("""
## 4. Version 2 (CloneCheck): real voices from five datasets

The fix: real voices from five datasets in training, each training clip also heard through the simulated phone line, and the alarm level set on dev voices from all five.
""")
code("""
clonecheck = detect.train(features, train_clips, "xlsr", 10, conditions=("clean", "phone"))
real_dev = dev_clips[(dev_clips["label"] == "real") & dev_clips["source"].isin(TRAIN_SOURCES)]
detect.set_threshold(clonecheck, features, real_dev)
report.flag_table(clonecheck, features, dev_clips)
""")
md("""
On the two real datasets it never trained on (English Dialects, SOVA), false alarms fell from 52-92% to 6-22%, and it still catches 91-99% of Silero. These dev results are what the settings were chosen on. The settings were then frozen and committed before the test below.
""")

md("""
## 5. The test (scored once)

Baselines for comparison: always saying "real", and the same logistic regression on the 80 simple audio numbers.
""")
code("""
simple = detect.train(features, train_clips, "mfcc", None, conditions=("clean", "phone"))
detect.set_threshold(simple, features, real_dev)
methods = {"Always say \\"real\\"": None, "Simple audio features": simple,
           "Version 1 (one real dataset)": version1, "CloneCheck (many real datasets)": clonecheck}
report.compare(methods, features, test_clips, condition="phone")
""")
md("""
These are the numbers for clips heard through the phone line. "New" means generators and datasets the detector never met.

- **Version 1** catches 97% of new-generator fakes only because it flags 53% of new real people too.
- **CloneCheck** catches 70% and wrongly flags 8%. That is far better than the simple baseline on both counts, and still not good enough for someone to rely on.
- On generators it trained on, CloneCheck catches 100%. The drop to 70% is the cost of a generator being new.
""")
code('display(Image("results/test_never_seen.png"))')

md("### By generator and by dataset\n\nFake clips: % caught. Real clips: % wrongly flagged.")
code('report.flag_table(clonecheck, features, test_clips, counts=True)')
md("""
- **Vosk** is caught every time; **Bark** and **Kokoro** get through about half the time.
- **My own voice** is the worst row among the real voices. Section 6 looks at why.
""")

md("### Does it treat groups differently?\n\nFalse alarms on real test voices by speaker gender, for the datasets that record gender (FLEURS and VoxPopuli).")
code('report.by_gender("xlsr, many + phone")')
md("The ranges overlap, so this test does not show a gender gap. It is too small to rule one out. False alarms are a little higher in Russian than in English for both.")

md("""
## 6. Where it fails

### Real people it was most sure were AI
""")
code('report.mistakes("xlsr, many + phone", "real")')
md("### AI voices it was most sure were real")
code('report.mistakes("xlsr, many + phone", "fake")')

md("""
### It flagged me

My own real voice was called AI in 5 of 15 English clips and 13 of 15 Russian clips. In Russian, it flagged the real me more often than it caught the AI copy of me.

My recordings were made in a browser, which removes background noise and leaves exact digital silence between words. To test whether that is the reason, we added steady background hiss to the same clips and scored them again (`scripts/hiss_experiment.py`):
""")
code('pd.read_csv("results/own_voice_hiss.csv")')
md("""
With hiss added, the detector stops flagging my real voice (18 of 30 down to 1 of 30). But it also stops catching the AI copy (19 of 30 down to 2 of 30).

So part of what CloneCheck learned is **"a clean recording is AI, background noise is human"**. This is the same kind of shortcut as version 1, one level down. A scammer could exploit it by playing street noise behind a cloned voice.
""")

md("""
## 7. Summary

| | AI voices caught (new generators, phone line) | Real people wrongly flagged (new datasets, phone line) |
|---|---|---|
| Always say "real" | 0% | 0% |
| Simple audio features | 53% | 19% |
| Version 1 (one real dataset) | 97% | 53% |
| **CloneCheck** | **70%** | **8%** |

**The main takeaway:** a voice-clone detector can score 100% on its own test and still be useless, because it learns where its recordings came from instead of what a human sounds like. Holding out whole generators and whole datasets exposed that twice: first for datasets (version 1), then for recording conditions (my own microphone).

**What it would take to trust it:** training that cannot use "how clean is the recording?" as a clue, a test on real phone calls, and many more voices. Until then the safest advice does not need AI: hang up and call back on a number you already have.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(notebook, ROOT / "CloneCheck.ipynb")
print(f"wrote CloneCheck.ipynb with {len(cells)} cells")
