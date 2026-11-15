# CloneCheck: is that really your grandson on the phone?

**Track:** AI + Finance
**Team:** Gosha (solo)

*This project is a prototype and is not financial advice.*

**Code and results:** [CloneCheck.ipynb](CloneCheck.ipynb) (runs in a minute) · **Full method and results:** [DETAILS.md](DETAILS.md) · **Starting point:** our own idea

## The problem

In a "family emergency" scam, a caller pretends to be a grandchild in trouble and asks for money right now. Free AI tools can now copy a voice from a few seconds of audio: I copied my own on a laptop in five minutes. A grandparent who hears a familiar voice asking for help cannot tell, in the moment, whether a person or a program is speaking.

## What we built

CloneCheck helps a grandparent who gets a call from "a grandchild in trouble" by taking a few seconds of the caller's voice and saying whether it sounds like a real person or an AI-generated voice.

> **In:** a 5-second clip of *"I'm in a bit of trouble and I need your help right now."*
> **Out:** **Sounds like a real person** (AI-voice score 0%) for my own recording; **Sounds AI-generated** (score 26%, alarm above 8%) for an AI copy of my voice.

It runs in the notebook and as a small web app (`python app.py`).

**The result in one line: it catches every AI voice from the generators it trained on, 70% from generators it never met, and only 30% of AI copies of my own voice.**

## What we added

Our own idea. The organisers' sample scam-text filter says it cannot catch family-emergency scams, because they contain no scammy words. This project goes after that case as it now arrives: a phone call.

- **Our own data.** 14,375 clips in English and Russian: 6,282 real voices (ten public datasets plus 60 recordings of my own voice) and 8,093 fake voices I generated with eight AI voice generators, including copies of my own voice.
- **Our own question.** A scam call comes from a voice generator the detector has never heard, on a microphone it has never heard, over a phone line. Does it still work then?
- **Our own user.** For a grandparent, missing a fake and accusing a real grandchild are different mistakes, so they are reported separately.

## How it works

**Data.** Real speech from public datasets; fake speech made by AI voice generators reading the same texts. Every clip is trimmed, cut to 10 seconds, set to the same loudness, and scored both as recorded and through a simulated phone line.

**Model.** Each clip goes through [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m), a speech model pretrained by Meta, which we do not train. We average one of its layers (1,024 numbers per clip) and train our own logistic regression on those numbers. It is simple to explain and hard to overfit.

**Fair testing.** Data is split by where it came from, not at random. The detector trains on three generators and five real-voice datasets. Four other generators and four other sources of real voices are never seen until the test. Settings were chosen on separate dev data, and each step was committed before the next was run.

## Results

Voices and generators the detector never met, heard through a phone line:

| Method | AI voices caught | Real people wrongly flagged |
|---|---|---|
| Always say "real" (simplest baseline) | 0% (0 of 1,090) | 0% (0 of 580) |
| Logistic regression on simple audio features | 53% | 19% |
| Version 1: real voices from one dataset | 97% | 53% |
| **CloneCheck: real voices from five datasets** | **70%** | **8%** |

![Results on never-seen voices and generators](results/test_never_seen.png)

- **A perfect score was the warning.** Version 1 caught 100% of fakes on its own data. It had learned what one dataset's recordings sound like and called everything else fake, including 53% of real people. Training on five datasets fixed that; 70% is the honest number.
- **Groups.** Real people were wrongly flagged 3% of the time in English and 6% in Russian; women 1 to 3%, men 2 to 5%. This test shows no clear gap, but it is too small to rule one out.

**Where it fails.**

- **It flagged me.** It called my own laptop recordings AI in 18 of 30 clips. Adding background hiss changed that to 1 of 30, and also made it miss the AI copy of my voice (19 of 30 caught, down to 2). It had partly learned "clean recording means AI", which a scammer could beat with street noise.
- **Version 3 removed that shortcut but is not a fix.** Trained with noise and echo on real and fake clips alike, it catches 58% of new-generator fakes with or without noise (version 2: 70% without, 19% with). On a fresh phone recording of my voice it flagged the real me 10 of 30 times and caught 14 of 30 copies.
- **It misses half of Bark** (one of the new generators), usually with full confidence.

**The main takeaway.**

| Situation (CloneCheck, phone line) | AI voices caught |
|---|---|
| Generators it trained on | 100% |
| Generators it never met | 70% |
| A copy of my own voice, made from a phone memo | 30% (9 of 30) |

The further a call is from what the detector trained on, the less it catches, and a test on familiar generators hides this completely. Today the honest answer to "is that really your grandson?" is that this tool cannot tell you.

## What we'd do next

- **Train on cloning tools and hold some out.** The training generators are ordinary text-to-speech; scammers use cloning tools.
- **Real calls and more people.** Record real phone calls, and voices from family members with their permission.
- **Do not rely on the voice alone.** Hang up and call back on a number you already have, or agree a family code word.

## Sources

- **Real speech:** [FLEURS](https://huggingface.co/datasets/google/fleurs), [LibriSpeech](https://www.openslr.org/12), [VoxPopuli](https://github.com/facebookresearch/voxpopuli), [Golos](https://github.com/sberdevices/golos), [RuLibriSpeech](https://huggingface.co/datasets/bond005/rulibrispeech), [English Dialects](https://huggingface.co/datasets/ylacombe/english_dialects), [SOVA RuDevices](https://huggingface.co/datasets/bond005/sova_rudevices), [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14), [AMI](https://huggingface.co/datasets/edinburghcstr/ami), [Podlodka Speech](https://huggingface.co/datasets/bond005/podlodka_speech), and 60 recordings of my own voice.
- **Voice generators:** macOS `say`, [Piper](https://github.com/OHF-Voice/piper1-gpl), [MMS-TTS](https://huggingface.co/facebook/mms-tts-eng), [Silero](https://github.com/snakers4/silero-models), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [Bark](https://huggingface.co/suno/bark-small), [Chatterbox](https://github.com/resemble-ai/chatterbox), [Vosk TTS](https://github.com/alphacep/vosk-tts).
- **Pretrained models:** [XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (the detector's features); [Whisper small](https://huggingface.co/openai/whisper-small) (only to find the sentences inside my phone memo).
- **Libraries:** PyTorch, Hugging Face Transformers, scikit-learn, librosa, SciPy, pandas, soundfile, matplotlib, Gradio.
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used heavily as a coding assistant: it wrote most of the code and helped draft this write-up, working from my idea and direction.
- Licences and the full list of links are in [DETAILS.md](DETAILS.md).

## Run it yourself

```bash
git clone https://github.com/ImGosha/clonecheck.git && cd clonecheck
pip install -r requirements.txt
python scripts/evaluate_test.py     # re-trains and re-scores from saved features, no audio needed
python app.py                       # the demo app
```
