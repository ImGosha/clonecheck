# CloneCheck: is that really your grandson on the phone?

**Track:** AI + Finance
**Team:** Gosha (solo)

This project is a prototype and is not financial advice.

**Code:** [CloneCheck.ipynb](CloneCheck.ipynb), saved with its outputs ([open in Colab](https://colab.research.google.com/github/ImGosha/clonecheck/blob/main/CloneCheck.ipynb)) · **Long version:** [DETAILS.md](DETAILS.md)

## The problem

In a "family emergency" scam, a caller pretends to be a grandchild in trouble and asks for money right now. Free AI tools can now copy a voice from a few seconds of audio: I copied my own on a laptop in five minutes. A grandparent who hears a familiar voice asking for help cannot tell, in the moment, whether a person or a program is speaking.

## What we built

CloneCheck helps a grandparent who gets a call from "a grandchild in trouble" by taking a few seconds of the caller's voice and giving back "sounds like a real person" or "sounds AI-generated", with a score.

> **In:** a 5-second clip of *"I'm in a bit of trouble and I need your help right now."*
> **Out:** for my own recording, **Sounds like a real person** (AI-voice score 0%). For an AI copy of my voice saying the same words, **Sounds AI-generated** (score 26%; the alarm goes off above 8%).

To use it, you put the call on speaker, record a few seconds, and drop the clip into the app (`python app.py`).

## What we added

Our own idea. The sample scam-text filter says it cannot catch family-emergency scams, because they contain no scammy words. This project goes after that case as it now arrives: a phone call.

- **Our own data.** 14,375 clips in English and Russian. I generated the 8,093 fake ones with eight AI voice generators, recorded my own voice twice, and made AI copies of it.
- **Our own question.** A scam call uses a voice generator the detector has never heard, on a microphone it has never heard, over a phone line. Does it still work then?
- **Our own user.** For a grandparent, missing a fake and accusing a real grandchild are different mistakes, so they are counted separately.

## How it works

**Data.** 6,282 real clips from ten public speech datasets and my own recordings, and 8,093 fake clips in which AI voice generators read the same texts. Cleaning: every clip is trimmed of silence, cut to 10 seconds and set to the same loudness, then scored as recorded and through a simulated phone line. Sentences with digits were left out of the fake set, because generators read them differently.

**Split.** By where a clip came from, not at random: 4,459 clips to train, 4,433 dev clips to choose settings, 5,483 test clips (60 of them, a second recording of my voice, were added after the main test). The detector trains on three generators and five real-voice datasets. Four other generators and four other sources of real voices appear only in the test. The clips have no dates, so the "train on earlier, test on later" rule does not apply.

**Model.** Each clip goes through [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m), a pretrained speech model that we do not train. We average one of its layers (1,024 numbers per clip) and train our own logistic regression on those numbers. We chose it because it is simple to explain and hard to overfit with a few thousand clips.

**Measure.** AI voices caught (recall) and real people wrongly flagged (false alarms), reported separately. Accuracy would hide which mistake is being made.

## Results

Our baseline (always answer "real") caught 0% of AI voices. Our model caught 70% on held-out test data and wrongly flagged 8% of real people. These are voices and generators it never met, heard through a phone line:

| Method | AI voices caught | Real people wrongly flagged |
|---|---|---|
| Baseline: always say "real" | 0% (0 of 1,090) | 0% (0 of 580) |
| Baseline: logistic regression on simple audio features | 53% | 19% |
| Version 1: real voices from one dataset | 97% | 53% |
| **CloneCheck: real voices from five datasets** | **70% (760)** | **8% (45)** |

![Results on never-seen voices and generators](results/test_never_seen.png)

- **Our improvement.** Version 1 caught 100% of fakes on its own data, which looked too good. It had learned what one dataset's recordings sound like and called everything else fake, including half of all real people. Training on five datasets cut false alarms from 53% to 8%.
- **What an alarm means.** In this test 94% of alarms were right. If only 1 call in 100 were an AI voice, 8% would be.
- **Groups.** Real people wrongly flagged: 3% in English and 6% in Russian; women 1 to 3%, men 2 to 5%. On dev data, speakers with Irish, Midlands and Welsh accents were flagged 17 to 22% of the time. That may be the accent or the unfamiliar dataset; we cannot separate them.

**Where it fails, and why we think so.**

- **Two clips, same request.** A real bank customer saying *"I need to freeze my card"* scored 99.8% AI. An AI voice (Bark) saying *"Freeze my card please"* scored 0%.
- **It flagged me.** It called 18 of my 30 laptop recordings AI. When we added background hiss, that fell to 1 of 30, and it also stopped catching the AI copy of my voice (19 of 30, down to 2). It had partly learned "a clean recording is AI", which a scammer could beat with street noise.
- **A second improvement removed that shortcut but did not fix it.** Version 3, trained with noise and echo on real and fake clips alike, catches 58% of new-generator fakes with or without noise (version 2: 70% without, 19% with). On a fresh phone recording of my voice it flagged the real me 10 of 30 times and caught 14 of 30 copies.

**The main takeaway.** CloneCheck caught 100% of AI voices from generators it trained on, 70% from generators it never met, and 30% (9 of 30) of copies of my own voice made from a phone memo. The further a call is from the training data, the less it catches, and a test on familiar generators hides this. Today the honest answer to "is that really your grandson?" is that this tool cannot tell you.

## What we'd do next

- **Train on voice-cloning tools and hold some out.** The training generators are ordinary text-to-speech; scammers use cloning tools.
- **Test on real phone calls and more people,** with recordings from family members who give permission.

Until then the safest advice needs no AI: hang up and call back on a number you already have.

## Sources

- **Datasets (real speech):** [FLEURS](https://huggingface.co/datasets/google/fleurs), [LibriSpeech](https://www.openslr.org/12), [VoxPopuli](https://github.com/facebookresearch/voxpopuli), [Golos](https://github.com/sberdevices/golos) ([10-hour subset](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd)), [RuLibriSpeech](https://huggingface.co/datasets/bond005/rulibrispeech), [English Dialects](https://huggingface.co/datasets/ylacombe/english_dialects), [SOVA RuDevices](https://huggingface.co/datasets/bond005/sova_rudevices), [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14), [AMI](https://huggingface.co/datasets/edinburghcstr/ami), [Podlodka Speech](https://huggingface.co/datasets/bond005/podlodka_speech). My own voice is the only personal data; nobody else was recorded.
- **Voice generators (fake speech):** macOS `say`, [Piper](https://github.com/OHF-Voice/piper1-gpl) with [piper-voices](https://huggingface.co/rhasspy/piper-voices), [MMS-TTS](https://huggingface.co/facebook/mms-tts-eng), [Silero](https://github.com/snakers4/silero-models), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [Bark](https://huggingface.co/suno/bark-small), [Chatterbox](https://github.com/resemble-ai/chatterbox), [Vosk TTS](https://github.com/alphacep/vosk-tts).
- **Pretrained models:** [XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m) for the detector's features; [Whisper small](https://huggingface.co/openai/whisper-small) only to find the sentences inside my phone memo.
- **Libraries:** [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers), [scikit-learn](https://scikit-learn.org/), [librosa](https://librosa.org/), [SciPy](https://scipy.org/), [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [soundfile](https://github.com/bastibe/python-soundfile), [matplotlib](https://matplotlib.org/), [Gradio](https://www.gradio.app/), [FFmpeg](https://ffmpeg.org/).
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used heavily as a coding assistant: it wrote most of the code and helped draft this write-up, working from my idea and direction.
- **Tutorials:** none. The write-up follows the hackathon's template and sample submission.
