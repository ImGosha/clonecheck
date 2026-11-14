# CloneCheck: is that really your grandson on the phone?

**Track:** AI + Finance
**Team:** Gosha (solo)

*This project is a prototype and is not financial advice.*

> Work in progress. This file currently holds the test plan, written and committed before any model was scored on the held-out test data. Results will be added below it.

## The idea in one sentence

CloneCheck helps a grandparent who gets a phone call from "a grandchild in trouble" asking for money, by taking a few seconds of the caller's voice and saying whether it sounds like a real person or an AI-generated voice.

## Test plan (fixed before testing)

**Question.** A detector can only be trained on the voice generators that exist today. A scammer will use one it has never heard, over a phone line. Does the detector still work then?

**Real voices.** People speaking, from ten public datasets and my own recordings. Like the generators, the sources are split by role, so the final test includes real voices recorded in places the detector has never heard.

| Real source | Language | Role |
|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (read sentences) | English, Russian | **train** |
| [LibriSpeech](https://www.openslr.org/12) (audiobooks), [VoxPopuli](https://github.com/facebookresearch/voxpopuli) (parliament speeches) | English | **train** |
| [Golos](https://github.com/sberdevices/golos) (phrases spoken into phones), [RuLibriSpeech](https://huggingface.co/datasets/bond005/rulibrispeech) (audiobooks) | Russian | **train** |
| [English Dialects](https://huggingface.co/datasets/ylacombe/english_dialects), [SOVA RuDevices](https://huggingface.co/datasets/bond005/sova_rudevices) | English, Russian | **dev**: never trained on, used to choose settings |
| [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) (people phoning a bank), [AMI](https://huggingface.co/datasets/edinburghcstr/ami) (meetings), [Podlodka](https://huggingface.co/datasets/bond005/podlodka_speech) (podcast), my own voice | English, Russian | **test**: never trained on, never used to choose anything |

For the training sources, different speakers go to train, dev and test.

**Fake voices.** Eight AI voice generators that run on a laptop read out the same texts as the real clips, so real and fake clips say the same kind of thing.

| Generator | Role | Why |
|---|---|---|
| macOS `say`, Piper, MMS | **train** | The detector learns from these three |
| Silero | **dev** | Never trained on. Used to choose settings (which model layer, which training recipe) |
| Kokoro, Bark, Chatterbox, Vosk, and any generator added later | **test** | Never trained on, never used to choose anything. Scored once at the end |

> **Change of plan after the first dev run (before any test data was scored).** The first version trained on real voices from FLEURS only. On dev data it caught every fake, including the generator it had never heard, which looked too good. It also flagged 83% of real audiobook readers (LibriSpeech) as fake. It had learned to recognise FLEURS recordings, not human voices. The source roles above are the fix: real voices from several sources in training, and sources held out for dev and test in the same way as generators. The first plan and that dev run are in the commit history.

**Sentences.** No sentence appears in more than one of train, dev and test, so a model cannot pass by remembering words.

**Listening conditions.** Every clip is scored twice: as recorded ("clean"), and after a simulated landline phone call ("phone": 300 to 3400 Hz, 8 kHz, mu-law).

**Same treatment for real and fake.** Every clip is trimmed of silence, cut to at most 10 seconds and set to the same loudness before any model sees it.

**Methods compared.**

1. Always say "real" (simplest baseline).
2. Logistic regression on 80 classic audio features (MFCCs).
3. Logistic regression on features from XLS-R, a pretrained multilingual speech model. This is CloneCheck.

**Alarm level.** Chosen on real dev recordings only, so that at most 5% of them are flagged. It is then frozen.

**What is reported.**

- Fakes caught, separately for every generator, for English and Russian, and for clean and phone audio.
- False alarms on real voices, separately for English and Russian and for male and female speakers.
- At least one improvement, chosen on dev and then measured on test.
- Examples of what it gets wrong.

**Known weak points of this plan, stated up front.**

- A detector can learn to recognise a dataset's recordings instead of human voices (the first dev run did exactly that). Holding out whole sources, plus recordings of my own voice on my own microphone, are the check for this.
- The phone line is simulated, not a real call.
- The fake voices read the same style of sentence as the real ones, which is not how a scam call sounds.

## Sources so far

- **Data (real speech):** [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google, CC BY 4.0); [LibriSpeech](https://www.openslr.org/12) (CC BY 4.0); [VoxPopuli](https://github.com/facebookresearch/voxpopuli) (Meta, CC0); [Golos](https://github.com/sberdevices/golos) (SberDevices), through the [10-hour crowd subset](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd); [RuLibriSpeech](https://huggingface.co/datasets/bond005/rulibrispeech); [English Dialects](https://huggingface.co/datasets/ylacombe/english_dialects) (Google, CC BY-SA 4.0); [SOVA RuDevices](https://huggingface.co/datasets/bond005/sova_rudevices); [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) (PolyAI, CC BY 4.0); [AMI](https://huggingface.co/datasets/edinburghcstr/ami) (CC BY 4.0); [Podlodka Speech](https://huggingface.co/datasets/bond005/podlodka_speech).
- **Voice generators:** macOS `say`; [Piper](https://github.com/OHF-Voice/piper1-gpl) with [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices); [MMS-TTS](https://huggingface.co/facebook/mms-tts-eng) (Meta); [Silero](https://github.com/snakers4/silero-models); [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M); [Bark](https://huggingface.co/suno/bark-small) (Suno); [Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI); [Vosk TTS](https://github.com/alphacep/vosk-tts).
- **Pretrained model:** [XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (Meta).
- **Libraries:** PyTorch, Hugging Face Transformers, scikit-learn, librosa, SciPy, pandas, soundfile.
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used heavily as a coding assistant.
