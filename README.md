# CloneCheck: is that really your grandson on the phone?

**Track:** AI + Finance
**Team:** Gosha (solo)

*This project is a prototype and is not financial advice.*

> Work in progress. This file currently holds the test plan, written and committed before any model was scored on the held-out test data. Results will be added below it.

## The idea in one sentence

CloneCheck helps a grandparent who gets a phone call from "a grandchild in trouble" asking for money, by taking a few seconds of the caller's voice and saying whether it sounds like a real person or an AI-generated voice.

## Test plan (fixed before testing)

**Question.** A detector can only be trained on the voice generators that exist today. A scammer will use one it has never heard, over a phone line. Does the detector still work then?

**Real voices.** [FLEURS](https://huggingface.co/datasets/google/fleurs): people reading sentences aloud, in English (US) and Russian. 2,172 recordings.

**Fake voices.** The same kind of sentences read by six AI voice generators that run on a laptop.

| Generator | Role | Why |
|---|---|---|
| macOS `say`, Piper, MMS | **train** | The detector learns from these three |
| Silero | **dev** | Never trained on. Used to choose settings (which model layer, which training recipe) |
| Kokoro, Bark, and any generator added later | **test** | Never trained on, never used to choose anything. Scored once at the end |

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

- All real voices come from one dataset. A detector could learn "sounds like a FLEURS recording" instead of "sounds human". Recordings of my own voice on my own microphone are the check for this.
- The phone line is simulated, not a real call.
- The fake voices read the same style of sentence as the real ones, which is not how a scam call sounds.

## Sources so far

- **Data:** [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google, CC BY 4.0).
- **Voice generators:** macOS `say`; [Piper](https://github.com/OHF-Voice/piper1-gpl) with [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices); [MMS-TTS](https://huggingface.co/facebook/mms-tts-eng) (Meta); [Silero](https://github.com/snakers4/silero-models); [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M); [Bark](https://huggingface.co/suno/bark-small) (Suno).
- **Pretrained model:** [XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (Meta).
- **Libraries:** PyTorch, Hugging Face Transformers, scikit-learn, librosa, SciPy, pandas, soundfile.
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used heavily as a coding assistant.
