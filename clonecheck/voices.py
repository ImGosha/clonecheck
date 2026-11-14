"""AI voice generators used to make the fake half of the data.

Every generator takes a sentence and returns speech as a 16 kHz mono
waveform (a numpy array of numbers between -1 and 1), so real and fake clips
reach the detector in exactly the same format.

Generators (all free and run on this computer):
    say      the voices built into macOS
    piper    Piper, a small neural text-to-speech model
    mms      Meta's MMS text-to-speech models
    silero   Silero text-to-speech
    kokoro   Kokoro-82M (English only)
    bark     Suno's Bark, a generative audio model
    vosk     Vosk TTS (Russian only)
    chatterbox  Resemble AI's Chatterbox, a 2025 voice-cloning model
"""
from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16_000
CACHE = Path(__file__).resolve().parent.parent / "models_cache"

# voice names per generator and language ("en" or "ru")
VOICES = {
    "say": {"en": ["Samantha", "Fred", "Kathy", "Ralph", "Albert", "Junior"],
            "ru": ["Milena"]},
    "piper": {"en": ["en_US-lessac-medium", "en_US-ryan-medium", "en_US-amy-medium",
                     "en_US-joe-medium"],
              "ru": ["ru_RU-irina-medium", "ru_RU-denis-medium", "ru_RU-dmitri-medium",
                     "ru_RU-ruslan-medium"]},
    "mms": {"en": ["facebook/mms-tts-eng"], "ru": ["facebook/mms-tts-rus"]},
    "silero": {"en": ["en_0", "en_1", "en_2", "en_3", "en_4", "en_5"],
               "ru": ["aidar", "baya", "kseniya", "xenia", "eugene"]},
    "kokoro": {"en": ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george"],
               "ru": []},
    "bark": {"en": [f"v2/en_speaker_{i}" for i in range(6)],
             "ru": [f"v2/ru_speaker_{i}" for i in range(6)]},
    "vosk": {"en": [], "ru": ["0", "1", "2", "3", "4"]},
    # Chatterbox needs older libraries, so it runs from its own environment
    # (scripts/generate_chatterbox.py), not through speak() below.
    "chatterbox": {"en": ["default"], "ru": ["default"]},
}


def to_16k(wave: np.ndarray, rate: int) -> np.ndarray:
    """Mono, 16 kHz, float32."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != RATE:
        g = np.gcd(rate, RATE)
        wave = resample_poly(wave, RATE // g, rate // g).astype(np.float32)
    return wave


# ------------------------------------------------------------------ macOS say
def _say(text: str, voice: str, lang: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = f"{tmp}/a.aiff", f"{tmp}/a.wav"
        Path(f"{tmp}/a.txt").write_text(text)   # from a file, so odd punctuation cannot confuse it
        subprocess.run(["say", "-v", voice, "-o", aiff, "-f", f"{tmp}/a.txt"], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff,
                        "-ar", str(RATE), "-ac", "1", wav], check=True)
        wave, rate = sf.read(wav, dtype="float32")
    return to_16k(wave, rate)


# ---------------------------------------------------------------------- Piper
@lru_cache(maxsize=None)
def _piper_voice(name: str):
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice
    locale, speaker, quality = name.split("-")
    folder = f"{locale[:2]}/{locale}/{speaker}/{quality}"
    paths = [hf_hub_download("rhasspy/piper-voices", f"{folder}/{name}{ext}",
                             cache_dir=CACHE) for ext in (".onnx", ".onnx.json")]
    return PiperVoice.load(paths[0], config_path=paths[1])


def _piper(text: str, voice: str, lang: str) -> np.ndarray:
    model = _piper_voice(voice)
    chunks = list(model.synthesize(text))
    wave = np.concatenate([c.audio_float_array for c in chunks])
    return to_16k(wave, chunks[0].sample_rate)


# ------------------------------------------------------------------------ MMS
@lru_cache(maxsize=None)
def _mms_model(name: str):
    from transformers import VitsModel, VitsTokenizer
    return (VitsTokenizer.from_pretrained(name, cache_dir=CACHE),
            VitsModel.from_pretrained(name, cache_dir=CACHE).eval())


def _mms(text: str, voice: str, lang: str) -> np.ndarray:
    import torch
    tokenizer, model = _mms_model(voice)
    with torch.no_grad():
        wave = model(**tokenizer(text, return_tensors="pt")).waveform[0].numpy()
    return to_16k(wave, model.config.sampling_rate)


# --------------------------------------------------------------------- Silero
@lru_cache(maxsize=None)
def _silero_model(lang: str):
    from silero import silero_tts
    model, _ = silero_tts(language=lang, speaker="v3_en" if lang == "en" else "v4_ru")
    return model


def _silero(text: str, voice: str, lang: str) -> np.ndarray:
    wave = _silero_model(lang).apply_tts(text=text, speaker=voice, sample_rate=48_000)
    return to_16k(wave.numpy(), 48_000)


# --------------------------------------------------------------------- Kokoro
@lru_cache(maxsize=None)
def _kokoro_pipeline(code: str):
    from kokoro import KPipeline
    return KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")


def _kokoro(text: str, voice: str, lang: str) -> np.ndarray:
    pipeline = _kokoro_pipeline(voice[0])  # "a" American, "b" British
    parts = [np.asarray(audio) for _, _, audio in pipeline(text, voice=voice)]
    return to_16k(np.concatenate(parts), 24_000)


# ----------------------------------------------------------------------- Bark
@lru_cache(maxsize=None)
def _bark_model():
    import torch
    from transformers import AutoProcessor, BarkModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return (AutoProcessor.from_pretrained("suno/bark-small", cache_dir=CACHE),
            BarkModel.from_pretrained("suno/bark-small", cache_dir=CACHE).to(device).eval(), device)


def _bark(text: str, voice: str, lang: str) -> np.ndarray:
    import torch
    processor, model, device = _bark_model()
    inputs = processor(text, voice_preset=voice, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        wave = model.generate(**inputs, do_sample=True, pad_token_id=10000)
    return to_16k(wave[0].cpu().numpy(), model.generation_config.sample_rate)


# ----------------------------------------------------------------------- Vosk
@lru_cache(maxsize=None)
def _vosk_synth():
    from vosk_tts import Model, Synth
    return Synth(Model(model_name="vosk-model-tts-ru-0.9-multi"))


def _vosk(text: str, voice: str, lang: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        for odd, plain in {"–": "-", "—": "-", "…": ".", "«": "", "»": "", "“": "", "”": ""}.items():
            text = text.replace(odd, plain)     # Vosk only knows plain punctuation
        _vosk_synth().synth(text, f"{tmp}/a.wav", speaker_id=int(voice))
        wave, rate = sf.read(f"{tmp}/a.wav", dtype="float32")
    return to_16k(wave, rate)


ENGINES = {"say": _say, "piper": _piper, "mms": _mms, "silero": _silero,
           "kokoro": _kokoro, "bark": _bark, "vosk": _vosk}


def speak(engine: str, voice: str, lang: str, text: str) -> np.ndarray:
    """Generate one clip. Returns a 16 kHz mono float32 waveform."""
    return ENGINES[engine](text, voice, lang)
