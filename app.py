"""CloneCheck demo app.

Run:  python app.py      then open the local address it prints.

Record or upload a few seconds of a voice. CloneCheck says whether it sounds
like a real person or an AI-generated voice, and how sure it is.
"""
from __future__ import annotations

import gradio as gr

from clonecheck.predict import check

ADVICE = ("Whatever this says, if a caller asks for money: hang up and call the person back "
          "on a number you already have.")


def card(title: str, colour: str, detail: str) -> str:
    return (f"<div style='border-radius:12px;padding:18px 20px;border:2px solid {colour}'>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{colour}'>{title}</div>"
            f"<div style='margin-top:8px'>{detail}</div>"
            f"<div style='margin-top:12px;font-size:0.9rem;opacity:0.8'>{ADVICE}</div></div>")


def render(recording: str | None, phone: bool) -> str:
    if recording is None:
        return "<p>Record or upload a voice clip, then press Check.</p>"
    try:
        result = check(recording, as_phone_call=phone)
    except Exception as error:  # show the problem instead of a blank page
        return f"<p><b>Could not check this clip.</b> {error}</p>"
    if result["verdict"] == "too short":
        return card("Too short to judge", "#8a6d00", "CloneCheck needs at least one second of speech.")
    p, level = result["probability"], result["threshold"]
    detail = (f"AI-voice score: <b>{p:.0%}</b> (the alarm goes off above {level:.0%}). "
              f"Based on {result['seconds']:.1f} seconds of audio"
              + (", heard through a simulated phone line." if phone else "."))
    if result["verdict"] == "ai":
        return card("Sounds AI-generated", "#c0392b", detail)
    return card("Sounds like a real person", "#1a7f4b",
                detail + " This is not proof: CloneCheck misses some AI voices, "
                         "especially from voice generators it has never heard.")


with gr.Blocks(title="CloneCheck") as demo:
    gr.Markdown(
        "# CloneCheck\n"
        "**Is that really your grandson on the phone?** Scammers now use AI to copy a family "
        "member's voice. Record or upload a few seconds of a voice and CloneCheck estimates "
        "whether it is a real person or an AI-generated voice.\n\n"
        "*A student prototype, not financial advice. It makes mistakes in both directions.*")
    clip = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice clip")
    phone = gr.Checkbox(label="Hear it the way a phone line would carry it (simulated)", value=False)
    button = gr.Button("Check", variant="primary")
    output = gr.HTML()
    button.click(render, inputs=[clip, phone], outputs=output)

if __name__ == "__main__":
    demo.launch(server_port=7862)
