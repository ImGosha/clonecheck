"""Tables and charts for the notebook and the write-up.

Everything here reads the saved results in results/ (written by
scripts/evaluate_dev.py and scripts/evaluate_test.py), so the notebook shows
exactly the numbers the scripts produced.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .config import ENGINE_ROLE, REAL_SOURCE_ROLE, RESULTS
from .detect import load_clips, wilson

BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
METHOD_NAMES = {
    "always real": "Always say \"real\"",
    "mfcc, many + phone": "Simple audio features",
    "xlsr, one source": "Version 1 (one real dataset)",
    "xlsr, many + phone": "CloneCheck (many real datasets)",
}
NICE = {"say": "macOS say", "piper": "Piper", "mms": "MMS", "silero": "Silero", "kokoro": "Kokoro",
        "bark": "Bark", "chatterbox": "Chatterbox", "vosk": "Vosk",
        "chatterbox_clone": "Chatterbox copying my voice",
        "fleurs": "FLEURS", "librispeech": "LibriSpeech", "voxpopuli": "VoxPopuli", "golos": "Golos",
        "rulibrispeech": "RuLibriSpeech", "dialects": "English Dialects", "sova": "SOVA",
        "minds": "MInDS-14 (calls to a bank)", "ami": "AMI (meetings)", "podlodka": "Podlodka (podcast)",
        "own": "My own voice", "own2": "My own voice, second recording",
        "chatterbox_clone2": "Chatterbox copying my voice, second recording",
        "chatterbox_voices": "Chatterbox copying training speakers", "xtts_clone": "XTTS copying a voice"}


def clip_counts() -> pd.DataFrame:
    """How many clips of each kind there are, and what role each plays."""
    clips = load_clips()
    who = [r.source if r.label == "real" else r.engine for r in clips.itertuples()]
    role = [(REAL_SOURCE_ROLE if r.label == "real" else ENGINE_ROLE)[w]
            for r, w in zip(clips.itertuples(), who)]
    table = clips.assign(who=[NICE.get(w, w) for w in who], role=role).pivot_table(
        index=["label", "role", "who"], columns="split", values="clip_id", aggfunc="count",
        fill_value=0)
    return table.reindex(columns=["train", "dev", "test"]).sort_index(
        level=["label", "role"], ascending=[False, False])


def dev_table(features: str, recipe: str) -> pd.DataFrame:
    """Dev results for one recipe: % of clips flagged as fake."""
    table = pd.read_csv(RESULTS / "dev_recipes.csv")
    table = table[(table["features"] == features) & (table["recipe"] == recipe)]
    wide = table.assign(who=table["who"].map(lambda w: NICE.get(w, w))).pivot_table(
        index=["label", "who"], columns=["lang", "condition"], values="share", sort=False)
    return (100 * wide).round(0)


def test_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "test_results.csv")


def summary(condition: str = "phone") -> pd.DataFrame:
    """One row per method: fakes caught and real voices wrongly flagged, seen and never seen."""
    table = test_results()
    table = table[(table["who"] == "ALL") & (table["condition"] == condition)]
    rows = []
    for method, group in table.groupby("method", sort=False):
        row = {"Method": METHOD_NAMES.get(method, method)}
        for label, seen, title in [("fake", "never seen", "AI voices caught: new generators"),
                                   ("real", "never seen", "Real people flagged: new datasets"),
                                   ("fake", "seen in training", "AI voices caught: known generators"),
                                   ("real", "seen in training", "Real people flagged: known datasets")]:
            part = group[(group["label"] == label) & (group["seen"] == seen)].iloc[0]
            row[title] = f"{part['share']:.0%} ({int(part['flagged'])} of {int(part['clips'])})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def breakdown(method: str, label: str) -> pd.DataFrame:
    """% flagged for one method, by generator or dataset, language and listening condition."""
    table = test_results()
    table = table[(table["method"] == method) & (table["label"] == label) & (table["who"] != "ALL")]
    table = table.assign(who=table["who"].map(lambda w: NICE.get(w, w)),
                         cell=[f"{s:.0%} ({k}/{n})" for s, k, n in
                               zip(table["share"], table["flagged"], table["clips"])])
    return table.pivot_table(index=["seen", "who"], columns=["lang", "condition"], values="cell",
                             aggfunc="first", sort=False).fillna("")


def by_gender(method: str) -> pd.DataFrame:
    """False alarms on real test voices by speaker gender (only some datasets record it)."""
    scores = pd.read_csv(RESULTS / "test_scores.csv", keep_default_na=False)
    level = json.loads((RESULTS / "test_thresholds.json").read_text())[method]
    real = scores[(scores["label"] == "real") & scores["gender"].isin(["male", "female"])]
    rows = []
    for (gender, lang), group in real.groupby(["gender", "lang"]):
        for condition in ["clean", "phone"]:
            k = int((group[f"{method} | {condition}"] > level).sum())
            low, high = wilson(k, len(group))
            rows.append({"gender": gender, "lang": lang, "condition": condition, "clips": len(group),
                         "flagged": k, "share": f"{k / len(group):.0%}",
                         "95% range": f"{low:.0%} to {high:.0%}"})
    return pd.DataFrame(rows)


def mistakes(method: str, label: str, condition: str = "phone", n: int = 8) -> pd.DataFrame:
    """The most confident mistakes: real voices scored most fake, or fakes scored most real."""
    scores = pd.read_csv(RESULTS / "test_scores.csv", keep_default_na=False)
    clips = load_clips().set_index("clip_id")
    level = json.loads((RESULTS / "test_thresholds.json").read_text())[method]
    column = f"{method} | {condition}"
    part = scores[scores["label"] == label]
    wrong = part[part[column] > level] if label == "real" else part[part[column] <= level]
    wrong = wrong.sort_values(column, ascending=(label == "fake")).head(n)
    return pd.DataFrame({"who": [NICE.get(w, w) for w in wrong["who"]], "lang": wrong["lang"].values,
                         "score": wrong[column].round(3).values,
                         "text": [clips.loc[c, "text"][:90] for c in wrong["clip_id"]]},
                        index=wrong["clip_id"].values)


def headline_chart(path=None, condition: str = "phone"):
    """Two small charts: what each method does with voices and generators it has never met."""
    table = test_results()
    table = table[(table["who"] == "ALL") & (table["seen"] == "never seen")
                  & (table["condition"] == condition)]
    methods = [m for m in reversed(list(METHOD_NAMES)) if m in set(table["method"])]  # CloneCheck on top
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.7), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    panels = [("fake", "AI voices caught", "higher is better"),
              ("real", "Real people wrongly flagged", "lower is better")]
    for ax, (label, title, note) in zip(axes, panels):
        part = table[table["label"] == label].set_index("method").reindex(methods)
        values = 100 * part["share"].to_numpy()
        ax.set_facecolor(SURFACE)
        ax.barh(range(len(methods)), values, height=0.55, color=BLUE)
        for y, (value, low, high) in enumerate(zip(values, part["low"], part["high"])):
            ax.plot([100 * low, 100 * high], [y, y], color=INK, linewidth=1)
            ax.text(min(100 * high + 2, 101), y, f"{value:.0f}%", va="center", color=INK, fontsize=10)
        ax.set_title(f"{title}\n", loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.text(0, 1.03, note, transform=ax.transAxes, fontsize=9, color=MUTED)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=MUTED)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_NAMES[m] for m in methods], fontsize=10, color=INK)
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("Voices and voice generators the detector never met in training, heard through a phone line",
                 x=0.012, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if path:
        fig.savefig(path, dpi=200, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------- live tables
# The functions below score a trained detector directly (used in the notebook),
# instead of reading saved results.

def _with_roles(clips: pd.DataFrame) -> pd.DataFrame:
    who = [r.source if r.label == "real" else r.engine for r in clips.itertuples()]
    role = [(REAL_SOURCE_ROLE if r.label == "real" else ENGINE_ROLE)[w]
            for r, w in zip(clips.itertuples(), who)]
    return clips.assign(who=[NICE.get(w, w) for w in who],
                        met=["trained on" if x == "train" else "never trained on" for x in role])


def flag_table(detector, features: dict, clips: pd.DataFrame, counts: bool = False) -> pd.DataFrame:
    """% of clips flagged as AI, by generator or dataset, language and listening condition.

    For fake clips this is "caught"; for real clips it is "wrongly flagged".
    """
    from .detect import report as score
    table = score(detector, features, _with_roles(clips), ["label", "met", "who", "lang"])
    table["cell"] = ([f"{s:.0%} ({k}/{n})" for s, k, n in zip(table["share"], table["flagged"], table["clips"])]
                     if counts else [f"{s:.0%}" for s in table["share"]])
    wide = table.pivot_table(index=["label", "met", "who"], columns=["lang", "condition"],
                             values="cell", aggfunc="first", sort=False).fillna("")
    return wide.sort_index(level=["label", "met"], ascending=[False, False], sort_remaining=False)


def compare(methods: dict, features: dict, clips: pd.DataFrame, condition: str = "phone") -> pd.DataFrame:
    """One row per method: AI voices caught and real people wrongly flagged, new and known."""
    from .detect import report as score
    rows = []
    for name, detector in methods.items():
        table = score(detector, features, _with_roles(clips), ["label", "met"], conditions=(condition,))
        row = {"Method": name}
        for label, met, title in [("fake", "never trained on", "AI voices caught: new generators"),
                                  ("real", "never trained on", "Real people flagged: new datasets"),
                                  ("fake", "trained on", "AI voices caught: known generators"),
                                  ("real", "trained on", "Real people flagged: known datasets")]:
            part = table[(table["label"] == label) & (table["met"] == met)].iloc[0]
            row[title] = f"{part['share']:.0%} ({int(part['flagged'])} of {int(part['clips'])})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def by_condition(methods: dict, features: dict, clips: pd.DataFrame,
                 conditions=("clean", "phone", "noisy", "gated")) -> pd.DataFrame:
    """New generators caught and new real voices flagged, under each listening condition."""
    from .detect import report as score
    names = {"clean": "Clean", "phone": "Phone line", "noisy": "Background noise",
             "gated": "Noise suppression"}
    rows = []
    for name, detector in methods.items():
        table = score(detector, features, _with_roles(clips), ["label", "met"], conditions)
        table = table[table["met"] == "never trained on"]
        for label, title in [("fake", "AI voices caught (new generators)"),
                             ("real", "Real people flagged (new datasets)")]:
            part = table[table["label"] == label].set_index("condition")
            rows.append({"": title, "Method": name,
                         **{names[c]: f"{part.loc[c, 'share']:.0%} ({int(part.loc[c, 'flagged'])} of {int(part.loc[c, 'clips'])})"
                            for c in conditions}})
    return pd.DataFrame(rows).set_index(["", "Method"]).sort_index(level=0, sort_remaining=False)


def own_voice(methods: dict, features: dict, clips: pd.DataFrame,
              conditions=("clean", "phone")) -> pd.DataFrame:
    """My real voice wrongly flagged, and AI copies of it caught, out of the clips given."""
    from .detect import report as score
    rows = []
    for name, detector in methods.items():
        table = score(detector, features, clips, ["label"], conditions)
        for c in conditions:
            part = table[table["condition"] == c].set_index("label")
            rows.append({"Method": name, "Heard": {"clean": "as recorded", "phone": "phone line"}[c],
                         "My real voice wrongly flagged": f"{int(part.loc['real', 'flagged'])} of {int(part.loc['real', 'clips'])}",
                         "AI copy of my voice caught": f"{int(part.loc['fake', 'flagged'])} of {int(part.loc['fake', 'clips'])}"})
    return pd.DataFrame(rows).set_index(["Method", "Heard"])


def alarm_meaning(caught: float, false_alarm: float, shares=(0.5, 0.1, 0.01)) -> pd.DataFrame:
    """If the alarm goes off, how likely is it that the voice really is AI? (precision)

    That depends on how common AI voices are among the calls checked. `caught`
    is the share of AI voices flagged (recall); `false_alarm` is the share of
    real voices flagged.
    """
    rows = []
    for share in shares:
        right = caught * share
        wrong = false_alarm * (1 - share)
        rows.append({"AI voices among the calls checked": f"1 in {round(1 / share)}",
                     "Alarms that are really AI (precision)": f"{right / (right + wrong):.0%}",
                     "False alarms for every real catch": round(wrong / right, 1)})
    return pd.DataFrame(rows)
