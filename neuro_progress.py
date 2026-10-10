"""
SM-2 spaced repetition engine for the neuro learning app.
Progress is persisted to neuro_progress.json in the project root.
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path

PROGRESS_FILE = Path("neuro_progress.json")

_DEFAULT_STATE = {
    "repetitions": 0,
    "easiness_factor": 2.5,
    "interval": 1,
    "last_review": None,
    "next_review": None,
    "history": [],
}


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with PROGRESS_FILE.open() as f:
            return json.load(f)
    return {"topics": {}}


def save_progress(data: dict) -> None:
    with PROGRESS_FILE.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def get_topic_state(progress: dict, topic_id: str) -> dict:
    return dict(progress.get("topics", {}).get(topic_id, _DEFAULT_STATE))


def score_to_quality(score: float) -> int:
    """Map 0-1.0 quiz score to SM-2 quality integer (0-5)."""
    if score < 0.20:
        return 0
    elif score < 0.40:
        return 1
    elif score < 0.60:
        return 2
    elif score < 0.70:
        return 3
    elif score < 0.85:
        return 4
    else:
        return 5


def update_topic(progress: dict, topic_id: str, score: float, session_date: date | None = None) -> dict:
    """Apply SM-2 update after a completed study session. Mutates and returns progress."""
    today = session_date or date.today()
    state = get_topic_state(progress, topic_id)
    q = score_to_quality(score)

    # SM-2 easiness factor update (clamped ≥ 1.3)
    ef = state["easiness_factor"] + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
    state["easiness_factor"] = max(1.3, round(ef, 4))

    if q < 3:
        # Incorrect — restart interval sequence
        state["repetitions"] = 0
        state["interval"] = 1
    else:
        reps = state["repetitions"]
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = math.ceil(state["interval"] * state["easiness_factor"])
        state["interval"] = interval
        state["repetitions"] = reps + 1

    state["last_review"] = today.isoformat()
    state["next_review"] = (today + timedelta(days=state["interval"])).isoformat()
    state.setdefault("history", []).append({
        "date": today.isoformat(),
        "score": round(score, 3),
        "quality": q,
    })

    progress.setdefault("topics", {})[topic_id] = state
    return progress


def get_due_topics(progress: dict, all_topic_ids: list[str]) -> list[str]:
    """Return topic IDs due for review today or overdue (seen at least once)."""
    today = date.today().isoformat()
    return [
        tid for tid in all_topic_ids
        if (nr := get_topic_state(progress, tid).get("next_review")) and nr <= today
    ]


def get_weak_topics(progress: dict, all_topic_ids: list[str], n: int = 10) -> list[str]:
    """Return up to n topics with lowest mastery (for rapid recall mode)."""
    def _weakness_score(tid: str) -> float:
        state = get_topic_state(progress, tid)
        if not state.get("last_review"):
            return 999.0  # never studied — highest priority
        history = state.get("history", [])
        if not history:
            return 5.0
        recent = history[-3:]
        avg_score = sum(h["score"] for h in recent) / len(recent)
        # Lower score → higher weakness; also weight by repetitions (fewer = weaker)
        return -avg_score + (1 / max(state["repetitions"], 0.5)) * 0.1

    sorted_ids = sorted(all_topic_ids, key=_weakness_score, reverse=True)
    return sorted_ids[:n]


def get_mastery_label(state: dict) -> str:
    """Return one of: not_started, struggling, learning, reviewing, mastered."""
    if not state.get("last_review"):
        return "not_started"
    reps = state.get("repetitions", 0)
    ef = state.get("easiness_factor", 2.5)
    if reps == 0:
        return "struggling"
    if reps <= 2:
        return "learning"
    if ef >= 2.0 and reps >= 3:
        return "mastered"
    return "reviewing"


def get_overall_stats(progress: dict, all_topic_ids: list[str]) -> dict:
    counts = {"not_started": 0, "struggling": 0, "learning": 0, "reviewing": 0, "mastered": 0}
    total_sessions = 0
    all_scores: list[float] = []
    for tid in all_topic_ids:
        state = get_topic_state(progress, tid)
        counts[get_mastery_label(state)] += 1
        for h in state.get("history", []):
            total_sessions += 1
            all_scores.append(h["score"])
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "counts": counts,
        "total_sessions": total_sessions,
        "avg_score": avg_score,
        "total_topics": len(all_topic_ids),
    }


# Display metadata for each mastery label
MASTERY_META: dict[str, dict] = {
    "not_started": {"emoji": "⬜", "label": "Not started", "color": "#666666"},
    "struggling":  {"emoji": "🟥", "label": "Struggling",  "color": "#cc3333"},
    "learning":    {"emoji": "🟨", "label": "Learning",    "color": "#ccaa00"},
    "reviewing":   {"emoji": "🟦", "label": "Due for review", "color": "#3366cc"},
    "mastered":    {"emoji": "🟩", "label": "Mastered",    "color": "#229933"},
}
