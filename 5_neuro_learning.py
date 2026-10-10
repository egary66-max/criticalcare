"""
ACVIM Neurology Boards Learning App
Structured curriculum from intern to graduating resident, with spaced repetition,
active learning, AI study notes, and multi-format quizzing.

Run: python3 -m streamlit run 5_neuro_learning.py
"""

import os

import streamlit as st

st.set_page_config(
    page_title="Neuro Learning — ACVIM Boards",
    page_icon="🧠",
    layout="wide",
)


# ── API key ───────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    # 1. Streamlit Cloud secrets
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 2. Local .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip()
    # 3. Environment variable
    return os.environ.get("ANTHROPIC_API_KEY", "")


ANTHROPIC_API_KEY = _load_api_key()
os.environ.pop("ANTHROPIC_API_KEY", None)

if not ANTHROPIC_API_KEY:
    st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    st.stop()


# ── Deferred imports ──────────────────────────────────────────────────────────

import chromadb
from neuro_curriculum import PHASES, TOPICS, PHASE_MAP, TOPIC_MAP, PHASE_TOPICS
from neuro_progress import (
    load_progress, save_progress, get_topic_state, update_topic,
    get_due_topics, get_weak_topics, get_mastery_label, get_overall_stats,
    MASTERY_META,
)
import neuro_study as study

DB_FOLDER = "chroma_db"
ALL_TOPIC_IDS = [t["id"] for t in TOPICS]


# ── DB load ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_collection():
    client = chromadb.PersistentClient(path=DB_FOLDER)
    return client.get_collection("critical_care")


try:
    collection = _load_collection()
except Exception as e:
    st.error(f"Could not load database: {e}. Run `python3 2_build_database.py` first.")
    st.stop()


# ── Session state defaults ────────────────────────────────────────────────────

_TAB_LABELS = ["🗺️ Roadmap", "📚 Study", "🔄 Review", "⚡ Rapid Recall", "📊 Progress"]

_DEFAULTS = {
    "progress": None,                  # loaded from JSON
    "active_tab": "🗺️ Roadmap",        # current navigation tab
    "active_topic_id": None,           # current study/review topic
    "study_note": None,                # generated study note dict
    "questions": None,                 # {mc, fitb, fr, citations}
    "q_queue": [],                     # [(fmt, idx), ...]
    "q_idx": 0,                        # current position in queue
    "answers_log": [],                 # [{q, score, ...}, ...]
    "last_grade": None,                # feedback for last answered q
    "mc_page": 0,                      # pagination for MC block
    "session_phase": "note",           # note → quiz → summary
    "recall_questions": None,          # rapid recall q set
    "recall_topic_ids": [],            # which topics the recall is from
    "recall_idx": 0,
    "recall_log": [],
    "recall_last_grade": None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load progress from disk once per session
if st.session_state.progress is None:
    st.session_state.progress = load_progress()

progress = st.session_state.progress


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save():
    save_progress(st.session_state.progress)


def _reset_study():
    for k in ["active_topic_id", "study_note", "questions", "q_queue",
              "answers_log", "last_grade", "session_phase"]:
        st.session_state[k] = _DEFAULTS.get(k)
    st.session_state.q_idx = 0
    st.session_state.mc_page = 0


def _reset_recall():
    for k in ["recall_questions", "recall_topic_ids", "recall_log", "recall_last_grade"]:
        st.session_state[k] = _DEFAULTS.get(k)
    st.session_state.recall_idx = 0


def _build_q_queue(qs: dict) -> list:
    queue = []
    for i in range(len(qs.get("mc", []))):
        queue.append(("mc", i))
    for i in range(len(qs.get("fitb", []))):
        queue.append(("fitb", i))
    for i in range(len(qs.get("fr", []))):
        queue.append(("fr", i))
    return queue


def _mastery_badge(topic_id: str) -> str:
    state = get_topic_state(progress, topic_id)
    label = get_mastery_label(state)
    m = MASTERY_META[label]
    return m["emoji"]


def _due_count() -> int:
    return len(get_due_topics(progress, ALL_TOPIC_IDS))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Neuro Learning")
    st.markdown("*ACVIM Boards Prep*")
    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    selected = st.radio(
        "Navigate",
        _TAB_LABELS,
        index=_TAB_LABELS.index(st.session_state.active_tab),
        label_visibility="collapsed",
    )
    if selected != st.session_state.active_tab:
        st.session_state.active_tab = selected
        st.rerun()

    st.divider()
    stats = get_overall_stats(progress, ALL_TOPIC_IDS)
    due_n = _due_count()

    col_a, col_b = st.columns(2)
    col_a.metric("Topics mastered", stats["counts"]["mastered"])
    col_b.metric("Due for review", due_n)

    pct_done = (
        stats["counts"]["mastered"] + stats["counts"]["reviewing"] +
        stats["counts"]["learning"]
    ) / stats["total_topics"] * 100
    st.progress(int(pct_done), text=f"{int(pct_done)}% of curriculum started")

    st.divider()
    st.markdown("**Mastery legend**")
    for label, meta in MASTERY_META.items():
        st.markdown(f"{meta['emoji']} {meta['label']}")

    st.divider()
    if st.button("↩️ Reset current session", use_container_width=True):
        _reset_study()
        _reset_recall()
        st.rerun()


# ── Active tab ────────────────────────────────────────────────────────────────

_active = st.session_state.active_tab


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROADMAP
# ══════════════════════════════════════════════════════════════════════════════

if _active == "🗺️ Roadmap":
    st.title("Curriculum Roadmap")
    st.caption("25 topics · 4 phases · ACVIM Neurology Boards framework")

    for phase in PHASES:
        pid = phase["id"]
        topics_in_phase = PHASE_TOPICS.get(pid, [])

        # Phase header with progress bar
        phase_states = [get_mastery_label(get_topic_state(progress, t["id"])) for t in topics_in_phase]
        mastered_n = sum(1 for s in phase_states if s == "mastered")
        started_n = sum(1 for s in phase_states if s != "not_started")

        st.markdown(f"### {phase['icon']} {phase['name']}")
        st.caption(phase["description"])
        st.progress(
            mastered_n / len(topics_in_phase),
            text=f"{mastered_n}/{len(topics_in_phase)} mastered · {started_n} started"
        )

        # Topic grid — 2 columns
        cols = st.columns(2)
        for i, topic in enumerate(topics_in_phase):
            state = get_topic_state(progress, topic["id"])
            label = get_mastery_label(state)
            meta = MASTERY_META[label]
            reps = state.get("repetitions", 0)
            next_rev = state.get("next_review", "—")

            with cols[i % 2]:
                with st.container(border=True):
                    col_emoji, col_info = st.columns([1, 5])
                    col_emoji.markdown(f"## {meta['emoji']}")
                    with col_info:
                        st.markdown(f"**{topic['title']}**")
                        st.caption(
                            f"{meta['label']} · {reps} session(s) · next review: {next_rev}"
                            if reps > 0 else meta["label"]
                        )
                    if st.button(
                        "Study this topic →",
                        key=f"roadmap_study_{topic['id']}",
                        use_container_width=True,
                    ):
                        _reset_study()
                        st.session_state.active_topic_id = topic["id"]
                        st.session_state.session_phase = "note"
                        st.session_state.active_tab = "📚 Study"
                        st.rerun()

        st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STUDY MODE
# ══════════════════════════════════════════════════════════════════════════════

if _active == "📚 Study":
    st.title("Study Mode")

    # ── Topic selector ────────────────────────────────────────────────────────
    if st.session_state.active_topic_id is None:
        st.markdown("Select a topic below, or launch one from the Roadmap tab.")

        for phase in PHASES:
            st.markdown(f"**{phase['icon']} {phase['name']}**")
            for topic in PHASE_TOPICS.get(phase["id"], []):
                badge = _mastery_badge(topic["id"])
                if st.button(
                    f"{badge} {topic['title']}",
                    key=f"study_sel_{topic['id']}",
                    use_container_width=True,
                ):
                    _reset_study()
                    st.session_state.active_topic_id = topic["id"]
                    st.session_state.session_phase = "note"
                    st.rerun()
            st.write("")
        st.stop()

    topic = TOPIC_MAP[st.session_state.active_topic_id]
    state = get_topic_state(progress, topic["id"])
    label = get_mastery_label(state)

    st.markdown(f"### {topic['title']}")
    st.caption(
        f"{MASTERY_META[label]['emoji']} {MASTERY_META[label]['label']} · "
        f"{state.get('repetitions', 0)} prior sessions"
    )

    if st.button("← Choose a different topic", key="study_back"):
        _reset_study()
        st.rerun()

    st.divider()

    # ── PHASE: note ──────────────────────────────────────────────────────────
    if st.session_state.session_phase == "note":

        if st.session_state.study_note is None:
            with st.spinner("Generating study note…"):
                try:
                    note = study.generate_study_note(topic, collection, ANTHROPIC_API_KEY)
                    st.session_state.study_note = note
                except Exception as e:
                    st.error(f"Study note generation failed: {e}")
                    st.stop()

        note = st.session_state.study_note

        if note.get("knowledge_supplemented"):
            st.info(
                "ℹ️ This topic includes content supplemented from clinical knowledge "
                "where the textbook database has limited coverage. Facts marked "
                "*(supplemented)* should be verified with your primary sources.",
                icon="📌",
            )

        st.markdown(note["text"])

        if note.get("db_citations"):
            with st.expander("Database passages used"):
                for c in note["db_citations"]:
                    st.markdown(f"- {c}")

        st.divider()
        st.markdown("**Ready to test yourself?**")
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(
                "Start quiz (20 MC · 3 FITB · 1 FR) →",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Generating questions (20 MC + 3 FITB + 1 FR)…"):
                    try:
                        qs = study.generate_questions(topic, collection, ANTHROPIC_API_KEY)
                        st.session_state.questions = qs
                        st.session_state.q_queue = _build_q_queue(qs)
                        st.session_state.q_idx = 0
                        st.session_state.answers_log = []
                        st.session_state.last_grade = None
                        st.session_state.session_phase = "quiz"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Question generation failed: {e}")
        with col2:
            if st.button("Skip quiz → Next topic", use_container_width=True):
                _reset_study()
                st.rerun()

    # ── PHASE: quiz ──────────────────────────────────────────────────────────
    elif st.session_state.session_phase == "quiz":
        qs = st.session_state.questions
        queue = st.session_state.q_queue
        qidx = st.session_state.q_idx
        total = len(queue)
        done = qidx >= total

        # Progress bar
        st.progress(qidx / max(total, 1), text=f"Question {min(qidx + 1, total)} of {total}")

        # Show last grade feedback
        last = st.session_state.last_grade
        if last:
            score = last.get("score", 0)
            emoji = "✅" if score >= 0.7 else ("➗" if score > 0 else "❌")
            st.markdown(f"### {emoji} Score: {int(score * 100)}%")

            if last.get("feedback"):
                st.markdown(f"**Feedback:** {last['feedback']}")
            if last.get("correct_answer"):
                st.markdown(f"**Correct answer:** `{last['correct_answer']}`")
            if last.get("per_blank"):
                for i, b in enumerate(last["per_blank"], 1):
                    mark = "✅" if b["correct"] else "❌"
                    accepted_str = " / ".join(b["accepted"])
                    st.markdown(f"- Blank {i}: {mark} you wrote `{b['user_answer']}` — accepted: {accepted_str}")
            if last.get("missed_points"):
                st.markdown("**Missed points:**")
                for p in last["missed_points"]:
                    st.markdown(f"- {p}")
            if last.get("explanation"):
                with st.expander("Explanation"):
                    st.markdown(last["explanation"])
            if last.get("sample_answer"):
                with st.expander("Model answer"):
                    st.markdown(last["sample_answer"])

            next_label = "Next question →" if qidx < total else "See session summary →"
            if st.button(next_label, type="primary"):
                st.session_state.last_grade = None
                if done:
                    st.session_state.session_phase = "summary"
                st.rerun()
            st.stop()

        if done:
            st.session_state.session_phase = "summary"
            st.rerun()

        fmt, sub_idx = queue[qidx]
        st.markdown(f"**Question {qidx + 1} of {total}** · *{fmt.upper()}*")
        st.divider()

        if fmt == "mc":
            q = qs["mc"][sub_idx]
            st.markdown("#### Multiple Choice")
            st.markdown(q["stem"])
            choice = st.radio(
                "Select one answer:",
                q["options"],
                index=None,
                key=f"mc_{qidx}",
            )
            if st.button("Submit", type="primary", disabled=choice is None, key=f"mc_sub_{qidx}"):
                grade = study.grade_mc(q, q["options"].index(choice))
                st.session_state.answers_log.append({"q": "mc", **grade})
                st.session_state.last_grade = grade
                st.session_state.q_idx += 1
                st.rerun()

        elif fmt == "fitb":
            q = qs["fitb"][sub_idx]
            st.markdown("#### Fill in the Blanks")
            st.markdown(q["stem_with_blanks"])
            answers = []
            for i, _ in enumerate(q["accepted_answers"]):
                answers.append(
                    st.text_input(f"Blank {i + 1}", key=f"fitb_{qidx}_{i}")
                )
            if st.button("Submit", type="primary", key=f"fitb_sub_{qidx}"):
                grade = study.grade_fitb(q, answers)
                st.session_state.answers_log.append({"q": "fitb", **grade})
                st.session_state.last_grade = grade
                st.session_state.q_idx += 1
                st.rerun()

        elif fmt == "fr":
            q = qs["fr"][sub_idx]
            st.markdown("#### Free Response")
            st.markdown(q["stem"])
            st.caption("Write a complete clinical answer — be specific with drugs, doses, and reasoning.")
            user_text = st.text_area("Your answer:", height=220, key=f"fr_{qidx}")
            if st.button("Submit for grading", type="primary", key=f"fr_sub_{qidx}"):
                with st.spinner("Grading your answer…"):
                    try:
                        grade = study.grade_fr(q, user_text, ANTHROPIC_API_KEY)
                        st.session_state.answers_log.append({"q": "fr", **grade})
                        st.session_state.last_grade = grade
                        st.session_state.q_idx += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Grading failed: {e}")

    # ── PHASE: summary ────────────────────────────────────────────────────────
    elif st.session_state.session_phase == "summary":
        log = st.session_state.answers_log
        session_score = study.compute_session_score(log)

        emoji = "🎉" if session_score >= 0.85 else ("✅" if session_score >= 0.7 else "📈")
        st.markdown(f"## {emoji} Session complete — {int(session_score * 100)}%")

        # Per-format breakdown
        cols = st.columns(3)
        for col, fmt in zip(cols, ("mc", "fitb", "fr")):
            scores = [a["score"] for a in log if a.get("q") == fmt]
            if scores:
                avg = sum(scores) / len(scores)
                col.metric(fmt.upper(), f"{int(avg * 100)}%", f"{len(scores)} questions")

        st.divider()

        # SM-2 update
        updated_progress = update_topic(progress, topic["id"], session_score)
        st.session_state.progress = updated_progress
        _save()

        new_state = get_topic_state(updated_progress, topic["id"])
        next_rev = new_state.get("next_review", "?")
        interval = new_state.get("interval", 1)

        with st.container(border=True):
            st.markdown(f"**Spaced repetition scheduled:**")
            st.markdown(
                f"Next review in **{interval} day(s)** · {next_rev}  \n"
                f"Mastery: {MASTERY_META[get_mastery_label(new_state)]['emoji']} "
                f"{MASTERY_META[get_mastery_label(new_state)]['label']}"
            )

        # Chapter references from the study note
        note = st.session_state.study_note
        if note:
            with st.expander("📖 Recommended reading for this topic"):
                for ref in topic.get("read_refs", []):
                    st.markdown(f"- {ref}")
            if note.get("db_citations"):
                with st.expander("Database passages used in this session"):
                    for c in note["db_citations"]:
                        st.markdown(f"- {c}")
            if qs := st.session_state.questions:
                if qs.get("citations"):
                    with st.expander("Question source passages"):
                        for c in qs["citations"]:
                            st.markdown(f"- {c}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Study another topic →", type="primary", use_container_width=True):
                _reset_study()
                st.rerun()
        with col2:
            if st.button("Review study note again", use_container_width=True):
                st.session_state.session_phase = "note"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REVIEW (spaced repetition)
# ══════════════════════════════════════════════════════════════════════════════

if _active == "🔄 Review":
    st.title("Spaced Repetition Review")

    due_ids = get_due_topics(progress, ALL_TOPIC_IDS)

    if not due_ids:
        st.success("✅ Nothing due for review today. Keep up the momentum!")
        st.markdown("Topics appear here when the SM-2 algorithm schedules them for review.")

        # Show upcoming
        from datetime import date
        import datetime
        upcoming = []
        for tid in ALL_TOPIC_IDS:
            state = get_topic_state(progress, tid)
            nr = state.get("next_review")
            if nr and nr > date.today().isoformat():
                upcoming.append((nr, tid))
        upcoming.sort()
        if upcoming:
            st.markdown("**Upcoming reviews:**")
            for nr, tid in upcoming[:8]:
                t = TOPIC_MAP[tid]
                state = get_topic_state(progress, tid)
                badge = MASTERY_META[get_mastery_label(state)]["emoji"]
                st.markdown(f"- {badge} **{t['title']}** — {nr}")
    else:
        st.markdown(f"**{len(due_ids)} topic(s) due for review today.**")

        for tid in due_ids:
            t = TOPIC_MAP[tid]
            state = get_topic_state(progress, tid)
            label = get_mastery_label(state)
            meta = MASTERY_META[label]
            reps = state.get("repetitions", 0)

            with st.container(border=True):
                col_badge, col_info, col_btn = st.columns([1, 5, 2])
                col_badge.markdown(f"## {meta['emoji']}")
                with col_info:
                    st.markdown(f"**{t['title']}**")
                    st.caption(f"{reps} session(s) · {meta['label']}")
                with col_btn:
                    if st.button("Review →", key=f"review_{tid}", use_container_width=True):
                        _reset_study()
                        st.session_state.active_topic_id = tid
                        st.session_state.session_phase = "note"
                        st.session_state.active_tab = "📚 Study"
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAPID RECALL
# ══════════════════════════════════════════════════════════════════════════════

if _active == "⚡ Rapid Recall":
    st.title("Rapid Recall")
    st.caption(
        "10 MC questions drawn from your weakest topics. "
        "Fast-fire: no study notes, just answer. Builds retrieval strength."
    )

    # ── No session started ────────────────────────────────────────────────────
    if st.session_state.recall_questions is None:
        weak_ids = get_weak_topics(progress, ALL_TOPIC_IDS, n=5)
        if not weak_ids:
            st.info("Study at least one topic first to unlock Rapid Recall.")
        else:
            st.markdown("**Your weakest topics:**")
            for tid in weak_ids:
                t = TOPIC_MAP[tid]
                badge = _mastery_badge(tid)
                st.markdown(f"- {badge} {t['title']}")

            st.write("")
            if st.button("🎯 Start Rapid Recall (10 questions)", type="primary"):
                with st.spinner("Pulling questions from your weak areas…"):
                    try:
                        all_mc: list[dict] = []
                        import random
                        selected_ids = weak_ids[:5]

                        for tid in selected_ids:
                            t = TOPIC_MAP[tid]
                            qs = study.generate_questions(
                                t, collection, ANTHROPIC_API_KEY,
                                num_mc=2, num_fitb=0, num_fr=0
                            )
                            for q in qs.get("mc", []):
                                q["_topic_id"] = tid
                            all_mc.extend(qs.get("mc", []))

                        random.shuffle(all_mc)
                        all_mc = all_mc[:10]

                        st.session_state.recall_questions = all_mc
                        st.session_state.recall_topic_ids = selected_ids
                        st.session_state.recall_idx = 0
                        st.session_state.recall_log = []
                        st.session_state.recall_last_grade = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Recall generation failed: {e}")
        st.stop()

    # ── Session in progress ───────────────────────────────────────────────────
    all_mc = st.session_state.recall_questions
    ridx = st.session_state.recall_idx
    total_r = len(all_mc)
    recall_done = ridx >= total_r

    # Show last grade
    last_r = st.session_state.recall_last_grade
    if last_r:
        score = last_r.get("score", 0)
        emoji = "✅" if score >= 0.7 else "❌"
        st.markdown(f"### {emoji} {'Correct' if score >= 0.7 else 'Incorrect'}")
        st.markdown(f"**Correct answer:** `{last_r.get('correct_answer', '')}`")
        if last_r.get("explanation"):
            st.markdown(last_r["explanation"])
        topic_id = last_r.get("_topic_id")
        if topic_id:
            st.caption(f"Topic: {TOPIC_MAP[topic_id]['title']}")

        if recall_done:
            next_lbl = "See results →"
        else:
            next_lbl = f"Next question ({ridx + 1}/{total_r}) →"
        if st.button(next_lbl, type="primary"):
            st.session_state.recall_last_grade = None
            if recall_done:
                _reset_recall()
                st.rerun()
            st.rerun()
        st.stop()

    if recall_done:
        # Results
        rlog = st.session_state.recall_log
        avg_r = sum(a.get("score", 0) for a in rlog) / max(len(rlog), 1)
        st.markdown(f"## ⚡ Rapid Recall complete — {int(avg_r * 100)}%")
        correct_n = sum(1 for a in rlog if a.get("score", 0) >= 0.7)
        st.markdown(f"**{correct_n}/{len(rlog)} correct**")
        if st.button("Start another Rapid Recall", type="primary"):
            _reset_recall()
            st.rerun()
        st.stop()

    # Current question
    st.progress(ridx / total_r, text=f"Question {ridx + 1} of {total_r}")
    q = all_mc[ridx]
    topic_id = q.get("_topic_id")
    if topic_id:
        st.caption(f"Topic: {TOPIC_MAP[topic_id]['title']}")
    st.markdown(f"**{q['stem']}**")
    choice = st.radio("Select one:", q["options"], index=None, key=f"recall_mc_{ridx}")
    if st.button("Submit", type="primary", disabled=choice is None, key=f"recall_sub_{ridx}"):
        grade = study.grade_mc(q, q["options"].index(choice))
        grade["_topic_id"] = topic_id
        st.session_state.recall_log.append({"q": "mc", **grade})
        st.session_state.recall_last_grade = grade
        st.session_state.recall_idx += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PROGRESS
# ══════════════════════════════════════════════════════════════════════════════

if _active == "📊 Progress":
    st.title("Progress Dashboard")

    stats = get_overall_stats(progress, ALL_TOPIC_IDS)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sessions", stats["total_sessions"])
    col2.metric("Avg score", f"{int(stats['avg_score'] * 100)}%")
    col3.metric("Mastered", f"{stats['counts']['mastered']}/{stats['total_topics']}")
    col4.metric("Due today", _due_count())

    st.divider()

    # Mastery distribution bar chart
    import pandas as pd
    dist_data = {
        "Status": [MASTERY_META[k]["label"] for k in MASTERY_META],
        "Count": [stats["counts"][k] for k in MASTERY_META],
    }
    df_dist = pd.DataFrame(dist_data)
    st.bar_chart(df_dist.set_index("Status"))

    st.divider()

    # Per-topic detail table
    st.markdown("### All Topics")

    rows = []
    for topic in TOPICS:
        state = get_topic_state(progress, topic["id"])
        label = get_mastery_label(state)
        history = state.get("history", [])
        last_score = history[-1]["score"] if history else None
        rows.append({
            "Phase": PHASE_MAP[topic["phase"]]["name"].split(":")[0].strip(),
            "Topic": topic["title"],
            "Status": f"{MASTERY_META[label]['emoji']} {MASTERY_META[label]['label']}",
            "Sessions": state.get("repetitions", 0),
            "Last score": f"{int(last_score * 100)}%" if last_score is not None else "—",
            "Next review": state.get("next_review", "—"),
            "EF": round(state.get("easiness_factor", 2.5), 2),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Score history sparkline (if any data)
    all_history: list[dict] = []
    for tid in ALL_TOPIC_IDS:
        state = get_topic_state(progress, tid)
        for h in state.get("history", []):
            all_history.append({"date": h["date"], "score": h["score"], "topic": tid})

    if all_history:
        df_hist = pd.DataFrame(all_history).sort_values("date")
        df_hist["score_pct"] = df_hist["score"] * 100
        st.markdown("### Score History")
        st.line_chart(df_hist.set_index("date")["score_pct"], y_label="Score (%)")
