"""
ER Overnight Quiz — Cohort Edition

A pared-down Streamlit app for the intern cohort. Exposes ONLY the Quiz Mode
behind a single shared password. No Q&A, no Specialist Consultation, no record
upload. Same RAG library as the local app.

Run locally:
    python3 -m streamlit run 4_quiz_app.py --server.port 8502

Deployed on Streamlit Community Cloud, configured via st.secrets:
    APP_PASSWORD       = "cohort-shared-password"
    ANTHROPIC_API_KEY  = "sk-ant-..."
"""

import os


# ── Page config FIRST (Streamlit requires this before any other st.* call) ────

import streamlit as st

st.set_page_config(page_title="ER Quiz — Intern Edition", page_icon="🎓", layout="centered")


# ── Secrets / env loading ─────────────────────────────────────────────────────
# In cloud, Streamlit injects st.secrets from the app's Secrets editor.
# Locally, we fall back to .env for convenience.

def _read_env_file_value(name: str):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return None
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None


def _secret(name: str, default=None):
    """Try st.secrets first (cloud); fall back to .env file; then env var."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return _read_env_file_value(name) or os.environ.get(name, default)


ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
APP_PASSWORD = _secret("APP_PASSWORD")

# ChromaDB's pydantic_settings rejects unknown env vars; same dance as 3_clinical_agent.
os.environ.pop("ANTHROPIC_API_KEY", None)

if not ANTHROPIC_API_KEY:
    st.error("Server misconfigured: ANTHROPIC_API_KEY is not set.")
    st.stop()
if not APP_PASSWORD:
    st.error("Server misconfigured: APP_PASSWORD is not set.")
    st.stop()


# ── Password gate ─────────────────────────────────────────────────────────────

if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.title("🎓 ER Quiz — Intern Edition")
    st.markdown(
        "A timed-by-difficulty study tool for ER overnight prep.  \n"
        "Generated cases drawn from the cohort's reference library."
    )
    with st.form("login", clear_on_submit=False):
        pw = st.text_input("Cohort password", type="password")
        submitted = st.form_submit_button("Enter", type="primary", use_container_width=True)
    if submitted:
        if pw == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password. Ask whoever shared the link.")
    st.stop()


# ── Deferred imports (only after auth — keeps unauthed page light) ────────────

import chromadb

from specialists import SPECIALIST_MAP
import quiz


# ── Session state defaults for the quiz ──────────────────────────────────────

for key, default in [
    ("collection", None),
    ("quiz_difficulty_tier", 1),
    ("quiz_case", None),
    ("quiz_questions", None),
    ("quiz_queue", []),
    ("quiz_queue_idx", 0),
    ("quiz_answers_log", []),
    ("quiz_specialist_id", None),
    ("quiz_image_bytes", None),
    ("quiz_image_mime", None),
    ("quiz_image_uploader_key", 0),
    ("quiz_last_grade", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── DB load ───────────────────────────────────────────────────────────────────

DB_FOLDER = "chroma_db"
if st.session_state.collection is None:
    if not os.path.exists(DB_FOLDER):
        st.error("Reference database not found in the deploy. Contact the admin.")
        st.stop()
    try:
        client = chromadb.PersistentClient(path=DB_FOLDER)
        st.session_state.collection = client.get_collection("critical_care")
    except Exception as e:
        st.error(f"Could not load database: {e}")
        st.stop()


# ── Title ─────────────────────────────────────────────────────────────────────

st.title("🎓 ER Overnight Quiz")
st.caption("Powered by the cohort's veterinary reference library")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_quiz_state():
    st.session_state.quiz_case = None
    st.session_state.quiz_questions = None
    st.session_state.quiz_queue = []
    st.session_state.quiz_queue_idx = 0
    st.session_state.quiz_answers_log = []
    st.session_state.quiz_specialist_id = None
    st.session_state.quiz_image_bytes = None
    st.session_state.quiz_image_mime = None
    st.session_state.quiz_image_uploader_key += 1
    st.session_state.quiz_last_grade = None


def _build_quiz_queue(qs: dict) -> list:
    """Flatten the question dict into an ordered queue of (format, sub_index) pairs."""
    queue = []
    for fmt in ("mc", "fitb", "fr"):
        for i in range(len(qs.get(fmt, []))):
            queue.append((fmt, i))
    return queue


# ── Quiz UI ───────────────────────────────────────────────────────────────────

tier = st.session_state.quiz_difficulty_tier
tier_info = quiz.DIFFICULTY_TIERS[tier]
st.markdown(
    f"**Current tier:** {tier}/5 — *{tier_info['label']}*  \n"
    f"Each case: {quiz.NUM_MC} MC · {quiz.NUM_FITB} fill-in-blank · {quiz.NUM_FR} free-response "
    "(plus optional image). Score ≥ 70% to advance a tier."
)

case = st.session_state.quiz_case
qs = st.session_state.quiz_questions

# --- No case loaded → Start button ---------------------------------------
if case is None:
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🎲 Start New Case", type="primary", use_container_width=True):
            sid = quiz.pick_random_specialist()
            with st.spinner(
                f"Generating a {tier_info['label'].lower()} case from "
                f"{SPECIALIST_MAP[sid]['name']} and {quiz.NUM_MC + quiz.NUM_FITB + quiz.NUM_FR} questions..."
            ):
                try:
                    new_case = quiz.generate_case(sid, tier, st.session_state.collection, ANTHROPIC_API_KEY)
                    new_qs = quiz.generate_questions(new_case, sid, tier, st.session_state.collection, ANTHROPIC_API_KEY)
                    st.session_state.quiz_specialist_id = sid
                    st.session_state.quiz_case = new_case
                    st.session_state.quiz_questions = new_qs
                    st.session_state.quiz_queue = _build_quiz_queue(new_qs)
                    st.session_state.quiz_queue_idx = 0
                    st.session_state.quiz_answers_log = []
                    st.session_state.quiz_last_grade = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Case generation failed: {e}")
    with col2:
        if st.button("🔄 Reset Tier to 1", use_container_width=True):
            st.session_state.quiz_difficulty_tier = 1
            _reset_quiz_state()
            st.rerun()

# --- Case loaded → show case + current question --------------------------
else:
    sid = st.session_state.quiz_specialist_id
    spec = SPECIALIST_MAP[sid]
    st.caption(f"{spec['icon']} {spec['name']} · Tier {tier}")

    with st.container(border=True):
        st.markdown(f"**Signalment:** {case['signalment']}")
        st.markdown(f"**Presenting complaint:** {case['presenting_complaint']}")
        st.markdown(f"**History:** {case['history']}")
        st.markdown(f"**Physical exam:** {case['physical_exam']}")
        if case.get("initial_diagnostics"):
            st.markdown(f"**Initial diagnostics:** {case['initial_diagnostics']}")

    queue = st.session_state.quiz_queue
    qidx = st.session_state.quiz_queue_idx
    total = len(queue)
    case_done = qidx >= total

    suggested = case.get("suggested_image")
    if (
        suggested
        and st.session_state.quiz_image_bytes is None
        and not qs.get("image")
        and not case_done
    ):
        label = quiz.MODALITY_LABELS.get(suggested.get("modality", ""), "image")
        st.info(
            f"💡 This case suggests a **{label}**: {suggested.get('prompt','')}. "
            "Upload one for a bonus image-interpretation question, or skip."
        )
        uploaded = st.file_uploader(
            f"Upload a {label} (png/jpg, ≤ 8 MB)",
            type=["png", "jpg", "jpeg"],
            key=f"quiz_img_upl_{st.session_state.quiz_image_uploader_key}",
            accept_multiple_files=False,
        )
        if uploaded is not None:
            img_bytes = uploaded.read()
            if len(img_bytes) > 8 * 1024 * 1024:
                st.error("Image too large (max 8 MB).")
            else:
                st.session_state.quiz_image_bytes = img_bytes
                st.session_state.quiz_image_mime = uploaded.type or "image/png"
                with st.spinner("Generating image question..."):
                    try:
                        iq = quiz.generate_image_question(
                            case,
                            img_bytes,
                            st.session_state.quiz_image_mime,
                            suggested["modality"],
                            ANTHROPIC_API_KEY,
                        )
                        qs["image"] = [iq]
                        st.session_state.quiz_questions = qs
                        st.session_state.quiz_queue.append(("image", 0))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Image question failed: {e}")

    st.divider()

    # Refresh after possible queue mutation above
    queue = st.session_state.quiz_queue
    total = len(queue)
    qidx = st.session_state.quiz_queue_idx
    case_done = qidx >= total
    fmt, sub_idx = (None, None) if case_done else queue[qidx]

    if not case_done:
        st.markdown(f"**Question {qidx + 1} of {total}** · *{fmt.upper()}*")

    last = st.session_state.quiz_last_grade
    if last:
        score = last.get("score", 0)
        color_emoji = "✅" if score >= 0.7 else ("➗" if score > 0 else "❌")
        st.markdown(f"### {color_emoji} Score: {int(score * 100)}%")
        if last.get("feedback"):
            st.markdown(f"**Feedback:** {last['feedback']}")
        if last.get("correct_answer"):
            st.markdown(f"**Correct answer:** {last['correct_answer']}")
        if last.get("per_blank"):
            for i, b in enumerate(last["per_blank"], 1):
                mark = "✅" if b["correct"] else "❌"
                st.markdown(f"- Blank {i}: {mark} you wrote `{b['user_answer']}` — accepted: {', '.join(b['accepted'])}")
        if last.get("missed_points"):
            st.markdown("**Missed points:**")
            for p in last["missed_points"]:
                st.markdown(f"- {p}")
        if last.get("sample_answer") and fmt in ("fr", "image"):
            with st.expander("Model answer"):
                st.markdown(last["sample_answer"])
        if last.get("explanation"):
            with st.expander("Explanation"):
                st.markdown(last["explanation"])
        if last.get("citations"):
            with st.expander("Sources"):
                for c in last["citations"]:
                    st.markdown(f"- {c}")

        next_label = "Next question →" if (qidx + 1) < total else "See case summary →"
        if st.button(next_label, type="primary"):
            st.session_state.quiz_queue_idx += 1
            st.session_state.quiz_last_grade = None
            st.rerun()

    elif fmt == "mc":
        q = qs["mc"][sub_idx]
        st.markdown("#### Multiple Choice")
        st.markdown(q["stem"])
        choice = st.radio("Pick one", q["options"], index=None, key=f"quiz_mc_choice_{qidx}")
        if st.button("Submit answer", type="primary", disabled=choice is None):
            grade = quiz.grade_multiple_choice(q, q["options"].index(choice))
            st.session_state.quiz_answers_log.append({"q": "mc", **grade})
            st.session_state.quiz_last_grade = grade
            st.rerun()

    elif fmt == "fitb":
        q = qs["fitb"][sub_idx]
        st.markdown("#### Fill in the Blanks")
        st.markdown(q["stem_with_blanks"])
        answers = []
        for i, _ in enumerate(q["accepted_answers"]):
            answers.append(st.text_input(f"Blank {i + 1}", key=f"quiz_fitb_{qidx}_{i}"))
        if st.button("Submit answers", type="primary"):
            grade = quiz.grade_fill_in_blank(q, answers)
            st.session_state.quiz_answers_log.append({"q": "fitb", **grade})
            st.session_state.quiz_last_grade = grade
            st.rerun()

    elif fmt == "fr":
        q = qs["fr"][sub_idx]
        st.markdown("#### Free Response")
        st.markdown(q["stem"])
        user_text = st.text_area("Your answer", height=180, key=f"quiz_fr_text_{qidx}")
        if st.button("Submit answer", type="primary"):
            with st.spinner("Grading..."):
                try:
                    grade = quiz.grade_free_response(q, user_text, ANTHROPIC_API_KEY)
                    st.session_state.quiz_answers_log.append({"q": "fr", **grade})
                    st.session_state.quiz_last_grade = grade
                    st.rerun()
                except Exception as e:
                    st.error(f"Grading failed: {e}")

    elif fmt == "image":
        q = qs["image"][sub_idx]
        st.markdown("#### Image Interpretation")
        st.image(
            st.session_state.quiz_image_bytes,
            caption=quiz.MODALITY_LABELS.get(q.get("modality", ""), "image"),
        )
        st.markdown(q["stem"])
        user_text = st.text_area("Your interpretation", height=180, key=f"quiz_img_text_{qidx}")
        if st.button("Submit answer", type="primary"):
            with st.spinner("Grading..."):
                try:
                    grade = quiz.grade_image_response(
                        q,
                        user_text,
                        st.session_state.quiz_image_bytes,
                        st.session_state.quiz_image_mime,
                        ANTHROPIC_API_KEY,
                    )
                    st.session_state.quiz_answers_log.append({"q": "image", **grade})
                    st.session_state.quiz_last_grade = grade
                    st.rerun()
                except Exception as e:
                    st.error(f"Grading failed: {e}")

    elif case_done:
        log = st.session_state.quiz_answers_log
        avg = sum(a.get("score", 0) for a in log) / max(len(log), 1)
        st.markdown(f"### Case complete · Average score: {int(avg * 100)}%")
        for q_type in ("mc", "fitb", "fr", "image"):
            scores = [a.get("score", 0) for a in log if a.get("q") == q_type]
            if scores:
                per_type = sum(scores) / len(scores)
                st.markdown(f"- **{q_type.upper()}** ({len(scores)} q): {int(per_type * 100)}%")

        with st.container(border=True):
            st.markdown(f"**Correct diagnosis:** {case.get('correct_diagnosis', '')}")
            st.markdown(f"**Correct treatment plan:** {case.get('correct_treatment_plan', '')}")
            if case.get("key_drugs"):
                st.markdown("**Key drugs:**")
                for d in case["key_drugs"]:
                    st.markdown(f"- **{d.get('name','')}** ({d.get('dose','')}) — *{d.get('moa','')}*")
            if case.get("teaching_pearl"):
                st.markdown(f"💡 **Teaching pearl:** {case['teaching_pearl']}")

        new_tier = quiz.advance_difficulty(tier, avg)
        if new_tier > tier:
            st.success(f"🎉 Tier up! → {new_tier}/5 ({quiz.DIFFICULTY_TIERS[new_tier]['label']})")
            st.session_state.quiz_difficulty_tier = new_tier

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("➡️ Next Case", type="primary", use_container_width=True):
                _reset_quiz_state()
                st.rerun()
        with col2:
            if st.button("End session", use_container_width=True):
                _reset_quiz_state()
                st.session_state.quiz_difficulty_tier = 1
                st.rerun()
