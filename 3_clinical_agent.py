"""
Small Animal Critical Care — Clinical Decision Support Agent
Three modes:
  1. Case Builder  — synthesize uploaded records into structured case context
  2. Q&A           — textbook-grounded chat Q&A
  3. Specialist Consultation — multi-agent panel consultation
Run with:  streamlit run 3_clinical_agent.py
"""

import os

# ChromaDB's pydantic_settings reads ALL env vars and rejects unknown ones.
# Load our key first, then remove it before chromadb is imported.
def _load_api_key():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return os.environ.get("ANTHROPIC_API_KEY")

ANTHROPIC_API_KEY = _load_api_key()
os.environ.pop("ANTHROPIC_API_KEY", None)

import streamlit as st
import chromadb
import anthropic
from specialists import SPECIALISTS, SPECIALIST_MAP, run_full_consultation
import quiz

DB_FOLDER = "chroma_db"
N_RESULTS = 6


# ── Stage 1: Records Synthesis ─────────────────────────────────────────────────

SYNTHESIS_PROMPT = """You are a veterinary records-synthesis engine for a local clinical decision-support app.

Your job is to process uploaded veterinary records together with clinician manual input from the current visit and produce a structured, source-grounded case context for downstream textbook-informed reasoning.

Allowed sources at this stage:
1. Uploaded veterinary records (provided below)
2. Clinician manual input from the current encounter (provided below)

Your role:
- Extract and normalize patient-specific facts from uploaded records
- Merge them with clinician-entered current findings
- Preserve source attribution (note whether each fact comes from records or clinician input)
- Surface contradictions, contraindications, missing data, and uncertainty
- Do not perform freeform unsupported reasoning
- Do not invent patient facts
- Prefer omission over invention
- If a fact is unclear, label it [uncertain]
- If sources conflict, preserve the conflict explicitly
- Distinguish historical record facts from current clinician findings
- Preserve negative findings only if explicitly documented
- Mark fields as [unknown] when not available

Output the following structured case context packet:

## 1. Record-Derived Facts
Signalment, history, prior diagnoses, prior medications, allergies/adverse reactions, prior diagnostics/labs/imaging, prior assessments — sourced from uploaded records only.

## 2. Current Clinician Input
Physical exam findings, neurologic findings, current medications, and any other details entered by the clinician for this visit.

## 3. Merged Problem List
Consolidated active problems derived from both sources. Note source for each.

## 4. Contraindications / Red Flags
Any documented allergies, adverse reactions, organ dysfunction, or history that constrains treatment options.

## 5. Key Diagnostic Evidence
Most clinically significant lab values, imaging findings, or test results. Note source and date if available.

## 6. Contradictions / Ambiguities
Any conflicts between records and clinician input, or internal inconsistencies within the records.

## 7. Missing Critical Data
Important information not present in either source that would materially affect clinical reasoning.

## 8. Evidence Map
Brief table mapping each major clinical finding to its source (Record / Clinician Input / Both).

Every fact must remain attributable to its source. Do not apply textbook knowledge at this stage."""


# ── Stage 2: Q&A ──────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are a clinical decision support assistant for a veterinary small animal rotating intern. \
Your knowledge base contains multiple veterinary reference textbooks. \
You help with real clinical situations in the ICU and emergency room.

Guidelines:
- Be direct and clinically practical — the user is actively managing a patient
- Lead with the most important information first
- Include specific drug doses, monitoring parameters, and decision thresholds where relevant
- Distinguish between dogs and cats when there are important species differences
- Flag if something is outside your source material or if specialist consultation is strongly recommended
- Format answers clearly: use bullet points, bold key terms, and short paragraphs
- Always ground your answers in the retrieved book passages provided to you
- If the passages don't contain enough information, say so clearly rather than guessing
- If a patient case context has been provided, apply it directly — tailor doses, differentials, and \
recommendations to this specific patient"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_db_collection():
    client = chromadb.PersistentClient(path=DB_FOLDER)
    return client.get_collection("critical_care")


def search_book(query, collection, n=N_RESULTS):
    results = collection.query(query_texts=[query], n_results=n)
    passages = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        passages.append({
            "text": results["documents"][0][i],
            "book": meta.get("book", "Small Animal Critical Care Medicine"),
            "chapter": meta.get("chapter", ""),
        })
    return passages


def build_context(passages):
    ctx = "RELEVANT PASSAGES FROM VETERINARY REFERENCE BOOKS:\n\n"
    for i, p in enumerate(passages, 1):
        ctx += f"[Passage {i} — {p['book']} | {p['chapter']}]\n{p['text']}\n\n"
    return ctx


# ── Q&A agent ─────────────────────────────────────────────────────────────────

def ask_agent(user_question, collection, chat_history, case_context=""):
    passages = search_book(user_question, collection)
    book_context = build_context(passages)

    messages = [{"role": t["role"], "content": t["content"]} for t in chat_history]

    parts = []
    if case_context.strip():
        parts.append(f"PATIENT CASE CONTEXT (from uploaded records):\n{case_context}")
    parts.append(book_context)
    parts.append(f"CLINICIAN QUESTION:\n{user_question}")
    messages.append({"role": "user", "content": "\n\n".join(parts)})

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=QA_SYSTEM_PROMPT,
        messages=messages
    )
    answer = response.content[0].text
    sources = list({f"{p['book']} — {p['chapter']}" for p in passages})
    return answer, sources


# ── Records synthesis agent ───────────────────────────────────────────────────

def run_records_synthesis(record_texts, clinician_notes):
    combined = "\n\n---\n\n".join(record_texts) if record_texts else "(none uploaded)"
    clinician = clinician_notes.strip() or "(none provided)"
    msg = (
        f"UPLOADED VETERINARY RECORDS:\n\n{combined}\n\n"
        f"---\n\nCLINICIAN MANUAL INPUT (current visit):\n\n{clinician}\n\n"
        "Please produce the structured case context packet."
    )
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=SYNTHESIS_PROMPT,
        messages=[{"role": "user", "content": msg}]
    )
    return response.content[0].text


# ── File extraction ───────────────────────────────────────────────────────────

def extract_pdf(uploaded_file):
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def extract_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


# ── App setup ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Critical Care Assistant", page_icon="🐾", layout="wide")
st.title("Small Animal Critical Care Assistant")
st.caption("Powered by your veterinary reference library")

# Session state defaults
for key, default in [
    ("collection", None),
    ("chat_history", []),
    ("display_history", []),
    ("case_context", ""),
    ("consultation_result", None),
    ("quiz_difficulty_tier", 1),
    ("quiz_case", None),
    ("quiz_questions", None),
    ("quiz_current_q", "mc"),
    ("quiz_answers_log", []),
    ("quiz_specialist_id", None),
    ("quiz_image_bytes", None),
    ("quiz_image_mime", None),
    ("quiz_image_uploader_key", 0),
    ("quiz_last_grade", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load DB
if st.session_state.collection is None:
    if not os.path.exists(DB_FOLDER):
        st.error("Database not found. Run `python3 2_build_database.py` first.")
        st.stop()
    try:
        st.session_state.collection = get_db_collection()
    except Exception as e:
        st.error(f"Could not load database: {e}")
        st.stop()


# ── Sidebar: Case Builder ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Case Builder")
    st.markdown("Upload records + enter current findings to build a structured case context.")

    uploaded_files = st.file_uploader(
        "Upload records (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    clinician_notes = st.text_area(
        "Current visit findings",
        placeholder="e.g. 5yr MN DSH, mydriatic right eye, no PLR, no menace, 3rd eyelid elevation...",
        height=150,
    )

    if st.button("Build Case Context", type="primary", use_container_width=True):
        if not uploaded_files and not clinician_notes.strip():
            st.warning("Upload a record or enter findings.")
        else:
            with st.spinner("Synthesizing records..."):
                texts = []
                for f in uploaded_files:
                    raw = extract_pdf(f) if f.name.endswith(".pdf") else extract_txt(f)
                    texts.append(f"[Source: {f.name}]\n{raw}")
                try:
                    st.session_state.case_context = run_records_synthesis(texts, clinician_notes)
                    st.session_state.chat_history = []
                    st.session_state.display_history = []
                    st.session_state.consultation_result = None
                    st.success("Case context built.")
                except Exception as e:
                    st.error(f"Synthesis failed: {e}")

    if st.session_state.case_context:
        st.info("Patient case loaded.")
        if st.button("Clear case", use_container_width=True):
            st.session_state.case_context = ""
            st.session_state.chat_history = []
            st.session_state.display_history = []
            st.session_state.consultation_result = None
            st.rerun()

    st.divider()
    st.header("About the Specialists")
    for s in SPECIALISTS:
        st.markdown(f"**{s['icon']} {s['name']}**  \n{s['description']}")
        st.markdown("")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_qa, tab_consult, tab_quiz = st.tabs(["💬 Q&A", "👥 Specialist Consultation", "🎓 Quiz Mode"])


# ── TAB 1: Q&A ────────────────────────────────────────────────────────────────

with tab_qa:
    if st.session_state.case_context:
        with st.expander("Structured Case Context", expanded=False):
            st.markdown(st.session_state.case_context)
        st.divider()

    for turn in st.session_state.display_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("sources"):
                with st.expander("Sources"):
                    for src in turn["sources"]:
                        st.markdown(f"- {src}")

    if prompt := st.chat_input("Ask a clinical question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.display_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching textbook..."):
                answer, sources = ask_agent(
                    prompt,
                    st.session_state.collection,
                    st.session_state.chat_history[-6:],
                    case_context=st.session_state.case_context,
                )
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.display_history.append({
            "role": "assistant", "content": answer, "sources": sources
        })


# ── TAB 2: Specialist Consultation ────────────────────────────────────────────

with tab_consult:
    st.markdown(
        "Present a case to the specialist panel. The coordinator selects the most relevant "
        "specialists, each consults their section of the textbook independently, then a senior "
        "consultant synthesizes their input into a unified recommendation."
    )

    # Case input — pre-fill from Case Builder if available
    default_case = ""
    if st.session_state.case_context:
        default_case = st.session_state.case_context

    consult_case = st.text_area(
        "Case summary for consultation",
        value=default_case,
        placeholder=(
            "e.g. 5yr MN DSH presenting for right-sided cranial nerve deficits affecting CN II, III, "
            "and V. Signs since Dec 2025. Right eye mydriatic, no PLR, no menace, 3rd eyelid "
            "elevation, absent palpebral reflex, no facial sensation right side. No prior neuro hx."
        ),
        height=180,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        run_btn = st.button("Consult Specialist Panel", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear results", use_container_width=True):
            st.session_state.consultation_result = None
            st.rerun()

    if run_btn:
        if not consult_case.strip():
            st.warning("Enter a case summary first.")
        else:
            status = st.status("Running specialist consultation...", expanded=True)
            progress_messages = []

            def on_progress(stage, message):
                progress_messages.append(message)
                status.write(f"• {message}")

            try:
                result = run_full_consultation(
                    case_text=consult_case,
                    collection=st.session_state.collection,
                    api_key=ANTHROPIC_API_KEY,
                    progress_callback=on_progress,
                )
                st.session_state.consultation_result = result
                status.update(label="Consultation complete", state="complete")
            except Exception as e:
                status.update(label=f"Consultation failed: {e}", state="error")
                st.error(f"Error: {e}")

    # Display results
    result = st.session_state.consultation_result
    if result:
        # Routing info
        routing = result["coordinator"].get("routing_rationale", "")
        selected_names = [
            f"{SPECIALISTS[next(i for i, s in enumerate(SPECIALISTS) if s['id'] == item['id'])]['icon']} "
            f"{SPECIALISTS[next(i for i, s in enumerate(SPECIALISTS) if s['id'] == item['id'])]['name']}"
            for item in result["coordinator"].get("selected_specialists", [])
        ]
        st.info(f"**Panel:** {' · '.join(selected_names)}  \n**Routing:** {routing}")

        st.divider()

        # Synthesis — the main output
        st.subheader("Synthesis")
        st.markdown(result["synthesis"])

        st.divider()

        # Individual specialist reports
        st.subheader("Individual Specialist Reports")
        for out in result["specialists"]:
            s = out["specialist"]
            with st.expander(f"{s['icon']} {s['name']}"):
                st.markdown(f"**Question:** *{out['question']}*")
                st.markdown(out["answer"])
                if out["sources"]:
                    st.markdown("**Sources:** " + " · ".join(f"`{src}`" for src in out["sources"]))


# ── TAB 3: Quiz Mode ──────────────────────────────────────────────────────────

def _reset_quiz_state():
    st.session_state.quiz_case = None
    st.session_state.quiz_questions = None
    st.session_state.quiz_current_q = "mc"
    st.session_state.quiz_answers_log = []
    st.session_state.quiz_specialist_id = None
    st.session_state.quiz_image_bytes = None
    st.session_state.quiz_image_mime = None
    st.session_state.quiz_image_uploader_key += 1
    st.session_state.quiz_last_grade = None


def _next_q_after(current: str, image_loaded: bool) -> str:
    order = ["mc", "fitb", "fr", "image" if image_loaded else "done", "done"]
    idx = order.index(current)
    return order[idx + 1]


with tab_quiz:
    tier = st.session_state.quiz_difficulty_tier
    tier_info = quiz.DIFFICULTY_TIERS[tier]
    st.markdown(
        f"**Current tier:** {tier}/5 — *{tier_info['label']}*  \n"
        "Random ER specialist each case. Score ≥ 70% to advance a tier."
    )

    case = st.session_state.quiz_case
    qs = st.session_state.quiz_questions

    # --- No case loaded → Start button ---------------------------------------
    if case is None:
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🎲 Start New Case", type="primary", use_container_width=True):
                sid = quiz.pick_random_specialist()
                with st.spinner(f"Generating a {tier_info['label'].lower()} case from {SPECIALIST_MAP[sid]['name']}..."):
                    try:
                        new_case = quiz.generate_case(sid, tier, st.session_state.collection, ANTHROPIC_API_KEY)
                        new_qs = quiz.generate_questions(new_case, sid, tier, st.session_state.collection, ANTHROPIC_API_KEY)
                        st.session_state.quiz_specialist_id = sid
                        st.session_state.quiz_case = new_case
                        st.session_state.quiz_questions = new_qs
                        st.session_state.quiz_current_q = "mc"
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

        # Optional image upload — shown until user uploads or finishes the case
        suggested = case.get("suggested_image")
        if (
            suggested
            and st.session_state.quiz_image_bytes is None
            and "image" not in (qs or {})
            and st.session_state.quiz_current_q != "done"
        ):
            label = quiz.MODALITY_LABELS.get(suggested.get("modality", ""), "image")
            st.info(f"💡 This case suggests a **{label}**: {suggested.get('prompt','')}. "
                    "Upload one for a bonus image-interpretation question, or skip.")
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
                            qs["image"] = iq
                            st.session_state.quiz_questions = qs
                            st.rerun()
                        except Exception as e:
                            st.error(f"Image question failed: {e}")

        st.divider()

        current = st.session_state.quiz_current_q
        image_loaded = "image" in (qs or {})

        # Show last-grade feedback (sticky until user clicks Next)
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
            if last.get("sample_answer") and current in ("fr", "image"):
                with st.expander("Model answer"):
                    st.markdown(last["sample_answer"])
            if last.get("explanation"):
                with st.expander("Explanation"):
                    st.markdown(last["explanation"])
            if last.get("citations"):
                with st.expander("Sources"):
                    for c in last["citations"]:
                        st.markdown(f"- {c}")

            next_label = "Next question →" if _next_q_after(current, image_loaded) != "done" else "See case summary →"
            if st.button(next_label, type="primary"):
                st.session_state.quiz_current_q = _next_q_after(current, image_loaded)
                st.session_state.quiz_last_grade = None
                st.rerun()

        # --- Current question widgets (only when no pending grade) -----------
        elif current == "mc":
            q = qs["mc"]
            st.markdown(f"#### Multiple Choice")
            st.markdown(q["stem"])
            choice = st.radio("Pick one", q["options"], index=None, key="quiz_mc_choice")
            if st.button("Submit answer", type="primary", disabled=choice is None):
                grade = quiz.grade_multiple_choice(q, q["options"].index(choice))
                st.session_state.quiz_answers_log.append({"q": "mc", **grade})
                st.session_state.quiz_last_grade = grade
                st.rerun()

        elif current == "fitb":
            q = qs["fitb"]
            st.markdown(f"#### Fill in the Blanks")
            st.markdown(q["stem_with_blanks"])
            answers = []
            for i, _ in enumerate(q["accepted_answers"]):
                answers.append(st.text_input(f"Blank {i + 1}", key=f"quiz_fitb_{i}"))
            if st.button("Submit answers", type="primary"):
                grade = quiz.grade_fill_in_blank(q, answers)
                st.session_state.quiz_answers_log.append({"q": "fitb", **grade})
                st.session_state.quiz_last_grade = grade
                st.rerun()

        elif current == "fr":
            q = qs["fr"]
            st.markdown(f"#### Free Response")
            st.markdown(q["stem"])
            user_text = st.text_area("Your answer", height=180, key="quiz_fr_text")
            if st.button("Submit answer", type="primary"):
                with st.spinner("Grading..."):
                    try:
                        grade = quiz.grade_free_response(q, user_text, ANTHROPIC_API_KEY)
                        st.session_state.quiz_answers_log.append({"q": "fr", **grade})
                        st.session_state.quiz_last_grade = grade
                        st.rerun()
                    except Exception as e:
                        st.error(f"Grading failed: {e}")

        elif current == "image":
            q = qs["image"]
            st.markdown(f"#### Image Interpretation")
            st.image(st.session_state.quiz_image_bytes, caption=quiz.MODALITY_LABELS.get(q.get("modality", ""), "image"))
            st.markdown(q["stem"])
            user_text = st.text_area("Your interpretation", height=180, key="quiz_img_text")
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

        elif current == "done":
            log = st.session_state.quiz_answers_log
            avg = sum(a.get("score", 0) for a in log) / max(len(log), 1)
            st.markdown(f"### Case complete · Average score: {int(avg * 100)}%")
            for a in log:
                st.markdown(f"- **{a['q'].upper()}**: {int(a['score'] * 100)}%")

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
