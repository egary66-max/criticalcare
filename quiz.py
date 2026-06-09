"""
ER Overnight Quiz / Study Mode.

Generates mock ER cases via the existing specialist pipeline, then quizzes the
user with one multiple choice, one fill-in-blank, and one free-response question
per case (plus an optional 4th image-based question when the user uploads a
radiograph, ECG, or POCUS still).

Public API:
    ER_SPECIALIST_IDS, DIFFICULTY_TIERS
    generate_case(specialist_id, tier, collection, api_key) -> dict
    generate_questions(case, specialist_id, tier, collection, api_key) -> dict
    grade_multiple_choice(question, user_index) -> dict
    grade_fill_in_blank(question, user_answers) -> dict
    grade_free_response(question, user_answer, api_key) -> dict
    generate_image_question(case, image_bytes, mime_type, modality, api_key) -> dict
    grade_image_response(question, user_answer, image_bytes, mime_type, api_key) -> dict
    advance_difficulty(current_tier, case_score) -> int
"""

import base64
import json
import random
import re
from typing import Optional

import anthropic

from specialists import SPECIALIST_MAP, build_passages_context, search_for_specialist


MODEL = "claude-sonnet-4-6"


# Specialists whose domains commonly come up on overnight ER shifts.
ER_SPECIALIST_IDS = [
    "emergency_triage",
    "cardiovascular",
    "respiratory",
    "fluids_electrolytes",
    "hematology",
    "cc_neurology",
    "infectious_disease",
    "renal",
    "gi_only",
    "endocrine",
    "toxicology",
    "tox_drugs_chemicals",
    "tox_plants_biologicals",
    "analgesia_sedation",
    "immune_mediated",
    "antimicrobial_pharmacology",
    "clinpath_hematology",
    "clinpath_body_fluids",
    "us_pocus_procedures",
    "radiology_thorax",
    "radiology_abdomen",
    "ophthalmology",
]


DIFFICULTY_TIERS = {
    1: {
        "label": "Intern Day 1",
        "description": (
            "A classic, textbook presentation of a common ER condition. "
            "Signalment, history, and exam findings point clearly to a single "
            "likely diagnosis. Treatment is standard. No surprises."
        ),
    },
    2: {
        "label": "Intern + One Twist",
        "description": (
            "A textbook presentation, but include ONE complication: an unusual "
            "concurrent disease, a drug interaction, atypical signalment for "
            "the condition, or a comorbidity that changes the standard plan."
        ),
    },
    3: {
        "label": "Ambiguous Presentation",
        "description": (
            "Multiple plausible differentials based on initial presentation. "
            "The case requires reasoning through which diagnostic tests would "
            "discriminate, and prioritizing among ~3 reasonable DDx."
        ),
    },
    4: {
        "label": "Multi-System Critical",
        "description": (
            "A multi-system critical patient (e.g. polytrauma, DKA with sepsis, "
            "GDV with arrhythmia). Multiple problems require prioritization. "
            "Treatment for one problem may worsen another."
        ),
    },
    5: {
        "label": "Resident-Level Zebra",
        "description": (
            "Either an uncommon condition with subtle clues, OR a common "
            "condition at a critical decision point requiring expert-level "
            "judgement (massive transfusion thresholds, when to call surgery "
            "at 3am, when NOT to treat aggressively)."
        ),
    },
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Robustly extract a JSON object from a model response."""
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)


def _client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def _normalize_blank(text: str) -> str:
    """Lowercase + strip punctuation/whitespace for fuzzy FITB matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ── Case generation ───────────────────────────────────────────────────────────

CASE_GEN_INSTRUCTION = """You are generating a realistic teaching case for a small-animal ER intern.

DIFFICULTY TIER: {tier} — {tier_label}
{tier_description}

Generate ONE case that falls clearly within your specialty. Ground all clinical \
details in the textbook passages provided below. Return ONLY valid JSON matching \
this exact schema (no markdown, no commentary):

{{
  "signalment": "e.g. 7yr MN Labrador Retriever, 32 kg",
  "presenting_complaint": "1 sentence chief complaint",
  "history": "2-4 sentences of relevant history",
  "physical_exam": "Vitals + key exam findings as a short paragraph",
  "initial_diagnostics": "PCV/TS, BG, lactate, iSTAT, or other point-of-care findings",
  "correct_diagnosis": "The diagnosis the intern should arrive at",
  "correct_treatment_plan": "Stepwise initial management",
  "key_drugs": [
    {{"name": "drug name", "moa": "mechanism of action in 1 sentence", "dose": "dose with route + frequency"}}
  ],
  "teaching_pearl": "One memorable takeaway for the intern",
  "suggested_image": null
}}

For "suggested_image": if the case would naturally come with imaging the intern \
must interpret on shift (thoracic or abdominal rad, ECG / rhythm strip, or \
point-of-care ultrasound still), set it to:
  {{"modality": "radiograph" | "ecg" | "pocus", "prompt": "what to upload, e.g. 'lateral thoracic rad'"}}
Otherwise set "suggested_image" to null. Don't force an image if the case doesn't need one.
"""


def _call_specialist_for_json(
    specialist_id: str,
    instruction: str,
    query: str,
    collection,
    api_key: str,
    max_tokens: int = 1500,
) -> dict:
    """
    Shared helper: retrieve specialist-specific passages, build a system prompt
    that combines the specialist persona with the instruction, and parse JSON.
    """
    specialist = SPECIALIST_MAP[specialist_id]
    passages = search_for_specialist(query, specialist, collection, n=5, ref_n=2)
    passages_context = build_passages_context(passages)

    system_prompt = (
        f"{specialist['system_prompt']}\n\n"
        f"---\n\n{instruction}"
    )
    user_msg = f"{passages_context}\n\nPRODUCE THE JSON NOW."

    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text
    result = _parse_json(raw)
    result["_citations"] = [
        f"{p['book']} — {p['chapter']}" for p in passages
    ]
    return result


def generate_case(specialist_id: str, tier: int, collection, api_key: str) -> dict:
    """Generate a single teaching case grounded in the specialist's textbook chunks."""
    tier_info = DIFFICULTY_TIERS[tier]
    instruction = CASE_GEN_INSTRUCTION.format(
        tier=tier,
        tier_label=tier_info["label"],
        tier_description=tier_info["description"],
    )
    # The query string drives the RAG retrieval — anchor it on the specialist's
    # domain plus the difficulty so passages reflect the right depth.
    query = f"common ER presentations and management {SPECIALIST_MAP[specialist_id]['name']}"
    case = _call_specialist_for_json(
        specialist_id, instruction, query, collection, api_key, max_tokens=1800
    )
    return case


# ── Question generation ───────────────────────────────────────────────────────

QUESTION_GEN_INSTRUCTION = """You are writing teaching questions for a small-animal ER intern based on a case you generated.

DIFFICULTY TIER: {tier} — {tier_label}

Produce exactly THREE questions about the case below. Each question targets a different angle:
  - "mc"   : a 4-option multiple choice testing pathophysiology or mechanism.
  - "fitb" : a fill-in-the-blank testing a key drug (MOA, dose, or class).
  - "fr"   : a free-response testing clinical decision-making (next step, when to act, what to monitor).

Return ONLY valid JSON in this exact schema (no markdown, no commentary):

{{
  "mc": {{
    "stem": "the question",
    "options": ["A …", "B …", "C …", "D …"],
    "answer_index": 0,
    "explanation": "1-3 sentences explaining why the correct answer is correct and why distractors are not"
  }},
  "fitb": {{
    "stem_with_blanks": "Drug X is the ___ of choice for ___ at a dose of ___ mg/kg.",
    "accepted_answers": [["drug-of-choice-synonyms"], ["condition", "synonym"], ["0.5", "0.5 mg/kg"]],
    "explanation": "1-3 sentences explaining the answer"
  }},
  "fr": {{
    "stem": "the question",
    "rubric": ["3-5 short bullet points of what a complete answer must contain"],
    "sample_answer": "A model answer that hits every rubric point in 3-6 sentences"
  }}
}}

Rules:
- "accepted_answers" is a list-of-lists: one inner list per blank, containing all forms you would accept (case-insensitive, punctuation-insensitive).
- The MC stem and the FR stem must NOT reveal the correct diagnosis if the intern has not yet been asked for it.
- Make each question genuinely useful for an overnight ER shift.
- Difficulty must match the tier: easy/classic at tier 1, expert nuance at tier 5.

CASE:
{case_json}
"""


def generate_questions(
    case: dict, specialist_id: str, tier: int, collection, api_key: str
) -> dict:
    """Generate {mc, fitb, fr} for a given case."""
    tier_info = DIFFICULTY_TIERS[tier]
    case_for_prompt = {k: v for k, v in case.items() if not k.startswith("_")}
    instruction = QUESTION_GEN_INSTRUCTION.format(
        tier=tier,
        tier_label=tier_info["label"],
        case_json=json.dumps(case_for_prompt, indent=2),
    )
    query = (
        f"{case.get('correct_diagnosis', '')} "
        f"{' '.join(d.get('name', '') for d in case.get('key_drugs', []))}"
    )
    qs = _call_specialist_for_json(
        specialist_id, instruction, query, collection, api_key, max_tokens=2000
    )
    # Attach citations to each question so the UI can render them.
    citations = qs.pop("_citations", [])
    for key in ("mc", "fitb", "fr"):
        if key in qs:
            qs[key].setdefault("citations", citations)
    return qs


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_multiple_choice(question: dict, user_index: int) -> dict:
    """Pure logic — no model call."""
    correct_idx = question["answer_index"]
    correct = user_index == correct_idx
    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "correct_answer": question["options"][correct_idx],
        "explanation": question.get("explanation", ""),
        "citations": question.get("citations", []),
    }


def grade_fill_in_blank(question: dict, user_answers: list) -> dict:
    """Case-insensitive, punctuation-insensitive set membership per blank."""
    accepted_lists = question["accepted_answers"]
    per_blank = []
    for i, accepted in enumerate(accepted_lists):
        user_answer = user_answers[i] if i < len(user_answers) else ""
        normalized_user = _normalize_blank(user_answer)
        normalized_accepted = {_normalize_blank(a) for a in accepted}
        # Partial credit if the user's answer contains any accepted token.
        is_correct = (
            normalized_user in normalized_accepted
            or any(
                a and a in normalized_user for a in normalized_accepted
            )
        )
        per_blank.append({
            "user_answer": user_answer,
            "accepted": accepted,
            "correct": is_correct,
        })
    correct_count = sum(1 for b in per_blank if b["correct"])
    score = correct_count / max(len(per_blank), 1)
    return {
        "score": score,
        "correct": score == 1.0,
        "per_blank": per_blank,
        "explanation": question.get("explanation", ""),
        "citations": question.get("citations", []),
    }


FR_GRADING_PROMPT = """You are a small-animal ER attending grading a brand-new intern's answer.
Grade leniently and constructively — partial credit for any reasoning that hits part of the rubric.
You are teaching, not gate-keeping.

Return ONLY valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "feedback": "1-3 sentences of warm, specific feedback. Always name at least one thing the intern did well, then the most important gap.",
  "missed_points": ["specific rubric points not covered"]
}}

QUESTION: {stem}

RUBRIC (the intern's answer should cover most of these):
{rubric}

SAMPLE STRONG ANSWER:
{sample}

INTERN'S ANSWER:
{user}
"""


def grade_free_response(question: dict, user_answer: str, api_key: str) -> dict:
    """Use Claude to grade a free-response answer leniently against a rubric."""
    prompt = FR_GRADING_PROMPT.format(
        stem=question["stem"],
        rubric="\n".join(f"- {p}" for p in question.get("rubric", [])),
        sample=question.get("sample_answer", ""),
        user=user_answer.strip() or "(no answer given)",
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    parsed = _parse_json(raw)
    parsed.setdefault("citations", question.get("citations", []))
    parsed.setdefault("sample_answer", question.get("sample_answer", ""))
    parsed["correct"] = parsed.get("score", 0) >= 0.7
    return parsed


# ── Image questions ───────────────────────────────────────────────────────────

MODALITY_LABELS = {
    "radiograph": "radiograph",
    "ecg": "ECG / rhythm strip",
    "pocus": "point-of-care ultrasound image",
}


IMAGE_QUESTION_INSTRUCTION = """You are an ER attending writing a teaching question about this {modality_label} \
for a small-animal ER intern.

Patient case for context:
{case_json}

Look at the image carefully. Then return ONLY valid JSON:
{{
  "stem": "A clear free-response question asking the intern to interpret the image in this clinical context. Ask for SPECIFIC findings and what they MEAN for the patient.",
  "rubric": ["3-5 short bullet points an excellent answer must hit, based on what you actually see in the image"],
  "sample_answer": "A 3-6 sentence model answer that names the abnormalities visible and ties them to management.",
  "modality": "{modality}"
}}
"""


def _image_content_block(image_bytes: bytes, mime_type: str) -> dict:
    data = base64.standard_b64encode(image_bytes).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime_type, "data": data},
    }


def generate_image_question(
    case: dict, image_bytes: bytes, mime_type: str, modality: str, api_key: str
) -> dict:
    """Generate one free-response question about a user-uploaded image."""
    case_for_prompt = {k: v for k, v in case.items() if not k.startswith("_")}
    prompt = IMAGE_QUESTION_INSTRUCTION.format(
        modality=modality,
        modality_label=MODALITY_LABELS.get(modality, modality),
        case_json=json.dumps(case_for_prompt, indent=2),
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=900,
        messages=[{
            "role": "user",
            "content": [
                _image_content_block(image_bytes, mime_type),
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _parse_json(response.content[0].text)


IMAGE_GRADING_PROMPT = """You are a small-animal ER attending grading an intern's interpretation of this image.
Grade leniently — partial credit for naming any finding that is actually visible.

Return ONLY valid JSON:
{{
  "score": <float 0.0 to 1.0>,
  "feedback": "1-3 sentences. Name one thing they got right, then the most important miss visible in the image.",
  "missed_points": ["specific visible findings the intern did not mention"]
}}

QUESTION: {stem}

RUBRIC (based on what is visible):
{rubric}

SAMPLE STRONG ANSWER:
{sample}

INTERN'S ANSWER:
{user}
"""


def grade_image_response(
    question: dict, user_answer: str, image_bytes: bytes, mime_type: str, api_key: str
) -> dict:
    """Grade an image-question free response with the image in context."""
    prompt = IMAGE_GRADING_PROMPT.format(
        stem=question["stem"],
        rubric="\n".join(f"- {p}" for p in question.get("rubric", [])),
        sample=question.get("sample_answer", ""),
        user=user_answer.strip() or "(no answer given)",
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                _image_content_block(image_bytes, mime_type),
                {"type": "text", "text": prompt},
            ],
        }],
    )
    parsed = _parse_json(response.content[0].text)
    parsed["correct"] = parsed.get("score", 0) >= 0.7
    parsed.setdefault("sample_answer", question.get("sample_answer", ""))
    return parsed


# ── Difficulty progression ────────────────────────────────────────────────────

def advance_difficulty(current_tier: int, case_score: float) -> int:
    """Bump tier by 1 (capped at 5) if the user averaged 70%+ on this case."""
    if case_score >= 0.7:
        return min(current_tier + 1, max(DIFFICULTY_TIERS.keys()))
    return current_tier


def pick_random_specialist(rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    return rng.choice(ER_SPECIALIST_IDS)
