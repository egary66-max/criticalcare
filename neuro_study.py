"""
Study note and question generation for the neuro learning app.

Each topic session generates:
  - An AI study note with explicit chapter citations
  - 20 MC + 3 FITB + 1 FR questions, grounded in DB passages
  - For topics marked knowledge_supplemented=True, Claude's training knowledge
    fills gaps the DB doesn't cover, with a visible disclaimer.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic

MODEL = "claude-sonnet-4-6"
NUM_MC = 20
NUM_FITB = 3
NUM_FR = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ── RAG retrieval ─────────────────────────────────────────────────────────────

def search_topic(query: str, collection, n: int = 12) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=n)
    passages = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        passages.append({
            "text": results["documents"][0][i],
            "book": meta.get("book", ""),
            "chapter": meta.get("chapter", ""),
        })
    return passages


def _passages_context(passages: list[dict]) -> str:
    ctx = "RELEVANT TEXTBOOK PASSAGES:\n\n"
    for i, p in enumerate(passages, 1):
        ctx += f"[Passage {i} — {p['book']} | {p['chapter']}]\n{p['text']}\n\n"
    return ctx


def _unique_citations(passages: list[dict]) -> list[str]:
    seen = set()
    out = []
    for p in passages:
        key = f"{p['book']} — {p['chapter']}"
        if p["book"] and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ── Study note generation ─────────────────────────────────────────────────────

_STUDY_NOTE_SYSTEM = """You are a board-certified veterinary neurologist writing high-yield study notes
for a resident preparing for ACVIM Neurology Boards. Be specific, practical, and clinically grounded.
Never be vague — every bullet must contain a concrete, testable fact."""

_STUDY_NOTE_PROMPT = """TOPIC: {title}
DESCRIPTION: {description}

Use the textbook passages below as your primary source. Where the passages do not cover the topic
adequately, draw on established veterinary neurology knowledge and mark that content with the tag:
*(supplemented — verify with primary source)*

Write a study note with EXACTLY these sections:

## Key Concepts
3-5 bullet points — the irreducible core facts. Include specific values, thresholds, breed names,
drug names with doses where relevant. **Bold** the single most critical fact per bullet.

## Clinical Application
2-3 bullets describing how this topic actually presents on the clinic floor. Include one common
pitfall or mistake that trips up residents.

## Differential Diagnosis Framework
Concise, prioritized approach: what to rule in/out first and why. Include specific differentiators
(CSF finding, MRI sign, breed, age, acuity).

## Must-Know Details (Boards Facts)
Tight bullet list of facts that appear on ACVIM Boards: specific numeric values, drug doses,
pathognomonic signs, prognostic figures, species differences. These are the details that
separate a pass from a fail.

## Chapter References
List the chapters the resident should read to solidify this topic. Format each as:
- **[Book] — [Chapter]**: [what to focus on in that reading]

Keep the total note under 700 words. No filler sentences.

{passages}
"""


def generate_study_note(topic: dict, collection, api_key: str) -> dict:
    passages = search_topic(topic["query_terms"], collection, n=12)
    passages_ctx = _passages_context(passages)
    db_citations = _unique_citations(passages)

    prompt = _STUDY_NOTE_PROMPT.format(
        title=topic["title"],
        description=topic["description"],
        passages=passages_ctx,
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=2200,
        system=_STUDY_NOTE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.content[0].text,
        "db_citations": db_citations,
        "knowledge_supplemented": topic.get("knowledge_supplemented", False),
    }


# ── Question generation ───────────────────────────────────────────────────────

_MC_PROMPT = """You are a veterinary neurology ACVIM Boards examiner.

TOPIC: {title}
DESCRIPTION: {description}

Write EXACTLY {n} four-option multiple-choice questions. Cover DIFFERENT angles — no repeated themes:
- Pathophysiology / mechanism
- Clinical signs and their neuroanatomical basis
- Differential discrimination (what finding rules this in/out)
- Drug MOA, dose, monitoring, or contraindication
- Diagnostic test selection or interpretation (MRI signal, CSF findings, EMG)
- Prognosis and prognostic factors
- Surgical vs medical decision criteria
- Species/breed/age differences
- Complications and escalation decisions
- Boards-level integration (multi-step reasoning)

Return ONLY valid JSON — no markdown, no commentary:
{{
  "mc": [
    {{
      "stem": "clinical scenario + question (scenario-based preferred over pure recall)",
      "options": ["A …", "B …", "C …", "D …"],
      "answer_index": 0,
      "explanation": "2-3 sentences: why correct, why each distractor is wrong, key fact to remember."
    }}
  ]
}}

Rules:
- Distractors must be clinically plausible, not obviously wrong
- Mix foundational and integration-level difficulty
- Include specific values/doses/thresholds where relevant

TEXTBOOK PASSAGES:
{passages}
"""

_FITB_PROMPT = """You are a veterinary neurology ACVIM Boards examiner.

TOPIC: {title}
DESCRIPTION: {description}

Write EXACTLY {n} fill-in-the-blank questions. Each must test a DIFFERENT high-yield, specific fact:
- Exact drug name + dose + route + frequency
- Therapeutic drug level (numeric range)
- Critical threshold (e.g., number of seizures that triggers treatment)
- Prognostic figure (% recovery with/without surgery)
- Pathognomonic sign name
- Specific diagnostic test or MRI finding name

Return ONLY valid JSON:
{{
  "fitb": [
    {{
      "stem_with_blanks": "The therapeutic serum phenobarbital level in dogs is ___ to ___ mcg/mL.",
      "accepted_answers": [["20"], ["40"]],
      "explanation": "1-2 sentences: the answer in clinical context."
    }}
  ]
}}

Rules:
- 2-4 blanks per question
- "accepted_answers" is a list-of-lists, one inner list per blank, with all accepted forms
- Boards love specific numbers — include numeric targets wherever possible

TEXTBOOK PASSAGES:
{passages}
"""

_FR_PROMPT = """You are a veterinary neurology ACVIM Boards examiner.

TOPIC: {title}
DESCRIPTION: {description}

Write EXACTLY {n} free-response question(s) requiring clinical reasoning and integration,
not pure recall. Present a realistic clinical scenario and ask for reasoning, prioritization,
or a management plan.

Return ONLY valid JSON:
{{
  "fr": [
    {{
      "stem": "A detailed clinical scenario with a multi-part question that demands reasoning.",
      "rubric": [
        "5-7 specific, concrete rubric points a complete answer must address (include values, drug names, tests)"
      ],
      "sample_answer": "A 5-8 sentence model answer demonstrating boards-level clinical reasoning, with specific facts."
    }}
  ]
}}

TEXTBOOK PASSAGES:
{passages}
"""


def _generate_batch(
    fmt_key: str,
    prompt_template: str,
    n: int,
    topic: dict,
    passages_ctx: str,
    api_key: str,
) -> tuple[str, list]:
    per_q = {"mc": 280, "fitb": 200, "fr": 650}.get(fmt_key, 280)
    max_tokens = max(1800, n * per_q + 600)

    prompt = prompt_template.format(
        title=topic["title"],
        description=topic["description"],
        n=n,
        passages=passages_ctx,
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    parsed = _parse_json(response.content[0].text)
    return fmt_key, parsed.get(fmt_key, [])


def generate_questions(
    topic: dict,
    collection,
    api_key: str,
    num_mc: int = NUM_MC,
    num_fitb: int = NUM_FITB,
    num_fr: int = NUM_FR,
) -> dict:
    """Generate all question formats in parallel. Returns {mc, fitb, fr, citations}."""
    passages = search_topic(topic["query_terms"], collection, n=12)
    passages_ctx = _passages_context(passages)
    citations = _unique_citations(passages)

    batches = [
        ("mc",   _MC_PROMPT,   num_mc),
        ("fitb", _FITB_PROMPT, num_fitb),
        ("fr",   _FR_PROMPT,   num_fr),
    ]
    result: dict = {"mc": [], "fitb": [], "fr": [], "citations": citations}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_generate_batch, fkey, tmpl, n, topic, passages_ctx, api_key)
            for fkey, tmpl, n in batches
        ]
        for fut in futures:
            fkey, items = fut.result()
            result[fkey] = items

    return result


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_mc(question: dict, user_index: int) -> dict:
    correct = user_index == question["answer_index"]
    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "correct_answer": question["options"][question["answer_index"]],
        "explanation": question.get("explanation", ""),
    }


def grade_fitb(question: dict, user_answers: list[str]) -> dict:
    accepted_lists = question["accepted_answers"]
    per_blank = []
    for i, accepted in enumerate(accepted_lists):
        ua = user_answers[i] if i < len(user_answers) else ""
        norm_ua = _normalize(ua)
        norm_acc = {_normalize(a) for a in accepted}
        correct = norm_ua in norm_acc or any(a and a in norm_ua for a in norm_acc if a)
        per_blank.append({"user_answer": ua, "accepted": accepted, "correct": correct})
    score = sum(1 for b in per_blank if b["correct"]) / max(len(per_blank), 1)
    return {
        "score": score,
        "correct": score == 1.0,
        "per_blank": per_blank,
        "explanation": question.get("explanation", ""),
    }


_FR_GRADE_PROMPT = """You are a veterinary neurology ACVIM Boards examiner grading a resident's answer.
Grade constructively — award partial credit for any rubric point addressed.

Return ONLY valid JSON:
{{
  "score": <float 0.0-1.0>,
  "feedback": "2-3 sentences: name one thing done well, then the single most important gap.",
  "missed_points": ["specific rubric points the answer did not address"]
}}

QUESTION: {stem}

RUBRIC:
{rubric}

SAMPLE STRONG ANSWER:
{sample}

RESIDENT'S ANSWER:
{user}
"""


def grade_fr(question: dict, user_answer: str, api_key: str) -> dict:
    prompt = _FR_GRADE_PROMPT.format(
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
    parsed = _parse_json(response.content[0].text)
    parsed["correct"] = parsed.get("score", 0) >= 0.7
    parsed.setdefault("sample_answer", question.get("sample_answer", ""))
    return parsed


# ── Session score aggregation ─────────────────────────────────────────────────

def compute_session_score(answers_log: list[dict]) -> float:
    """Weighted session score: MC counts 0.6, FITB 0.2, FR 0.2."""
    weights = {"mc": 0.6, "fitb": 0.2, "fr": 0.2}
    weighted_scores: dict[str, list[float]] = {"mc": [], "fitb": [], "fr": []}
    for a in answers_log:
        q_type = a.get("q", "mc")
        if q_type in weighted_scores:
            weighted_scores[q_type].append(a.get("score", 0.0))

    total_weight = 0.0
    total_score = 0.0
    for q_type, scores in weighted_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            w = weights[q_type]
            total_score += avg * w
            total_weight += w

    return total_score / total_weight if total_weight > 0 else 0.0
