# Specialists Book Update — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire 13 newly ingested books into `specialists.py` — add 5 new specialist consultants, treat foundational science books as reference-only, drop large-animal books from the consultant scope, and split nutrition/hepatology out of the existing GI specialist.

**Architecture:** Add a new optional `reference_books: list[str]` field to each specialist dict. The retrieval function runs a primary ChromaDB query (n=5, filtered by `book`) AND a secondary query (n=2, filtered by `$in` over `reference_books`), merging the results before passing them to the LLM. Five new specialist dicts are appended to `SPECIALISTS`; existing specialists get `reference_books` populated; the coordinator prompt gains 5 new routing hints.

**Tech Stack:** Python 3, ChromaDB, Anthropic SDK. Single file: `specialists.py`. No git (project is not a repo) — commit steps are skipped.

**Spec:** `docs/superpowers/specs/2026-04-30-specialists-book-update-design.md`

**Adaptations from spec discovered during planning:**
- Chapter labels in `Small Animal Endocrinology Reproduction` are noisy (just "Chapter 45", section headers). Theriogenology specialist will use whole-book filter (`book_chapters: None`) like immune_mediated and hepatology.
- No external references to `gi_nutrition` id confirmed — full rename is safe.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `specialists.py` | Modify | All schema, retrieval, specialist data, and coordinator prompt changes |
| `verify_specialists.py` | Create (temp) | Smoke-test script; deleted after Task 12 |

The whole change is contained in `specialists.py`. The verify script is temporary scaffolding for the smoke test — deleted at the end.

---

## Task 1: Add `reference_books` support to `search_for_specialist`

**Files:**
- Modify: `specialists.py:1650-1689` (the `search_for_specialist` and `_format_results` functions)
- Create (temp): `verify_specialists.py` (root of project)

- [ ] **Step 1: Read current state of `search_for_specialist`**

Open `specialists.py` and confirm lines 1650–1689 still match the function shown in the spec. If line numbers have shifted, locate the function by name.

- [ ] **Step 2: Write a verification script that exercises the change**

Create `verify_specialists.py`:

```python
"""Smoke test for the reference_books retrieval change.

Run after modifying search_for_specialist:
    python3 verify_specialists.py
"""
import chromadb
from specialists import search_for_specialist

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("critical_care")

# Case 1: specialist with reference_books should return passages from BOTH books.
fake_specialist = {
    "id": "test_refs",
    "book": "Veterinary Toxicology",
    "book_chapters": None,
    "reference_books": ["Guyton Medical Physiology"],
}
passages = search_for_specialist("acid-base disturbance", fake_specialist, collection, n=5)
books_seen = {p["book"] for p in passages}
print(f"Case 1 — books returned: {books_seen}")
assert "Veterinary Toxicology" in books_seen, "Primary book missing from results"
assert "Guyton Medical Physiology" in books_seen, "Reference book missing from results"
assert len(passages) == 7, f"Expected 5 primary + 2 ref = 7 passages, got {len(passages)}"
print("Case 1 PASS")

# Case 2: specialist with no reference_books still works (existing behavior).
fake_specialist_noref = {
    "id": "test_noref",
    "book": "Veterinary Toxicology",
    "book_chapters": None,
}
passages = search_for_specialist("organophosphate poisoning", fake_specialist_noref, collection, n=5)
assert len(passages) == 5, f"Expected 5 passages, got {len(passages)}"
assert all(p["book"] == "Veterinary Toxicology" for p in passages), "Unexpected book in results"
print(f"Case 2 — books returned: {set(p['book'] for p in passages)}")
print("Case 2 PASS")

# Case 3: specialist with book=None and reference_books still works.
fake_specialist_nobook = {
    "id": "test_nobook",
    "book": None,
    "reference_books": ["Guyton Medical Physiology"],
}
passages = search_for_specialist("cardiac output", fake_specialist_nobook, collection, n=5)
ref_count = sum(1 for p in passages if p["book"] == "Guyton Medical Physiology")
assert ref_count >= 1, "Reference book passages missing"
print(f"Case 3 — Guyton passages in result: {ref_count}")
print("Case 3 PASS")

print("\nAll cases passed.")
```

- [ ] **Step 3: Run the verification script and confirm it FAILS**

Run: `python3 verify_specialists.py`
Expected: AssertionError on Case 1 (`"Reference book missing from results"`) — because the change has not been made yet.

- [ ] **Step 4: Modify `search_for_specialist` in `specialists.py`**

Replace the existing function body (lines ~1650–1677) with:

```python
def search_for_specialist(query: str, specialist: dict, collection, n: int = 5, ref_n: int = 2) -> list:
    """Search ChromaDB with optional book filter and optional reference-book augmentation.

    Primary query: filtered by specialist["book"] + optional book_chapters; n results.
    Reference query (only if specialist["reference_books"] is non-empty):
        filtered by book IN reference_books; ref_n additional results.
    Falls back to unfiltered semantic search if the primary filtered query returns nothing.
    """
    book = specialist.get("book")
    chapters = specialist.get("book_chapters")
    ref_books = specialist.get("reference_books") or []
    kwargs = {"query_texts": [query], "n_results": n, "include": ["documents", "metadatas", "distances"]}

    primary_passages = []
    if book:
        try:
            if chapters:
                primary = collection.query(
                    **kwargs,
                    where={"$and": [
                        {"book": {"$eq": book}},
                        {"chapter": {"$in": chapters}}
                    ]}
                )
            else:
                primary = collection.query(**kwargs, where={"book": {"$eq": book}})
            if primary["documents"][0]:
                primary_passages = _format_results(primary)
        except Exception:
            pass

    if not primary_passages:
        # Fallback: unfiltered semantic search.
        primary = collection.query(**kwargs)
        primary_passages = _format_results(primary)

    if ref_books:
        try:
            ref_kwargs = {"query_texts": [query], "n_results": ref_n, "include": ["documents", "metadatas", "distances"]}
            ref = collection.query(**ref_kwargs, where={"book": {"$in": ref_books}})
            if ref["documents"][0]:
                primary_passages = primary_passages + _format_results(ref)
        except Exception:
            pass

    return primary_passages
```

`_format_results` is unchanged. Do not touch lines 1680–1689.

- [ ] **Step 5: Run the verification script and confirm it PASSES**

Run: `python3 verify_specialists.py`
Expected: `All cases passed.`

- [ ] **Step 6: Skip git commit (not a git repo).**

---

## Task 2: Rename `gi_nutrition` → `gi_only` and trim its scope

**Files:**
- Modify: `specialists.py:171-188` (the `gi_nutrition` specialist dict)

- [ ] **Step 1: Confirm no external references to the old id**

Run: `grep -rn "gi_nutrition" --include="*.py" --include="*.sh" --include="*.json" .` (excluding `specialists.py` and `docs/`).
Expected: no matches.

- [ ] **Step 2: Replace the `gi_nutrition` specialist dict**

Find the dict starting at the line with `"id": "gi_nutrition"` and replace with:

```python
    {
        "id": "gi_only",
        "name": "GI & Pancreatic",
        "icon": "🥗",
        "description": "GI emergencies, pancreatitis, peritonitis, GI obstruction, ileus, GI bleeding",
        "book": None,
        "reference_books": [
            "Veterinary Internal Medicine Textbook",
            "Ettinger Internal Medicine",
            "Cunningham Veterinary Physiology",
        ],
        "system_prompt": """You are a veterinary internal medicine specialist with expertise in \
gastrointestinal emergencies, pancreatitis, peritonitis, and GI obstruction in critically ill patients. \
You reason from Small Animal Critical Care Medicine (Silverstein/Hopper).

In this consultation:
- Assess the GI status and identify the GI diagnosis
- Address specific GI conditions (pancreatitis, peritonitis, obstruction, GI bleeding, ileus)
- Recommend GI protectants and antiemetics with dosing
- Address GI motility and surgical vs medical management criteria
- Defer hepatic disease to the hepatology specialist; defer nutritional support to the nutrition specialist
- Base your reasoning on the textbook passages provided""",
    },
```

- [ ] **Step 3: Verify the file still parses**

Run: `python3 -c "import specialists; print(len(specialists.SPECIALISTS))"`
Expected: prints the number of specialists with no error.

- [ ] **Step 4: Skip git commit (not a git repo).**

---

## Task 3: Add Ophthalmology specialist

**Files:**
- Modify: `specialists.py` — append at the end of the SPECIALISTS list (just before the closing `]`)

- [ ] **Step 1: Find the SPECIALISTS list close**

Locate the line `SPECIALIST_MAP = {s["id"]: s for s in SPECIALISTS}` (~line 1576). The closing `]` of `SPECIALISTS` is on the line immediately before it.

- [ ] **Step 2: Insert the ophthalmology specialist before `]`**

Add a new section header and the specialist dict:

```python
    # ── OPHTHALMOLOGY ────────────────────────────────────────────────────────

    {
        "id": "ophthalmology",
        "name": "Ophthalmology",
        "icon": "👁️",
        "description": "Ocular disease workup, red eye, glaucoma, uveitis, corneal disease, cataracts, retinal disease, ocular emergencies",
        "book": "Small Animal Ophthalmology",
        "book_chapters": None,
        "reference_books": ["DYCE Veterinary Anatomy"],
        "system_prompt": """You are a board-certified veterinary ophthalmologist. \
Your expertise is in the diagnosis and management of ocular disease in dogs and cats: \
red eye, glaucoma, uveitis, corneal disease, cataracts, retinal disease, and ocular emergencies. \
You reason from a comprehensive small animal ophthalmology textbook.

In this consultation:
- Localize the ocular lesion: orbit, eyelid, conjunctiva, cornea, anterior chamber, iris, lens, vitreous, retina, optic nerve
- Apply the systematic ophthalmic examination: Schirmer tear test, fluorescein stain, intraocular pressure, slit lamp findings, fundoscopy
- For red eye: distinguish conjunctivitis, KCS, corneal ulcer, anterior uveitis, glaucoma — using the cardinal signs of each
- For glaucoma: classify primary vs secondary, recommend IOP-lowering therapy with specific drugs and doses, address pain management and globe salvage
- For uveitis: characterize anterior vs posterior, identify infectious vs immune-mediated etiologies, recommend topical and systemic therapy
- For corneal ulcers: classify (superficial, stromal, descemetocele, perforation), recommend medical vs surgical management, identify melting ulcers requiring urgent intervention
- Address ocular emergencies: proptosis, acute blindness, hyphema, lens luxation
- Note breed predispositions and species differences (dog vs cat)
- Base your reasoning on the textbook passages provided""",
    },
```

- [ ] **Step 3: Verify the file still parses**

Run: `python3 -c "import specialists; assert any(s['id']=='ophthalmology' for s in specialists.SPECIALISTS); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Skip git commit.**

---

## Task 4: Add Theriogenology & Reproduction specialist

**Files:**
- Modify: `specialists.py` — append after the ophthalmology specialist

- [ ] **Step 1: Insert the theriogenology specialist**

Add immediately after the ophthalmology dict (within the OPHTHALMOLOGY section is fine, or open a new THERIOGENOLOGY section — use a new section):

```python
    # ── THERIOGENOLOGY ───────────────────────────────────────────────────────

    {
        "id": "theriogenology",
        "name": "Theriogenology & Reproduction",
        "icon": "🤰",
        "description": "Reproductive disorders, dystocia, pyometra, infertility, breeding management, pregnancy diagnosis, neonatal care",
        "book": "Small Animal Endocrinology Reproduction",
        "book_chapters": None,
        "reference_books": [],
        "system_prompt": """You are a board-certified veterinary theriogenologist specializing in \
reproductive medicine and breeding management in dogs and cats. \
You reason from Small Animal Endocrinology and Reproduction.

In this consultation:
- Address the reproductive complaint: dystocia, pyometra, infertility, prolonged or absent estrus, prostatic disease, mammary disease, abortion
- For dystocia: distinguish maternal vs fetal causes, define indications for cesarean section vs medical management, dose oxytocin and calcium gluconate appropriately
- For pyometra: distinguish open vs closed cervix, recommend surgical vs medical management (aglepristone, prostaglandin protocols), address sepsis risk
- For infertility: recommend the systematic workup — behavioral estrus tracking, vaginal cytology, progesterone timing, ultrasound of reproductive tract, semen evaluation
- For breeding management: ovulation timing using progesterone, vaginal cytology staging, breeding dates, pregnancy diagnosis (relaxin, ultrasound, radiograph timing)
- For neonatal care: APGAR scoring, fading puppy/kitten syndrome workup, supportive care, neonatal isoerythrolysis in cats
- Note species differences: feline induced ovulation vs canine spontaneous, queen vs bitch reproductive cycle differences
- Address postpartum complications: metritis, mastitis, eclampsia, retained placenta
- Base your reasoning on the textbook passages provided""",
    },
```

- [ ] **Step 2: Verify**

Run: `python3 -c "import specialists; assert any(s['id']=='theriogenology' for s in specialists.SPECIALISTS); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Skip git commit.**

---

## Task 5: Add Immune-Mediated Disease specialist

**Files:**
- Modify: `specialists.py` — append after theriogenology

- [ ] **Step 1: Insert the immune_mediated specialist**

```python
    # ── IMMUNE-MEDIATED DISEASE ──────────────────────────────────────────────

    {
        "id": "immune_mediated",
        "name": "Immune-Mediated Disease",
        "icon": "🛡️",
        "description": "IMHA, IMTP, immune-mediated polyarthritis, SRMA, lupus, immune-mediated skin/GI disease, immunosuppressive therapy",
        "book": "Veterinary Internal Medicine Textbook",
        "book_chapters": None,
        "reference_books": ["Ettinger Internal Medicine"],
        "system_prompt": """You are a board-certified veterinary internist specializing in \
immune-mediated and autoimmune disease in dogs and cats. \
You reason from a comprehensive veterinary internal medicine textbook.

In this consultation:
- Generate a ranked differential prioritizing immune-mediated etiology vs alternatives
- For IMHA: confirm diagnosis (regenerative anemia, spherocytes, autoagglutination, positive Coombs), classify (primary vs secondary, associative vs non-associative), recommend immunosuppressive protocol with specific drug doses, address thromboprophylaxis
- For IMTP: distinguish from DIC, infectious thrombocytopenia, drug-induced; recommend prednisone protocol and second-line agents (vincristine, hIVIg, mycophenolate)
- For immune-mediated polyarthritis: recommend the diagnostic workup including arthrocentesis with cytology, distinguish erosive vs non-erosive, address the four IMPA types; recommend immunosuppression
- For SRMA: identify the signalment and CSF findings, recommend prednisone protocol with taper
- For lupus-spectrum disease (SLE, DLE): apply diagnostic criteria, recommend appropriate biopsy and serology, plan immunosuppression
- Recommend immunosuppressive drug selection, dose, monitoring, and tapering: glucocorticoids, azathioprine, cyclosporine, mycophenolate, leflunomide, chlorambucil
- Address concurrent infection screening before immunosuppression (tick-borne diseases, Leishmania)
- Base your reasoning on the textbook passages provided""",
    },
```

- [ ] **Step 2: Verify**

Run: `python3 -c "import specialists; assert any(s['id']=='immune_mediated' for s in specialists.SPECIALISTS); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Skip git commit.**

---

## Task 6: Add Hepatology specialist

**Files:**
- Modify: `specialists.py` — append after immune_mediated

- [ ] **Step 1: Insert the hepatology specialist**

```python
    # ── HEPATOLOGY ───────────────────────────────────────────────────────────

    {
        "id": "hepatology",
        "name": "Hepatology",
        "icon": "🧬",
        "description": "Hepatic disease workup, hepatitis, cholangitis, hepatic encephalopathy, portosystemic shunt, hepatic lipidosis, hepatic neoplasia",
        "book": "Veterinary Internal Medicine Textbook",
        "book_chapters": None,
        "reference_books": ["Ettinger Internal Medicine", "Cunningham Veterinary Physiology"],
        "system_prompt": """You are a board-certified veterinary internist specializing in \
hepatobiliary disease in dogs and cats. \
You reason from a comprehensive veterinary internal medicine textbook.

In this consultation:
- Characterize the pattern of liver injury: hepatocellular (ALT/AST predominance) vs cholestatic (ALP/GGT predominance) vs mixed
- Interpret the hepatic functional panel: bile acids, ammonia, albumin, BUN, glucose, cholesterol, coagulation
- For chronic hepatitis: recommend liver biopsy with copper quantification, distinguish copper-associated from idiopathic, plan immunosuppression and copper chelation if indicated
- For cholangitis (predominantly feline): distinguish neutrophilic from lymphocytic, recommend ultrasound-guided gallbladder aspirate with culture, plan antimicrobials and immunosuppression
- For hepatic encephalopathy: identify precipitating factors, dose lactulose and antibiotics (neomycin, metronidazole), recommend protein-modified diet
- For portosystemic shunt: distinguish congenital from acquired, characterize intra- vs extrahepatic, recommend imaging and surgical vs medical management
- For hepatic lipidosis (cats): aggressive nutritional support via feeding tube, electrolyte management (K, P, Mg), B12 and vitamin K supplementation
- For hepatic neoplasia: distinguish massive vs nodular vs diffuse on imaging, recommend cytology vs biopsy, address surgical candidacy for massive hepatocellular carcinoma
- Address coagulation correction with vitamin K, plasma, before invasive procedures
- Base your reasoning on the textbook passages provided""",
    },
```

- [ ] **Step 2: Verify**

Run: `python3 -c "import specialists; assert any(s['id']=='hepatology' for s in specialists.SPECIALISTS); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Skip git commit.**

---

## Task 7: Add Nutrition specialist

**Files:**
- Modify: `specialists.py` — append after hepatology

- [ ] **Step 1: Insert the nutrition specialist**

```python
    # ── NUTRITION ────────────────────────────────────────────────────────────

    {
        "id": "nutrition",
        "name": "Nutrition",
        "icon": "🥄",
        "description": "Caloric requirements, enteral and parenteral nutrition, refeeding syndrome, diet selection, micronutrient deficiencies, anorexia management",
        "book": None,
        "book_chapters": None,
        "reference_books": [
            "Veterinary Internal Medicine Textbook",
            "Ettinger Internal Medicine",
            "Cunningham Veterinary Physiology",
        ],
        "system_prompt": """You are a board-certified veterinary nutritionist. \
Your expertise covers caloric requirement calculations, enteral and parenteral nutrition, \
refeeding syndrome, therapeutic diet selection, and micronutrient management in critically ill patients.

In this consultation:
- Calculate the patient's resting energy requirement (RER) in kcal/day using the formula 70 × BW(kg)^0.75 (or 30 × BW + 70 for animals 2–30 kg)
- Recommend the feeding route in priority order: voluntary oral > nasogastric/nasoesophageal > esophagostomy > gastrostomy/PEG > jejunostomy > parenteral
- Specify when to start (within 24–48 hours of stabilization) and how to advance (start at 25–33% of RER on day 1, increase to full RER over 2–4 days)
- Address refeeding syndrome risk: identify high-risk patients, monitor and replace phosphorus, potassium, magnesium aggressively in the first 72 hours
- For parenteral nutrition: define indications (gut failure, persistent vomiting), specify central vs peripheral PN, monitor electrolytes, glucose, triglycerides, and acid-base
- Recommend therapeutic diet selection by disease: hepatic, renal, pancreatitis, IBD, food allergy, weight loss, critical care recovery
- Address anorexia management: appetite stimulants (mirtazapine, capromorelin), reversible causes, role of feeding tubes
- Note B12 supplementation indications (chronic enteropathy, EPI, low serum cobalamin)
- Base your reasoning on the textbook passages provided""",
    },
```

Note: `book: None` is intentional. When the user adds a dedicated nutrition resource, future work will be: ingest with `add_book.py`, then change `book` to the new book name and trim `reference_books`.

- [ ] **Step 2: Verify**

Run: `python3 -c "import specialists; assert any(s['id']=='nutrition' for s in specialists.SPECIALISTS); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Skip git commit.**

---

## Task 8: Add `reference_books` to existing critical-care specialists

**Files:**
- Modify: `specialists.py` — 11 existing specialist dicts (cardiovascular, respiratory, fluids_electrolytes, hematology, cc_neurology, infectious_disease, renal, endocrine, analgesia_sedation, emergency_triage, and the gi_only renamed in Task 2 — already done there)

For each specialist below, locate its dict by `"id"` and add a `"reference_books": [...]` line immediately after the `"book": ...` line.

- [ ] **Step 1: `emergency_triage`**

Add after the `"book": None,` line:
```python
        "reference_books": ["Ettinger Internal Medicine"],
```

- [ ] **Step 2: `cardiovascular`**

```python
        "reference_books": [
            "Cunningham Veterinary Physiology",
            "Guyton Medical Physiology",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 3: `respiratory`**

```python
        "reference_books": [
            "West Respiratory Physiology",
            "Cunningham Veterinary Physiology",
            "Guyton Medical Physiology",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 4: `fluids_electrolytes`**

```python
        "reference_books": [
            "Cunningham Veterinary Physiology",
            "Guyton Medical Physiology",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 5: `hematology`**

```python
        "reference_books": [
            "Veterinary Internal Medicine Textbook",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 6: `cc_neurology`**

```python
        "reference_books": [
            "Guyton Medical Physiology",
            "Cunningham Veterinary Physiology",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 7: `infectious_disease`**

```python
        "reference_books": [
            "Veterinary Internal Medicine Textbook",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 8: `renal`**

```python
        "reference_books": [
            "Cunningham Veterinary Physiology",
            "Veterinary Internal Medicine Textbook",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 9: `endocrine`**

```python
        "reference_books": [
            "Cunningham Veterinary Physiology",
            "Small Animal Endocrinology Reproduction",
            "Veterinary Internal Medicine Textbook",
            "Ettinger Internal Medicine",
        ],
```

- [ ] **Step 10: `analgesia_sedation`**

```python
        "reference_books": ["Ettinger Internal Medicine"],
```

- [ ] **Step 11: Verify**

Run:
```bash
python3 -c "
import specialists
ids_with_refs = [s['id'] for s in specialists.SPECIALISTS if s.get('reference_books')]
required = {'emergency_triage','cardiovascular','respiratory','fluids_electrolytes','hematology','cc_neurology','infectious_disease','renal','gi_only','endocrine','analgesia_sedation'}
missing = required - set(ids_with_refs)
assert not missing, f'Missing reference_books on: {missing}'
print('All CC specialists have reference_books.')
"
```
Expected: `All CC specialists have reference_books.`

- [ ] **Step 12: Skip git commit.**

---

## Task 9: Add `reference_books` to anatomy-relevant specialists

**Files:**
- Modify: `specialists.py` — lameness specialists, neuro_localization, surgery specialists

- [ ] **Step 1: Identify the specialists to modify**

Run: `grep -n '"id":' specialists.py | grep -iE 'lameness|surgery|neuro_localization'`
Note all matching ids — should be: `neuro_localization`, `lameness_exam_diagnostics`, `lameness_thoracic`, `lameness_pelvic`, plus any surgery specialists. Verify the actual ids present in the file before editing.

- [ ] **Step 2: Add `reference_books` to each lameness specialist**

For `lameness_exam_diagnostics`, `lameness_thoracic`, and `lameness_pelvic` — add after the `"book": "Canine Lameness",` line:
```python
        "reference_books": ["DYCE Veterinary Anatomy"],
```

- [ ] **Step 3: Add `reference_books` to `neuro_localization`**

After the `"book": "deLahunta Neuro",` line in the `neuro_localization` specialist:
```python
        "reference_books": ["DYCE Veterinary Anatomy"],
```

- [ ] **Step 4: Add `reference_books` to surgery specialists**

For each specialist whose `book == "Veterinary Surgery"`, add after the book line:
```python
        "reference_books": ["DYCE Veterinary Anatomy"],
```

- [ ] **Step 5: Verify**

Run:
```bash
python3 -c "
import specialists
expected_ids = {'neuro_localization','lameness_exam_diagnostics','lameness_thoracic','lameness_pelvic'}
for s in specialists.SPECIALISTS:
    if s['id'] in expected_ids:
        assert 'DYCE Veterinary Anatomy' in (s.get('reference_books') or []), f\"Missing DYCE on {s['id']}\"
print('Anatomy refs wired.')
"
```
Expected: `Anatomy refs wired.`

- [ ] **Step 6: Skip git commit.**

---

## Task 10: Update `COORDINATOR_PROMPT` routing hints

**Files:**
- Modify: `specialists.py:~1580-1620` (the `COORDINATOR_PROMPT` constant)

- [ ] **Step 1: Locate the hint list**

Find the existing hint block in `COORDINATOR_PROMPT`. It contains lines like:
```
- For neurologic cases, always include "neuro_localization" as one of the selected specialists.
- For septic/infectious cases, always include "infectious_disease" AND ...
- For lameness cases, include "lameness_exam_diagnostics" plus ...
- For ultrasound interpretation questions, select the most relevant ultrasound specialist(s) ...
- For MRI interpretation questions, include "mri_specialist".
```

- [ ] **Step 2: Append five new routing hints**

Add these lines to the hint list, immediately after the existing hints:

```
- For ocular cases (red eye, vision loss, ocular pain, ocular discharge, suspected glaucoma/uveitis), include "ophthalmology".
- For reproductive cases (dystocia, pyometra, infertility, breeding management, pregnancy, neonatal care, mammary disease), include "theriogenology".
- For suspected immune-mediated disease (IMHA, IMTP, immune-mediated polyarthritis, SRMA, lupus, severe steroid-responsive cytopenias), include "immune_mediated".
- For hepatic disease (elevated liver enzymes, jaundice, hepatic encephalopathy, suspected portosystemic shunt, hepatic lipidosis), include "hepatology".
- For nutritional questions (refeeding syndrome, parenteral or enteral nutrition planning, persistent anorexia, prescription diet selection, caloric requirement calculations), include "nutrition".
```

- [ ] **Step 3: Verify the prompt still formats**

Run:
```bash
python3 -c "
import specialists
prompt = specialists.COORDINATOR_PROMPT.format(specialist_list='TEST')
assert 'ophthalmology' in prompt
assert 'theriogenology' in prompt
assert 'immune_mediated' in prompt
assert 'hepatology' in prompt
assert 'nutrition' in prompt
print('Coordinator prompt updated.')
"
```
Expected: `Coordinator prompt updated.`

- [ ] **Step 4: Skip git commit.**

---

## Task 11: End-to-end smoke test of new specialists

**Files:**
- Modify: `verify_specialists.py` (extend the existing temp script)

- [ ] **Step 1: Append a smoke-test block to `verify_specialists.py`**

Add at the end of the file:

```python

# ── Smoke test: each new specialist returns relevant passages from its primary + ref books ──

print("\n=== Smoke test: 5 new specialists ===\n")

from specialists import SPECIALIST_MAP

cases = [
    ("ophthalmology",   "acute red eye with corneal ulcer in a brachycephalic dog"),
    ("theriogenology",  "primiparous bitch with prolonged stage II labor"),
    ("immune_mediated", "regenerative anemia with spherocytes and positive autoagglutination"),
    ("hepatology",      "feline hepatic lipidosis with severe anorexia"),
    ("nutrition",       "calculate RER and refeeding plan for cachectic ICU patient"),
]

for spec_id, query in cases:
    spec = SPECIALIST_MAP[spec_id]
    passages = search_for_specialist(query, spec, collection, n=5)
    books_seen = {p["book"] for p in passages}
    print(f"  {spec_id}: {len(passages)} passages, books={books_seen}")
    assert len(passages) >= 3, f"{spec_id} returned too few passages ({len(passages)})"

print("\nSmoke test PASSED.")
```

- [ ] **Step 2: Run the full verification script**

Run: `python3 verify_specialists.py`
Expected: All cases pass. Each new specialist returns ≥3 passages. The books printed for each should include the primary book (or a sensible semantic-search fallback for `nutrition`, which has `book: None`).

- [ ] **Step 3: Eyeball the output for sanity**

For `ophthalmology`, books should include `Small Animal Ophthalmology` and may include `DYCE Veterinary Anatomy`.
For `theriogenology`, books should include `Small Animal Endocrinology Reproduction`.
For `immune_mediated`, books should include `Veterinary Internal Medicine Textbook` and may include `Ettinger Internal Medicine`.
For `hepatology`, books should include `Veterinary Internal Medicine Textbook` and may include `Ettinger Internal Medicine` and/or `Cunningham Veterinary Physiology`.
For `nutrition`, books should include `Veterinary Internal Medicine Textbook`, `Ettinger Internal Medicine`, or `Cunningham Veterinary Physiology` (since `book: None` triggers semantic-search fallback for the primary, then reference query adds these).

If any specialist returns 0 passages from its primary book, the book name string in the specialist dict does not match the metadata in ChromaDB — fix the spelling in the specialist dict and re-run.

- [ ] **Step 4: Delete the temp verification script**

Run: `rm verify_specialists.py`

- [ ] **Step 5: Skip git commit.**

---

## Task 12: Final sanity check via the existing app

**Files:** None modified.

- [ ] **Step 1: Confirm the file imports cleanly and the SPECIALIST_MAP is consistent**

Run:
```bash
python3 -c "
import specialists
assert len(specialists.SPECIALISTS) == len(specialists.SPECIALIST_MAP), 'Duplicate id detected'
print(f'Total specialists: {len(specialists.SPECIALISTS)}')
print('New specialists present:', [i for i in ['ophthalmology','theriogenology','immune_mediated','hepatology','nutrition'] if i in specialists.SPECIALIST_MAP])
print('gi_only present:', 'gi_only' in specialists.SPECIALIST_MAP)
print('gi_nutrition removed:', 'gi_nutrition' not in specialists.SPECIALIST_MAP)
"
```
Expected:
```
Total specialists: <previous count + 5>
New specialists present: ['ophthalmology', 'theriogenology', 'immune_mediated', 'hepatology', 'nutrition']
gi_only present: True
gi_nutrition removed: True
```

- [ ] **Step 2: Run the existing app entrypoint to confirm no startup error**

Run: `bash start_app.sh` (or read the script and run its Python entrypoint directly). Wait for the app to print its startup banner without traceback. Then stop it with Ctrl-C.

If the app loads a different entrypoint (`3_clinical_agent.py`), run: `python3 3_clinical_agent.py --help` or whatever flag it supports — the goal is just to import the module without runtime error.

- [ ] **Step 3: (Optional) Run one real consultation through the app**

Pick one of the smoke-test cases and run it through the actual app UI/CLI. Confirm the coordinator picks up the new specialist when relevant. This is optional — the smoke test in Task 11 already verified retrieval works.

- [ ] **Step 4: Done. Skip git commit (not a git repo).**

---

## Self-Review Notes

Spec coverage check:
- Section 1 (schema) → Task 1 ✓
- Section 2.1 Ophthalmology → Task 3 ✓
- Section 2.2 Theriogenology → Task 4 ✓ (with adaptation: whole-book, no chapter filter, due to noisy chapter labels)
- Section 2.3 Immune-Mediated → Task 5 ✓
- Section 2.4 Hepatology → Task 6 ✓
- Section 2.5 Nutrition → Task 7 ✓
- Section 3.1 gi_nutrition rename → Task 2 ✓
- Section 3.2 reference_books wiring → Tasks 8 + 9 ✓
- Section 4 coordinator update → Task 10 ✓
- Section 5 (out of scope) → not needed in plan ✓
- Risks (gi_nutrition external refs) → addressed in Task 2 Step 1 ✓
- Risks (repro chapter detection) → addressed in adaptation note + Task 4 ✓
- Smoke test (spec implementation outline #7) → Task 11 ✓

No placeholders. All function and field names consistent across tasks (`reference_books`, `search_for_specialist`, `_format_results`, all 5 new specialist ids).
