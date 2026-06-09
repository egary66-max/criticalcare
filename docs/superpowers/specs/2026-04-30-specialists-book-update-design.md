# Specialists Update — New Book Influx

**Date:** 2026-04-30
**Status:** Draft, awaiting user review
**Scope:** `specialists.py` only

## Goal

13 books now exist in `chroma_db` that have no corresponding specialist in `specialists.py`. Update the specialist roster so the new content is reachable through the existing multi-agent consultation pipeline, and fill clear gaps in clinical coverage.

## Decisions (already settled in brainstorming)

1. **Foundational science books** (Guyton, West, Cunningham, DYCE, DiFiore) become *reference books* available to existing specialists, not standalone consultants.
2. **Large-animal / farm books** (Ashdown Equine, Ashdown Bovine, Anatomy & Physiology of Farm Animals, Jubb/Kennedy/Palmer Pathology) are out of scope. Chunks remain in the DB but no specialist will reference them.
3. **Internal Medicine textbooks** (Vet IM Textbook, Ettinger IM) are *not* given a consolidated "Internal Medicine" specialist. Vet IM Textbook becomes the primary book for new gap-filling specialists; Ettinger becomes a reference book on existing CC specialists.
4. **Five new specialists** are added: Ophthalmology, Theriogenology & Reproduction, Immune-Mediated Disease, Hepatology, Nutrition.

## Section 1 — Schema change: `reference_books`

### Current schema (per specialist dict)

```python
{
    "id": "...",
    "name": "...",
    "icon": "...",
    "description": "...",
    "book": <str | None>,
    "book_chapters": <list[str] | None>,
    "system_prompt": "...",
}
```

### New field

Add an optional `reference_books: list[str]` field. When present, the retrieval function runs an additional ChromaDB query restricted to those books, with a smaller `n` (default 2), and merges the results with the primary query.

```python
{
    ...
    "book": "Silverstein CC",
    "reference_books": ["Cunningham Veterinary Physiology", "Guyton Medical Physiology"],
    ...
}
```

### Retrieval change in `search_for_specialist`

Pseudocode for the new retrieval flow:

```
primary = collection.query(query, n=5, where={book == specialist["book"]} if book else no-where)
ref_books = specialist.get("reference_books") or []
if ref_books:
    refs = collection.query(query, n=2, where={book IN ref_books})
    return _format_results(primary) + _format_results(refs)
return _format_results(primary)
```

- Primary query keeps its existing fallback behavior (unfiltered semantic if filtered returns empty).
- Reference query has no fallback — if it returns nothing, just primary results are used.
- `build_passages_context` already handles arbitrary lengths, no change needed.
- Default n for primary is 5 (existing); default n for reference is 2.

## Section 2 — Five new specialists

All system prompts follow the existing voice and formatting (terse instruction list, "Base your reasoning on the textbook passages provided" footer).

### 2.1 Ophthalmology

```python
{
    "id": "ophthalmology",
    "name": "Ophthalmology",
    "icon": "👁️",
    "description": "Ocular disease workup, red eye, glaucoma, uveitis, corneal disease, cataracts, retinal disease",
    "book": "Small Animal Ophthalmology",
    "book_chapters": None,  # 150 chunks total — full-book semantic search
    "reference_books": ["DYCE Veterinary Anatomy"],
    "system_prompt": <board-certified vet ophthalmologist persona>,
}
```

Routing hint: include for any ocular complaint (red eye, blindness, photophobia, ocular discharge, ocular pain).

### 2.2 Theriogenology & Reproduction

```python
{
    "id": "theriogenology",
    "name": "Theriogenology & Reproduction",
    "icon": "🤰",
    "description": "Reproductive disorders, dystocia, pyometra, infertility, breeding management, neonatal care",
    "book": "Small Animal Endocrinology Reproduction",
    "book_chapters": <reproduction-half chapter list — to be filled in implementation by inspecting metadata>,
    "reference_books": [],
    "system_prompt": <board-certified theriogenologist persona>,
}
```

Note: chapter list to be derived during implementation by inspecting `chroma_db` for chapter labels in this book and selecting the reproduction-side ones (excluding endocrine chapters, which stay with the existing `endocrine` specialist via reference_books).

Routing hint: include for dystocia, pyometra, infertility, breeding, pregnancy management, neonatal cases.

### 2.3 Immune-Mediated Disease

```python
{
    "id": "immune_mediated",
    "name": "Immune-Mediated Disease",
    "icon": "🛡️",
    "description": "IMHA, IMTP, immune-mediated polyarthritis, SRMA, lupus, immunosuppressive therapy",
    "book": "Veterinary Internal Medicine Textbook",
    "book_chapters": None,  # whole-book filter (chapter labels unusable in this book)
    "reference_books": ["Ettinger Internal Medicine"],
    "system_prompt": <board-certified internist focused on immune-mediated disease persona>,
}
```

Routing hint: include for suspected immune-mediated cytopenias (regenerative anemia + spherocytes, severe thrombocytopenia without DIC), polyarthritis, steroid-responsive meningitis-arteritis, lupus-spectrum disease.

### 2.4 Hepatology

```python
{
    "id": "hepatology",
    "name": "Hepatology",
    "icon": "🧬",
    "description": "Hepatic disease workup, hepatitis, cholangitis, hepatic encephalopathy, portosystemic shunt, hepatic lipidosis",
    "book": "Veterinary Internal Medicine Textbook",
    "book_chapters": None,
    "reference_books": ["Ettinger Internal Medicine", "Cunningham Veterinary Physiology"],
    "system_prompt": <board-certified internist focused on hepatobiliary disease persona>,
}
```

Routing hint: include for elevated liver enzymes, jaundice, hepatic encephalopathy, ascites with hepatic origin, suspected hepatitis or shunt.

### 2.5 Nutrition

```python
{
    "id": "nutrition",
    "name": "Nutrition",
    "icon": "🥄",
    "description": "Caloric requirement calculations, enteral and parenteral nutrition, diet selection, refeeding syndrome, micronutrient deficiencies",
    "book": None,  # placeholder — user will add a nutrition resource later
    "book_chapters": None,
    "reference_books": ["Veterinary Internal Medicine Textbook", "Ettinger Internal Medicine", "Cunningham Veterinary Physiology"],
    "system_prompt": <board-certified veterinary nutritionist persona>,
}
```

When the user adds a nutrition textbook, the implementation work is: ingest with `add_book.py`, then update `book` to the new book name and trim or remove `reference_books` as appropriate.

## Section 3 — Existing specialist updates

### 3.1 `gi_nutrition` → `gi_only` (rename + scope trim)

`gi_nutrition` currently combines GI emergencies, hepatic disease, pancreatitis, and nutrition. After this update:

- Rename id `gi_nutrition` → `gi_only`
- Rename name "GI & Nutrition" → "GI & Pancreatic"
- Description: "GI emergencies, pancreatitis, peritonitis, GI obstruction"
- System prompt: remove hepatic and nutritional responsibility paragraphs; refer them to the new specialists.
- Add `reference_books: ["Veterinary Internal Medicine Textbook", "Ettinger Internal Medicine", "Cunningham Veterinary Physiology"]`

### 3.2 Reference-book wiring on existing specialists

| Specialist | Add `reference_books` |
|---|---|
| `cardiovascular` | `["Cunningham Veterinary Physiology", "Guyton Medical Physiology", "Ettinger Internal Medicine"]` |
| `respiratory` | `["West Respiratory Physiology", "Cunningham Veterinary Physiology", "Guyton Medical Physiology", "Ettinger Internal Medicine"]` |
| `fluids_electrolytes` | `["Cunningham Veterinary Physiology", "Guyton Medical Physiology", "Ettinger Internal Medicine"]` |
| `hematology` | `["Veterinary Internal Medicine Textbook", "Ettinger Internal Medicine"]` |
| `cc_neurology` | `["Guyton Medical Physiology", "Cunningham Veterinary Physiology", "Ettinger Internal Medicine"]` |
| `infectious_disease` | `["Veterinary Internal Medicine Textbook", "Ettinger Internal Medicine"]` |
| `renal` | `["Cunningham Veterinary Physiology", "Veterinary Internal Medicine Textbook", "Ettinger Internal Medicine"]` |
| `gi_only` (renamed) | `["Veterinary Internal Medicine Textbook", "Ettinger Internal Medicine", "Cunningham Veterinary Physiology"]` |
| `endocrine` | `["Cunningham Veterinary Physiology", "Small Animal Endocrinology Reproduction", "Veterinary Internal Medicine Textbook", "Ettinger Internal Medicine"]` |
| `analgesia_sedation` | `["Ettinger Internal Medicine"]` |
| `emergency_triage` | `["Ettinger Internal Medicine"]` |
| `lameness_*` (3 specialists) | `["DYCE Veterinary Anatomy"]` (each) |
| `neuro_localization` | `["DYCE Veterinary Anatomy"]` |
| `surgery_*` (existing surgery specialists) | `["DYCE Veterinary Anatomy"]` |

Specialists not modified: toxicology specialists, oncology specialists, antimicrobial specialists, dermatology specialists, clinpath specialists, radiology specialists, deLahunta neuro specialists (other than localization), neurosurgery specialists, MRI, ultrasound specialists, ID-Greene specialists, lameness specialists already have their primary book — only `reference_books` is added.

DiFiore Atlas of Histology (6 chunks) is **not wired anywhere** — too thin to be useful.

## Section 4 — Coordinator prompt update

Lines ~1580–1620 hold `COORDINATOR_PROMPT`. The routing-rule list currently includes hints like "for neurologic cases, always include neuro_localization". Add equivalent hints for the 5 new specialists:

- For ocular cases (red eye, vision loss, ocular discharge), include `ophthalmology`.
- For reproductive cases (dystocia, pyometra, breeding, neonatal), include `theriogenology`.
- For suspected immune-mediated disease (IMHA, IMTP, polyarthritis, SRMA), include `immune_mediated`.
- For hepatic disease (elevated liver enzymes, jaundice, hepatic encephalopathy, shunt), include `hepatology`.
- For nutritional questions (refeeding, parenteral nutrition, caloric calc, anorexia), include `nutrition`.

The "select 2-4 specialists" cap is preserved. New specialists appear in `specialist_list` automatically because it's generated from `SPECIALISTS`.

## Section 5 — Out of scope

These are explicitly NOT part of this update:
- Re-ingesting any books (chunks are already in the DB).
- Touching `1_extract_book.py`, `2_build_database.py`, or `add_book.py`.
- Removing the dropped books (Ashdown Equine/Bovine, Farm Animals, Jubb) from the ChromaDB. They stay; nothing references them.
- Changing the synthesizer prompt or consultation pipeline beyond the coordinator hints.
- Adding new behavior, geriatric, pediatric, or oncology-related specialists. (Considered, declined.)

## Implementation outline (preview for the plan)

1. Modify `search_for_specialist` to handle `reference_books`.
2. Add 5 new specialist dicts to `SPECIALISTS`.
3. Rename `gi_nutrition` → `gi_only` and edit its scope.
4. Add `reference_books` to existing specialists per Section 3.2.
5. Update `COORDINATOR_PROMPT` routing hints.
6. Inspect `chroma_db` chapter metadata for "Small Animal Endocrinology Reproduction" to populate the theriogenology specialist's `book_chapters` list with reproduction-side chapters only.
7. Smoke-test by running a sample case through the pipeline (ocular, repro, hepatic, IMHA, nutrition).

## Risks / things to verify

- **Reference-book signal interference:** if reference passages dominate, recommendations might drift from the primary book. Mitigation: reference query uses `n=2` vs primary `n=5`.
- **Reproduction chapter detection:** the book "Small Animal Endocrinology Reproduction" combines two domains; chapter labels need inspection to pick the reproduction-side ones cleanly.
- **`gi_nutrition` rename:** if the id is referenced from elsewhere (e.g., a saved case routing log, a UI), renaming breaks it. Verify no external references exist before renaming. Falls back to leaving id as `gi_nutrition` and only updating name/description if external references exist.
