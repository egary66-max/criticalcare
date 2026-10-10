"""
ACVIM Neurology Boards curriculum: 25 topics across 4 phases.
Ordered from intern-level foundations to graduating-resident / boards-level mastery.
"""

PHASES = [
    {
        "id": "p1",
        "name": "Phase 1: Foundations",
        "icon": "🧱",
        "description": "Core tools every neurology intern must have: exam technique, localization, diagnostics.",
    },
    {
        "id": "p2",
        "name": "Phase 2: Spinal Cord & Peripheral Nervous System",
        "icon": "🦴",
        "description": "The most common clinical neuro problems: IVDD, myelopathies, and peripheral disease.",
    },
    {
        "id": "p3",
        "name": "Phase 3: Brain, Seizures & Inflammation",
        "icon": "🧠",
        "description": "Intracranial disease, epilepsy management, and CNS inflammatory conditions.",
    },
    {
        "id": "p4",
        "name": "Phase 4: Advanced / ACVIM Boards",
        "icon": "🎓",
        "description": "Pharmacology depth, MRI interpretation, neurosurgery, and boards-level integration.",
    },
]

TOPICS = [
    # ── Phase 1: Foundations ──────────────────────────────────────────────────

    {
        "id": "neuro_exam",
        "phase": "p1",
        "title": "Neurological Examination",
        "description": (
            "Systematic neurologic exam technique: mentation, gait, postural reactions "
            "(hopping, hemiwalking, wheelbarrowing, placing), spinal reflexes (patellar, "
            "withdrawal, perineal, panniculus), and cranial nerve assessment. Normal vs "
            "abnormal findings and their neuroanatomical significance."
        ),
        "query_terms": (
            "neurological examination gait postural reactions patellar reflex withdrawal "
            "cranial nerve mentation mental status veterinary neurology neuro exam"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Neurologic Examination and Neuroanatomic Diagnosis (Ch 259)",
            "Small Animal Critical Care Medicine — Neurological Evaluation of the ICU Patient (Ch 83)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "localization",
        "phase": "p1",
        "title": "Neuroanatomical Localization",
        "description": (
            "Localizing neurologic lesions to: forebrain, brainstem, cerebellum, C1-C5, "
            "C6-T2, T3-L3, L4-S3, multifocal/diffuse. UMN vs LMN signs — what each means "
            "for prognosis and DDx. Crossed reflexes, proprioceptive deficits, and "
            "Schiff-Sherrington posture."
        ),
        "query_terms": (
            "neuroanatomical localization forebrain brainstem cerebellum spinal cord segments "
            "UMN LMN upper motor neuron lower motor neuron lesion localization Schiff-Sherrington"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Neurologic Examination and Neuroanatomic Diagnosis (Ch 259)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "csf",
        "phase": "p1",
        "title": "CSF Analysis and Interpretation",
        "description": (
            "CSF collection technique (cisternal vs lumbar), normal values (cell count, "
            "protein, glucose), and interpretation: pleocytosis patterns (neutrophilic, "
            "mononuclear, mixed, eosinophilic), xanthochromia, high protein. CSF patterns "
            "by disease: GME, infectious meningoencephalitis, steroid-responsive "
            "meningitis-arteritis (SRMA), neoplasia."
        ),
        "query_terms": (
            "cerebrospinal fluid CSF analysis collection interpretation cell count protein "
            "pleocytosis xanthochromia cisternal lumbar puncture GME SRMA meningitis"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Cerebrospinal Fluid and Hydrocephalus (Ch 260)",
            "Small Animal Critical Care Medicine — CSF Sampling and Interpretation (Ch 206)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "neurodiagnostics",
        "phase": "p1",
        "title": "Neurodiagnostics: MRI, CT, and Electrodiagnostics",
        "description": (
            "When to choose MRI vs CT vs myelography. MRI sequences: T1, T2, FLAIR, "
            "DWI, post-contrast — what each shows. Electrodiagnostics: EMG (fibrillation "
            "potentials, positive sharp waves, fasciculations), nerve conduction velocity, "
            "repetitive stimulation for NMJ disease. BAER for deafness."
        ),
        "query_terms": (
            "MRI CT imaging neurodiagnostics EMG electromyography nerve conduction velocity "
            "BAER brain imaging spinal cord MRI sequences T1 T2 FLAIR contrast"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diagnostic Imaging of the Nervous System (Ch 260)",
        ],
        "knowledge_supplemented": True,
    },
    {
        "id": "neuropathology_basics",
        "phase": "p1",
        "title": "Neuropathology: Inflammation, Degeneration & Neoplasia",
        "description": (
            "CNS pathological mechanisms: neuroinflammation (VAIN rule: Vascular, "
            "Anomalous/Developmental, Inflammatory, Neoplastic, Degenerative/Traumatic/Toxic, "
            "Idiopathic, Nutritional), Wallerian degeneration, demyelination, necrosis. "
            "How pathology predicts MRI signal, CSF findings, and disease course."
        ),
        "query_terms": (
            "neuropathology inflammation degeneration demyelination neoplasia CNS pathology "
            "Wallerian degeneration neuroinflammation DAMNIT-V rule differential"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Brain (Ch 260)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "neuro_pharma_foundations",
        "phase": "p1",
        "title": "Pharmacology Foundations: CNS Drug Delivery",
        "description": (
            "Blood-brain barrier (BBB) structure and function, P-glycoprotein efflux, "
            "lipid solubility and CNS penetration. MDR1 (ABCB1) mutation in collies — "
            "which drugs are dangerous. Principles for selecting CNS-penetrant antibiotics "
            "(fluoroquinolones, chloramphenicol, metronidazole, trimethoprim-sulfa)."
        ),
        "query_terms": (
            "blood brain barrier CNS drug penetration pharmacokinetics MDR1 ABCB1 P-glycoprotein "
            "collie ivermectin CNS antibiotics fluoroquinolone metronidazole"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Principles of Drug Therapy",
            "Antimicrobial Therapy — Pharmacokinetics of Antimicrobials (Ch 4)",
        ],
        "knowledge_supplemented": True,
    },

    # ── Phase 2: Spinal Cord & PNS ────────────────────────────────────────────

    {
        "id": "cervical_ivdd",
        "phase": "p2",
        "title": "Cervical Spinal Cord Disease and IVDD",
        "description": (
            "Cervical IVDD: disc extrusion vs protrusion, clinical grades, management. "
            "Cervical spondylomyelopathy (CSM / 'Wobbler'): breed predispositions "
            "(Doberman, Great Dane), static vs dynamic compression, surgical options. "
            "Atlantoaxial instability: signalment, diagnosis, stabilization. "
            "Cervical pain assessment and analgesia."
        ),
        "query_terms": (
            "cervical spinal cord IVDD disc disease wobbler syndrome cervical spondylomyelopathy "
            "CSM atlantoaxial instability cervical pain ventral slot decompression"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Spinal Cord (Ch 265)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "tl_ivdd",
        "phase": "p2",
        "title": "Thoracolumbar IVDD: Grades and Management",
        "description": (
            "Hansen Type I vs II disc disease. Modified Frankel/ASIA grading scale "
            "(Grades 1-5), deep pain significance and prognosis. Medical management "
            "criteria vs surgical criteria. Hemilaminectomy approach and timing. "
            "Myelomalacia: recognition, ascending/descending signs, prognosis. "
            "Chondrodystrophic vs non-chondrodystrophic breeds."
        ),
        "query_terms": (
            "thoracolumbar IVDD intervertebral disc disease Hansen type hemilaminectomy "
            "deep pain prognosis grade myelomalacia chondrodystrophic Dachshund"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Spinal Cord (Ch 265)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "degenerative_myelopathy",
        "phase": "p2",
        "title": "Degenerative Myelopathy",
        "description": (
            "SOD1 gene mutation, breed predispositions (GSD, Pembroke Welsh Corgi, "
            "Labrador, Boxer). Clinical stages: pelvic limb ataxia/weakness → "
            "paraplegia → tetraplegia. Electrodiagnostic findings, MRI appearance "
            "(normal or subtle), diagnosis of exclusion. Rehabilitation and palliative "
            "care. Differentiation from compressive myelopathy."
        ),
        "query_terms": (
            "degenerative myelopathy SOD1 mutation progressive hindlimb weakness German Shepherd "
            "Labrador diagnosis exclusion EMG rehabilitation palliative"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Spinal Cord (Ch 265)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "fce_vascular",
        "phase": "p2",
        "title": "FCE and Vascular Myelopathy",
        "description": (
            "Fibrocartilaginous embolism (FCE): acute non-progressive onset, lateralization, "
            "non-painful, MRI intramedullary T2 hyperintensity, normal CSF. Large vs "
            "small breed patterns. Prognosis and rehabilitation. Acute non-compressive "
            "nucleus pulposus extrusion (ANNPE): high-velocity extrusion without "
            "compression, management."
        ),
        "query_terms": (
            "fibrocartilaginous embolism FCE ischemic myelopathy acute spinal cord infarct "
            "vascular non-progressive non-painful lateralized rehabilitation ANNPE"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Spinal Cord (Ch 265)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "lumbosacral",
        "phase": "p2",
        "title": "Lumbosacral Disease and Cauda Equina Syndrome",
        "description": (
            "Lumbosacral stenosis/IVDD, cauda equina syndrome signs: tail weakness, "
            "perineal hypoalgesia, urinary/fecal incontinence, lumbosacral pain on "
            "extension. Large breeds (GSD). CT myelography vs MRI for diagnosis. "
            "Medical vs surgical management (dorsal laminectomy/foraminotomy). "
            "Discospondylitis involving L7-S1."
        ),
        "query_terms": (
            "lumbosacral disease cauda equina syndrome urinary incontinence L7 S1 stenosis "
            "foraminal lumbosacral pain dorsal laminectomy discospondylitis"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Spinal Cord (Ch 265)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "spinal_neoplasia",
        "phase": "p2",
        "title": "Spinal Cord Neoplasia and Other Myelopathies",
        "description": (
            "Tumor classification: intramedullary (astrocytoma, ependymoma), "
            "intradural-extramedullary (meningioma, nerve sheath tumor, lymphoma), "
            "extradural (vertebral tumors, lymphoma). MRI patterns for each. "
            "Discospondylitis: organisms, diagnosis (blood culture, urine culture, "
            "disc aspirate), antibiotic duration (6-8 weeks minimum). Steroid-responsive "
            "meningitis-arteritis (SRMA): clinical signs, CSF, treatment."
        ),
        "query_terms": (
            "spinal cord neoplasia tumor meningioma nerve sheath lymphoma intramedullary "
            "extradural discospondylitis SRMA steroid responsive meningitis"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Spinal Cord (Ch 265)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "pns_cranial_nerves",
        "phase": "p2",
        "title": "Cranial Nerve Disorders and Peripheral Neuropathies",
        "description": (
            "CN mapping and expected deficits: CN II (vision/PLR), CN III (mydriasis, "
            "ventrolateral strabismus), CN V (jaw tone, facial sensation), CN VII (facial "
            "paresis), CN VIII (vestibular), CN IX/X (dysphagia, megaesophagus), "
            "CN XII (tongue). Horner's syndrome: 1st/2nd/3rd order neuron. "
            "Polyneuropathies: diabetic, hypothyroid, paraneoplastic, immune-mediated. "
            "Tick paralysis: mechanism and rapid reversal."
        ),
        "query_terms": (
            "cranial nerve palsy facial nerve Horner syndrome trigeminal neuropathy "
            "polyneuropathy peripheral neuropathy tick paralysis CN deficits"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Peripheral Nervous System (Ch 275)",
        ],
        "knowledge_supplemented": False,
    },

    # ── Phase 3: Brain, Seizures & Inflammation ───────────────────────────────

    {
        "id": "forebrain_disease",
        "phase": "p3",
        "title": "Forebrain Disease",
        "description": (
            "Cerebrocortical and diencephalic lesion signs: contralateral proprioceptive "
            "deficits, contralateral vision loss with normal PLR, compulsive circling "
            "toward lesion, behavioral changes, altered mentation. Causes: neoplasia, "
            "inflammation, vascular (ischemic/hemorrhagic stroke), hydrocephalus. "
            "Hepatic encephalopathy: mechanism, ammonia, diagnosis, management."
        ),
        "query_terms": (
            "forebrain disease cerebral cortex thalamus diencephalon behavioral changes "
            "circling vision deficits contralateral hepatic encephalopathy hydrocephalus"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Brain (Ch 260)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "brainstem_caudal_fossa",
        "phase": "p3",
        "title": "Brainstem and Caudal Fossa Disease",
        "description": (
            "Brainstem signs: ipsilateral CN deficits + contralateral proprioceptive loss "
            "(crossed signs = brainstem). Midbrain, pons, medulla localization. "
            "Chiari-like malformation (CLM) and syringomyelia in Cavalier King Charles "
            "Spaniels: signs, MRI, medical vs surgical management. Intracranial "
            "hypertension: recognition, Cushing reflex, mannitol, hypertonic saline."
        ),
        "query_terms": (
            "brainstem caudal fossa disease Chiari malformation syringomyelia intracranial "
            "hypertension Cushing reflex mannitol CKCS pons medulla midbrain"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Brain (Ch 260)",
            "Small Animal Critical Care Medicine — Intracranial Hypertension (Ch 85)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "cerebellar_disease",
        "phase": "p3",
        "title": "Cerebellar Disease",
        "description": (
            "Cerebellar signs: intention tremor, hypermetria, dysmetria, broad-based "
            "stance, truncal ataxia — distinguish from proprioceptive ataxia. "
            "Cerebellar hypoplasia (panleukopenia in cats, herpesvirus). "
            "Cerebellar cortical abiotrophy: breeds (Beagle, Gordon Setter, "
            "Kerry Blue Terrier). Acquired cerebellar disease: inflammatory, "
            "vascular, neoplastic. Paradoxical vestibular syndrome."
        ),
        "query_terms": (
            "cerebellar disease ataxia hypermetria intention tremor cerebellar hypoplasia "
            "abiotrophy cortical paradoxical vestibular cerebellar cortical degeneration"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of the Brain (Ch 260)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "vestibular_disease",
        "phase": "p3",
        "title": "Vestibular Disease: Central vs Peripheral",
        "description": (
            "Key differentiators: central (ipsilateral proprioceptive deficits, "
            "vertical/changing nystagmus, CN deficits, altered mentation) vs peripheral "
            "(normal proprioception, horizontal/rotary nystagmus, Horner's possible, "
            "normal mentation). Idiopathic geriatric vestibular (dogs) and feline "
            "idiopathic vestibular. Otitis media/interna: diagnosis and treatment. "
            "Nasopharyngeal polyps in cats."
        ),
        "query_terms": (
            "vestibular disease central peripheral nystagmus head tilt otitis media interna "
            "idiopathic vestibular Horner syndrome proprioception paradoxical"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Vestibular Disorders (Ch 260)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "seizures_epilepsy",
        "phase": "p3",
        "title": "Seizure Disorders and Epilepsy Syndromes",
        "description": (
            "ILAE seizure classification: focal, focal with secondary generalization, "
            "generalized. Idiopathic epilepsy (IE): age of onset 1-5 years, normal "
            "interictal exam, normal MRI/CSF. Structural vs reactive seizures. "
            "Breed-specific epilepsy syndromes (Labrador, Border Collie, GSD, Beagle). "
            "When to start anticonvulsants: >2 seizures/6 months, cluster seizures, "
            "SE, post-ictal >24hr. Metabolic workup for new-onset seizures."
        ),
        "query_terms": (
            "seizure classification epilepsy idiopathic structural reactive breed-specific "
            "epilepsy syndrome focal generalized anticonvulsant threshold interictal"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Seizures and Epilepsy (Ch 261)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "status_epilepticus",
        "phase": "p3",
        "title": "Status Epilepticus: Recognition and Emergency Management",
        "description": (
            "Status epilepticus (SE): continuous seizure >5 min or ≥2 without recovery. "
            "Step-up protocol: diazepam 0.5 mg/kg IV (repeat ×2), then midazolam "
            "0.2-0.5 mg/kg IV/IM, then phenobarbital 2-4 mg/kg IV slowly (repeat to "
            "16-20 mg/kg load). Refractory SE: propofol 1-6 mg/kg/hr CRI, "
            "levetiracetam 30-60 mg/kg IV, ketamine CRI. Neuroprotection, thermoregulation, "
            "glucose monitoring."
        ),
        "query_terms": (
            "status epilepticus treatment protocol diazepam phenobarbital loading propofol "
            "refractory seizures emergency midazolam levetiracetam ketamine CRI"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Status Epilepticus (Ch 261)",
            "Small Animal Critical Care Medicine — Neurological Emergencies",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "inflammatory_cns",
        "phase": "p3",
        "title": "Inflammatory CNS Disease: GME, NME, NLE, and Infectious",
        "description": (
            "GME (granulomatous meningoencephalomyelitis): focal/multifocal/ocular forms, "
            "any breed. NME (necrotizing meningoencephalitis): Pug, Maltese, Chihuahua. "
            "NLE (necrotizing leukoencephalitis): Yorkshire Terrier, French Bulldog. "
            "CSF and MRI patterns. Infectious: Cryptococcus (cats > dogs, India ink, "
            "antigen test), canine distemper, Toxoplasma/Neospora, Rocky Mountain spotted "
            "fever. Treatment: immunosuppression protocols."
        ),
        "query_terms": (
            "granulomatous meningoencephalitis GME NME NLE inflammatory brain disease "
            "cryptococcus distemper Toxoplasma meningoencephalitis CSF MRI necrotizing"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Inflammatory Diseases of the CNS (Ch 260)",
        ],
        "knowledge_supplemented": False,
    },
    {
        "id": "neuromuscular_disease",
        "phase": "p3",
        "title": "Neuromuscular Disease: MG, Myopathies, and LMN Disorders",
        "description": (
            "Myasthenia gravis (MG): focal vs generalized vs acute fulminating, "
            "acetylcholine receptor antibody titer, tensilon test, megaesophagus, "
            "aspiration pneumonia. Botulism: flaccid tetraplegia, intact mentation, "
            "no deep pain loss. Tick paralysis: rapid reversal after tick removal. "
            "Inflammatory myopathies: polymyositis (CK elevation, EMG), masticatory "
            "muscle myositis (type 2M antibody). Congenital myopathies by breed."
        ),
        "query_terms": (
            "myasthenia gravis neuromuscular junction disease polymyositis masticatory muscle "
            "myositis botulism tick paralysis LMN disorder megaesophagus AChR antibody"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diseases of Muscle and NMJ (Ch 276-277)",
        ],
        "knowledge_supplemented": False,
    },

    # ── Phase 4: Advanced / ACVIM Boards ─────────────────────────────────────

    {
        "id": "anticonvulsant_pharma",
        "phase": "p4",
        "title": "Anticonvulsant Pharmacology",
        "description": (
            "Phenobarbital: MOA (GABA-A enhancer), dose 2.5 mg/kg PO BID, therapeutic "
            "serum level 20-40 mcg/mL, hepatotoxicity monitoring (bile acids q6mo), "
            "induces CYP450. Potassium bromide (KBr): dose 20-30 mg/kg PO q24h (dog only — "
            "causes pulmonary disease in cats), therapeutic 1000-2000 mg/L, slow onset "
            "(3-4 months). Levetiracetam: 20 mg/kg PO/IV TID, minimal interactions, "
            "safe in hepatic disease. Zonisamide: 5-10 mg/kg PO BID. "
            "Pregabalin/Gabapentin: adjunct therapy."
        ),
        "query_terms": (
            "phenobarbital potassium bromide levetiracetam zonisamide anticonvulsant "
            "seizure pharmacology dose monitoring hepatotoxicity therapeutic level"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Antiepileptic Drug Therapy (Ch 262)",
        ],
        "knowledge_supplemented": True,
    },
    {
        "id": "neuro_immunosuppression",
        "phase": "p4",
        "title": "Neuro Immunology: Immunosuppressive Therapy for CNS Disease",
        "description": (
            "Prednisone induction (2 mg/kg/day) then taper. Cyclosporine 5-10 mg/kg/day "
            "(trough levels 200-400 ng/mL). CCNU (lomustine) 60 mg/m² q6w — CBC monitoring "
            "for myelosuppression (nadir day 7-14). Cytarabine (Ara-C) 200 mg/m² SC over "
            "48h q3w — for GME, NME. Mycophenolate mofetil (MMF) 10 mg/kg PO BID. "
            "Leflunomide as steroid-sparing. Combination protocols. Monitoring and "
            "side effect management."
        ),
        "query_terms": (
            "immunosuppressive therapy prednisone cyclosporine CCNU cytarabine lomustine "
            "MMF CNS inflammatory GME treatment immunosuppression protocol monitoring"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Treatment of Inflammatory Brain Disease (Ch 260)",
        ],
        "knowledge_supplemented": True,
    },
    {
        "id": "mri_interpretation",
        "phase": "p4",
        "title": "MRI Interpretation: Brain and Spine",
        "description": (
            "Sequence rules: T1 (anatomy, fat bright, fluid dark), T2 (pathology bright "
            "— edema, CSF, disc), FLAIR (suppresses CSF, shows perilesional edema), "
            "DWI (acute ischemia bright), post-contrast T1 (disrupted BBB enhances). "
            "Intra-axial vs extra-axial brain lesions (broad dural base = extra-axial). "
            "Spinal MRI: disc material (T2 dark), cord signal change (T2 bright = bad), "
            "meningeal enhancement (leptomeningeal disease)."
        ),
        "query_terms": (
            "MRI brain spine T1 T2 FLAIR DWI signal intensity contrast enhancement "
            "intra-axial extra-axial lesion perilesional edema disc material cord signal"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Diagnostic Imaging of the Nervous System (Ch 260)",
        ],
        "knowledge_supplemented": True,
    },
    {
        "id": "spinal_neurosurgery",
        "phase": "p4",
        "title": "Spinal Neurosurgery: Procedures and Decision-Making",
        "description": (
            "Hemilaminectomy (TL disc): unilateral dorsolateral approach, remove articular "
            "facet, identify disc in ventral epidural space. Ventral slot (cervical): "
            "ventral midline, between longus colli muscles — risk: vertebral venous sinus "
            "hemorrhage. Dorsal laminectomy (LS): cauda equina decompression. "
            "Spinal stabilization: vertebral body plating, pins/PMMA. Timing of surgery: "
            "Grade 5 (no deep pain) → surgery within 24-48h improves prognosis. "
            "Grade 3-4 → surgery if no improvement in 24-48h medical management."
        ),
        "query_terms": (
            "hemilaminectomy ventral slot dorsal laminectomy spinal surgery decompression "
            "cervical thoracolumbar stabilization surgical timing deep pain prognosis"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Surgical Management of Spinal Disease (Ch 265)",
        ],
        "knowledge_supplemented": True,
    },
    {
        "id": "intracranial_neurosurgery",
        "phase": "p4",
        "title": "Intracranial Neurosurgery and Complex Integration",
        "description": (
            "Craniotomy approaches: rostrotentorial (forebrain), caudotentorial (caudal "
            "fossa). Intracranial meningioma: most resectable CNS tumor in cats — "
            "median survival 26 months post-surgery vs 5 months without. Pituitary "
            "macroadenoma: signs, hypophysectomy or radiation. Neuroanesthesia: TIVA "
            "(propofol/fentanyl), avoid volatile agents that increase ICP, avoid "
            "hypercapnia. Complex boards integration: case-based multi-system reasoning."
        ),
        "query_terms": (
            "craniotomy intracranial surgery meningioma pituitary tumor neuroanesthesia "
            "brain tumor cat rostrotentorial caudotentorial TIVA propofol fentanyl"
        ),
        "read_refs": [
            "Ettinger Internal Medicine — Intracranial Surgery and Brain Tumors (Ch 260)",
        ],
        "knowledge_supplemented": True,
    },
]

PHASE_MAP = {p["id"]: p for p in PHASES}
TOPIC_MAP = {t["id"]: t for t in TOPICS}

# Topics grouped by phase, in curriculum order
PHASE_TOPICS: dict[str, list[dict]] = {}
for _t in TOPICS:
    PHASE_TOPICS.setdefault(_t["phase"], []).append(_t)
