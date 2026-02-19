# TruPharma — Implementation Instructions for AI Agent

> **READ THIS FIRST.** This document contains everything an AI coding agent needs to implement the remaining TruPharma features. It is self-contained — no prior conversation context is needed.
>
> **Reference Document:** `TruPharma_Technical_Architecture_Guide.md` in this same repo contains the full technical rationale, architecture diagrams, and field-level API documentation. Read it if you need deeper context on any design decision.

---

## Project Overview (30-Second Summary)

TruPharma is a RAG-based drug safety verification tool. Users ask drug questions; the system answers by cross-referencing **official FDA drug labels** with **real-world adverse event reports (FAERS)** to show where patient experience diverges from the label.

**Workspace:** `/Users/nithinreddy/TruPharma-MVP`
**Tech Stack:** Python 3.11, Streamlit, FAISS, BM25, Google Gemini 2.0 Flash, OpenFDA APIs, RxNorm API
**Deployed at:** `trupharm.streamlit.app`

---

## Current Codebase Structure

```
/Users/nithinreddy/TruPharma-MVP/
├── src/
│   ├── openfda_rag.py             # OpenFDA API client, chunking, indexing (533 lines)
│   ├── rag_engine.py              # RAG pipeline: retrieve → generate → log (437 lines)
│   ├── Week 4.ipynb               # Development notebook
│   └── app/
│       ├── .streamlit/config.toml # Streamlit theme
│       ├── streamlit_app.py       # Main Streamlit UI (343 lines)
│       └── pages/
│           └── stress_test.py     # Stress test page (316 lines)
├── data/                          # Empty placeholder
├── logs/
│   └── product_metrics.csv        # Interaction logs
├── requirements.txt               # Python dependencies
├── README.md                      # Project readme
├── TruPharma_Technical_Architecture_Guide.md  # Full architecture doc
├── Week4_Integration_Report.md
└── Week4_Integration_Report.pdf
```

---

## What Currently Works

- Streamlit app with query input and response display
- Fetches drug label records from OpenFDA `/drug/label/` API in real-time
- Hybrid retrieval: FAISS (dense, TF-IDF vectors) + BM25 (sparse) with reciprocal rank fusion
- Answer generation via Google Gemini 2.0 Flash (optional) or extractive fallback
- Confidence scoring, inline citations `[doc_id::field]`, CSV logging
- Stress test page with edge-case scenarios

## What Currently Does NOT Work / Is Missing

- **Only 10 of 70+ drug label fields are used** (hardcoded allowlist in `rag_engine.py` line 46-57)
- **No FAERS adverse event data** — the core "real-world evidence" half of the system
- **No NDC product metadata** — no structured product identity
- **No RxNorm entity resolution** — no brand→generic mapping, no fuzzy matching / spell-check
- **No agentic orchestration** (Agents A/B/C)
- **No Signal Heatmap** dashboard
- **No dual-source validation** (label vs. FAERS comparison)
- **No disparity analysis**
- **No evaluation framework** (A/B testing, hallucination audit, BLUE benchmark)

---

## Implementation Plan — Execute in This Order

### PHASE 1: Data Infrastructure

#### Step 1.1 — Expand Drug Labels to ALL Fields

**File to modify:** `src/rag_engine.py`

**What to do:** Replace the 10-field allowlist with a blocklist approach so ALL clinically relevant fields are included.

**Current code (lines 46-58):**
```python
FIELD_ALLOWLIST: List[str] = [
    "active_ingredient",
    "description",
    "dosage_and_administration",
    "drug_interactions",
    "information_for_patients",
    "when_using",
    "overdosage",
    "stop_use",
    "user_safety_warnings",
    "warnings",
]
FIELD_BLOCKLIST = {"spl_product_data_elements"}
```

**Replace with:**
```python
FIELD_ALLOWLIST = None  # None = include ALL fields (blocklist filters the noise)

FIELD_BLOCKLIST = {
    "spl_product_data_elements",
    "spl_indexing_data_elements",
    "effective_time",
    "set_id",
    "id",
    "version",
    "openfda",
    "package_label_principal_display_panel",
}
```

**Also update** the `run_rag_query` function's call to `build_openfda_query` — currently it passes `fields=FIELD_ALLOWLIST` which tries to build per-field search queries. When `FIELD_ALLOWLIST` is `None`, use a simpler general text search instead. The `build_openfda_query` function in `openfda_rag.py` already handles `fields=None` by returning a general text search.

**Verify:** Run the app locally, query "ibuprofen warnings" — you should now see chunks from fields like `adverse_reactions`, `contraindications`, `boxed_warning`, `clinical_pharmacology`, etc., not just the original 10.

---

#### Step 1.2 — Build RxNorm Resolution Module

**Create new file:** `src/rxnorm_resolver.py`

**Purpose:** Resolve drug names to canonical identifiers using the free RxNorm API. This is the first step in every query — before hitting any OpenFDA endpoint.

**API Details:**
- Base URL: `https://rxnav.nlm.nih.gov/REST/`
- No authentication needed
- Rate limit: 20 requests/second
- Returns JSON

**Functions to implement:**

```python
def resolve_drug_name(name: str) -> dict:
    """
    Master resolution function. Takes any drug name input and returns:
    {
        "input": "advil",
        "resolved_name": "ibuprofen",
        "rxcui": "5640",
        "brand_names": ["Advil", "Motrin", "Nuprin", ...],
        "generic_name": "ibuprofen",
        "all_rxcuis": ["153008", "206878", "5640", ...],
        "spelling_suggestion": None,  # populated if input was misspelled
        "confidence": "exact"  # or "approximate" or "spelling_corrected"
    }
    """
```

**Implementation approach:**

1. First try exact match: `GET /rxcui.json?name={name}&search=2`
   - If found → get rxcui, then get related info
2. If no exact match, try drug name lookup: `GET /drugs.json?name={name}`
   - Parses brand names and ingredients from the response
3. If still no match, try approximate match: `GET /approximateTerm.json?term={name}&maxEntries=5`
   - Returns fuzzy matches for misspelled names
4. If still nothing, try spelling suggestions: `GET /spellingsuggestions.json?name={name}`
   - Returns corrected spellings

**Helper functions:**
```python
def get_rxcui_by_name(name: str) -> Optional[str]:
    """Get RxCUI for an exact drug name."""

def get_drug_info(name: str) -> dict:
    """Get drug concepts (brand/generic) from /drugs endpoint."""

def get_approximate_match(term: str) -> list:
    """Fuzzy match for misspelled drug names."""

def get_spelling_suggestions(name: str) -> list:
    """Get spelling corrections."""

def get_related_brands(rxcui: str) -> list:
    """Get all brand names for a given RxCUI."""

def get_generic_from_brand(rxcui: str) -> Optional[str]:
    """Resolve a brand product RxCUI to its generic ingredient."""
```

**All HTTP calls** should use `urllib.request` (already used in `openfda_rag.py`) for consistency. Add a 10-second timeout and graceful error handling that returns empty results on failure rather than crashing.

---

#### Step 1.3 — Build FAERS Adverse Event Ingestion Module

**Create new file:** `src/faers_ingestion.py`

**Purpose:** Fetch and aggregate adverse event data from the FAERS API into text summaries suitable for RAG chunking.

**API Details:**
- Base URL: `https://api.fda.gov/drug/event.json`
- No authentication needed (API key optional for higher limits)
- Supports `count` parameter for server-side aggregation — USE THIS instead of downloading individual reports

**Key design decision:** Do NOT download thousands of individual FAERS reports. Instead, use the `count` endpoint to get pre-aggregated statistics from the server. This is dramatically faster and more efficient.

**Functions to implement:**

```python
def fetch_faers_summary(generic_name: str, rxcuis: List[str] = None) -> dict:
    """
    Fetch aggregated FAERS data for a drug. Returns:
    {
        "drug_name": "ibuprofen",
        "total_reports": 45231,
        "top_reactions": [{"term": "Nausea", "count": 3201}, ...],
        "seriousness": {"death": 1203, "hospitalization": 15678, ...},
        "reporter_types": {"physician": 34, "consumer": 42, ...},  # percentages
        "patient_sex": {"male": 35, "female": 58, "unknown": 7},
        "patient_age_groups": {"adult": 65, "elderly": 22, ...},
        "recent_reports": [...],  # last 90 days if available
        "sample_narratives": [...],  # 5-10 individual case summaries
    }
    """
```

**API calls to make (all in parallel if possible):**

1. Total report count:
   `GET /drug/event.json?search=patient.drug.openfda.generic_name:"{name}"&limit=1`
   → Read `meta.results.total`

2. Top adverse reactions (aggregated):
   `GET /drug/event.json?search=patient.drug.openfda.generic_name:"{name}"&count=patient.reaction.reactionmeddrapt.exact`
   → Returns list of `[{"term": "NAUSEA", "count": 3201}, ...]`

3. Seriousness breakdown:
   - Serious count: `?search=...+AND+serious:1&limit=1` → `meta.results.total`
   - Deaths: `?search=...+AND+seriousnessdeath:1&limit=1` → `meta.results.total`
   - Hospitalizations: `?search=...+AND+seriousnesshospitalization:1&limit=1` → `meta.results.total`
   - Life-threatening: `?search=...+AND+seriousnesslifethreatening:1&limit=1` → `meta.results.total`
   - Disabling: `?search=...+AND+seriousnessdisabling:1&limit=1` → `meta.results.total`

4. Reporter type distribution:
   `?search=...&count=primarysource.qualification`

5. Patient sex distribution:
   `?search=...&count=patient.patientsex`

6. Patient age group distribution:
   `?search=...&count=patient.patientagegroup`

7. Sample individual reports (for representative narratives):
   `?search=...&limit=10&sort=receivedate:desc`
   → Fetch 10 most recent reports, extract drug info, reactions, and outcomes

**Then synthesize into readable text:**

```python
def format_faers_as_text(summary: dict) -> str:
    """
    Convert aggregated FAERS data into a readable text block
    suitable for RAG chunking. Returns something like:

    "REAL-WORLD ADVERSE EVENT SUMMARY FOR IBUPROFEN
    Source: FDA FAERS | Total Reports: 45,231

    Top Reported Adverse Reactions:
    1. Nausea — 3,201 reports (7.1%)
    2. Drug ineffective — 2,845 reports (6.3%)
    ...

    Seriousness: 62.9% serious, 2.7% deaths, 34.7% hospitalizations
    ...

    Representative Case: 65-year-old female reported nausea, abdominal pain
    after taking ibuprofen 400mg orally for arthritis. Outcome: recovered.
    ..."
    """
```

**Search query construction:** The FAERS API search supports:
- `patient.drug.openfda.generic_name:"{name}"` — by generic name (most reliable)
- `patient.drug.openfda.rxcui:"{rxcui}"` — by RxCUI
- `patient.drug.openfda.brand_name:"{brand}"` — by brand name
- `patient.drug.medicinalproduct:"{name}"` — by raw drug name field (less reliable, may include misspellings)

Prefer searching by `generic_name`. Fall back to `medicinalproduct` if the drug doesn't have harmonized `openfda` fields.

**Error handling:** If the API returns 404 or 0 results, return an empty summary dict with `total_reports: 0`. Never crash.

---

#### Step 1.4 — Build NDC Product Metadata Module

**Create new file:** `src/ndc_ingestion.py`

**Purpose:** Fetch structured product metadata from the NDC Directory API.

**API Details:**
- Base URL: `https://api.fda.gov/drug/ndc.json`
- No authentication needed

**Functions to implement:**

```python
def fetch_ndc_metadata(generic_name: str, brand_name: str = None, rxcui: str = None) -> dict:
    """
    Fetch NDC product metadata. Returns:
    {
        "brand_names": ["ADVIL", "MOTRIN"],
        "generic_name": "ibuprofen",
        "manufacturer": "Haleon US Holdings LLC",
        "active_ingredients": [{"name": "IBUPROFEN", "strength": "200 mg/1"}],
        "dosage_forms": ["TABLET, COATED", "CAPSULE", "SUSPENSION"],
        "routes": ["ORAL"],
        "marketing_category": "NDA",
        "application_number": "NDA018989",
        "product_type": "HUMAN OTC DRUG",
        "pharm_class_epc": ["Nonsteroidal Anti-inflammatory Drug [EPC]"],
        "pharm_class_moa": ["Cyclooxygenase Inhibitors [MoA]"],
        "pharm_class_cs": ["Anti-Inflammatory Agents, Non-Steroidal [CS]"],
        "dea_schedule": None,
        "product_ndcs": ["0573-0154", ...],
        "rxcuis": ["153008", "310965"],
    }
    """
```

**Search strategy:**
1. Try by rxcui: `?search=openfda.rxcui:"{rxcui}"&limit=10`
2. Or by brand name: `?search=brand_name:"{brand}"&limit=10`
3. Or by generic name: `?search=generic_name:"{generic}"&limit=10`

Fetch multiple results and merge/deduplicate (same drug may have multiple NDC entries for different package sizes).

```python
def format_ndc_as_text(metadata: dict) -> str:
    """
    Format NDC metadata into readable text for RAG:

    "PRODUCT IDENTITY: ADVIL (ibuprofen)
    Manufacturer: Haleon US Holdings LLC
    Pharmacologic Class: Nonsteroidal Anti-inflammatory Drug [EPC]
    Mechanism of Action: Cyclooxygenase Inhibitors [MoA]
    Dosage Forms: TABLET, COATED; CAPSULE; SUSPENSION
    Route: ORAL
    Active Ingredients: IBUPROFEN 200 mg
    Product Type: HUMAN OTC DRUG
    Marketing: NDA (NDA018989)"
    """
```

---

#### Step 1.5 — Build Unified Drug Profile Assembler

**Create new file:** `src/drug_profile.py`

**Purpose:** Orchestrate all data sources into a single unified drug profile, then produce chunks for the RAG engine.

**This is the central orchestrator.** It calls RxNorm → then fetches Labels + FAERS + NDC in parallel → assembles the unified document → computes disparity analysis.

**Functions to implement:**

```python
def build_unified_profile(user_query: str) -> dict:
    """
    Master function. Takes a user query, resolves the drug, fetches all data,
    and returns:
    {
        "drug_identity": { ... },         # from RxNorm + NDC
        "label_data": { ... },            # from /drug/label (all fields)
        "faers_summary": { ... },         # from /drug/event (aggregated)
        "ndc_metadata": { ... },          # from /drug/ndc
        "disparity_analysis": { ... },    # computed
        "text_sections": [                # ready for chunking
            {"source": "ndc", "field": "product_identity", "text": "..."},
            {"source": "label", "field": "warnings", "text": "..."},
            {"source": "label", "field": "adverse_reactions", "text": "..."},
            {"source": "faers", "field": "top_reactions", "text": "..."},
            {"source": "faers", "field": "seriousness", "text": "..."},
            {"source": "disparity", "field": "analysis", "text": "..."},
            ...
        ]
    }
    """
```

**Disparity analysis logic:**
```python
def compute_disparity(label_adverse_reactions: str, faers_top_reactions: list) -> dict:
    """
    Compare adverse reactions mentioned in the label text against
    top reactions from FAERS reports.

    Returns:
    {
        "reactions_in_faers_not_on_label": [...],    # emerging signals
        "reactions_on_label_with_high_faers": [...],  # confirmed risks
        "reactions_on_label_with_low_faers": [...],   # potentially over-warned
        "disparity_score": 0.73,                      # 0-1 quantified gap
    }
    """
```

To match label text against FAERS reactions: extract medical terms from the label's `adverse_reactions` field using simple keyword matching against the MedDRA terms in FAERS reaction counts. This doesn't need to be perfect — approximate matching is fine for a disparity signal.

**Integration with existing code:** The `text_sections` list in the return value replaces the current `record_chunks` from `openfda_rag.py`. Each section becomes a `TextChunk` (same dataclass already defined in `openfda_rag.py`) that flows into the existing FAISS + BM25 indexing pipeline.

---

### PHASE 2: RAG Pipeline Upgrade

#### Step 2.1 — Upgrade rag_engine.py

**File to modify:** `src/rag_engine.py`

**What to change:**

1. **Import the new modules** at the top:
```python
from drug_profile import build_unified_profile
from rxnorm_resolver import resolve_drug_name
```

2. **Replace the `run_rag_query` function's data fetching logic.**

Currently it does:
```python
search_q = build_openfda_query(query, fields=FIELD_ALLOWLIST)
arts = build_artifacts(api_search=search_q, field_allowlist=FIELD_ALLOWLIST, ...)
```

Replace with:
```python
profile = build_unified_profile(query)
# Then convert profile["text_sections"] into TextChunk objects
# and feed them into the existing indexing pipeline (FAISS + BM25)
```

The retrieval, fusion, answer generation, and logging code should stay mostly the same. The key change is the **input data** is now multi-source instead of label-only.

3. **Update the system prompt** (`_RAG_SYSTEM` variable) to instruct the LLM about dual-source evidence:

```python
_RAG_SYSTEM = (
    "You are TruPharma Assistant, a drug safety verification tool.\n"
    "You have access to TWO types of evidence:\n"
    "1. OFFICIAL FDA DRUG LABELS — the regulatory ground truth\n"
    "2. REAL-WORLD FAERS DATA — what patients actually report experiencing\n\n"
    "Answer the question using ONLY the retrieved evidence below.\n"
    "When both label and FAERS data are available, compare them.\n"
    "Cite every claim with [source::field] notation.\n"
    "If a side effect appears in FAERS but NOT on the label, flag it as an 'emerging signal'.\n"
    "If evidence is insufficient, respond: "
    '"Not enough evidence in the retrieved context."\n'
    "Do NOT fabricate facts. Do NOT give medical advice."
)
```

4. **Update citation format** to include source type:
   - Label evidence: `[label::warnings]`, `[label::adverse_reactions]`
   - FAERS evidence: `[faers::top_reactions]`, `[faers::seriousness]`
   - NDC evidence: `[ndc::product_identity]`
   - Disparity: `[disparity::analysis]`

---

### PHASE 3: Agentic Orchestration

#### Step 3.1-3.3 — Implement Agents A/B/C

**Create new file:** `src/agents.py`

**Implementation approach:** Use structured Gemini prompts (not LangChain, to keep dependencies minimal). Each "agent" is a focused prompt + retrieval scope.

```python
def run_agent_a(query: str, label_chunks: list, gemini_key: str) -> str:
    """Agent A (Regulator): Extracts official safety info from label chunks only."""

def run_agent_b(query: str, faers_chunks: list, gemini_key: str) -> str:
    """Agent B (Observer): Identifies real-world patterns from FAERS chunks only."""

def run_agent_c(query: str, agent_a_output: str, agent_b_output: str, gemini_key: str) -> str:
    """Agent C (Verifier): Reconciles A + B, identifies confirmed risks vs emerging signals."""
```

Agent C's prompt should explicitly compare the two outputs and categorize findings into:
- **Confirmed risks** — on the label AND in FAERS
- **Emerging signals** — in FAERS but NOT on the label
- **Unconfirmed warnings** — on the label but rarely in FAERS

---

### PHASE 4: Frontend Upgrade

#### Step 4.1 — Upgrade Safety Chat

**File to modify:** `src/app/streamlit_app.py`

Changes:
- Add RxNorm drug name resolution before search (show resolved name to user)
- Display dual-source evidence panels (label evidence vs. FAERS evidence side by side)
- Show disparity analysis results
- Show drug identity metadata from NDC

#### Step 4.2 — Build Signal Heatmap Page

**Create new file:** `src/app/pages/signal_heatmap.py`

A Streamlit page that:
- Lets analysts select from a list of common drugs (or enter custom)
- Fetches FAERS count data for selected drugs
- Displays a heatmap (using Plotly or Altair) showing:
  - X-axis: Drugs
  - Y-axis: Adverse reactions
  - Color intensity: Disparity score (green = aligned with label, red = not on label)
- Shows a sortable table of drugs ranked by disparity score

---

### PHASE 5: Evaluation Framework

#### Step 5.1 — A/B Testing

**Create new file:** `src/evaluation/ab_test.py`

Compare three systems on the same set of 50+ test queries:
1. **Vanilla LLM** — Gemini with no RAG (just the question)
2. **Label-only RAG** — current system (labels only)
3. **Full TruPharma** — unified profile with all sources

Measure: answer groundedness, hallucination rate, citation precision, latency.

#### Step 5.2-5.3 — Grounding Audit + BLUE Benchmark

These are evaluation scripts, not core features. Implement after the main pipeline works.

---

## Important Technical Details

### API Endpoints Quick Reference

```
OpenFDA Drug Labels:  https://api.fda.gov/drug/label.json
OpenFDA FAERS Events: https://api.fda.gov/drug/event.json
OpenFDA NDC:          https://api.fda.gov/drug/ndc.json
RxNorm:               https://rxnav.nlm.nih.gov/REST/
```

All free. No authentication required. No API keys needed (optional for higher rate limits on OpenFDA).

### Cross-Dataset Join Keys

The `openfda` harmonized object exists in ALL three OpenFDA endpoints. The universal join keys are:
- **`openfda.rxcui`** — strongest link (present in labels, events, NDC, and RxNorm)
- **`openfda.generic_name`** — text-based fallback
- **`openfda.product_ndc`** — product-level link
- **`openfda.spl_set_id`** — label document identifier

### FAERS Count Endpoint — Critical Performance Feature

Instead of downloading individual reports, use the `count` parameter:
```
GET /drug/event.json?search=patient.drug.openfda.generic_name:"ibuprofen"&count=patient.reaction.reactionmeddrapt.exact
```
This returns server-side aggregated counts. Use it for: reactions, seriousness, reporter types, patient sex, age groups, and temporal trends.

### Drug Label Blocklist (replaces the current 10-field allowlist)

```python
FIELD_BLOCKLIST = {
    "spl_product_data_elements",
    "spl_indexing_data_elements",
    "effective_time",
    "set_id",
    "id",
    "version",
    "openfda",
    "package_label_principal_display_panel",
}
```

Everything NOT in this blocklist gets included — this captures 40+ clinically relevant fields.

### HTTP Requests

Use `urllib.request` (already used in `openfda_rag.py`) for all API calls. This avoids adding `requests` as a dependency. Pattern:

```python
import json, urllib.request, urllib.parse, urllib.error

def _api_get(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "TruPharma/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return {}
```

### Existing Dataclasses (in openfda_rag.py — reuse these)

```python
@dataclass
class TextChunk:
    chunk_id: str   # e.g., "label::warnings::c1" or "faers::top_reactions"
    doc_id: str     # e.g., "ibuprofen" or an SPL ID
    field: str      # e.g., "warnings", "top_reactions", "product_identity"
    text: str       # the actual text content
```

### Dependencies

Current `requirements.txt`:
```
streamlit>=1.32.0
numpy>=1.24.0
scikit-learn>=1.3.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
google-generativeai>=0.5.0
pandas>=2.0.0
```

The new modules (RxNorm, FAERS, NDC) use only `urllib` (stdlib) and `json` (stdlib), so **no new dependencies are needed** for Phase 1.

If agentic orchestration (Phase 3) uses LangChain, add `langchain` and `langchain-google-genai`. But prefer custom Gemini prompts to keep it simple.

---

## Execution Order Summary

```
Phase 1 (DO FIRST — foundation for everything):
  1.1  Expand drug label fields (modify rag_engine.py)           ← 15 min
  1.2  Build rxnorm_resolver.py                                  ← 1-2 hours
  1.3  Build faers_ingestion.py                                  ← 2-3 hours
  1.4  Build ndc_ingestion.py                                    ← 1 hour
  1.5  Build drug_profile.py (orchestrator)                      ← 2-3 hours

Phase 2 (integrates Phase 1 into working pipeline):
  2.1  Upgrade rag_engine.py to use unified profiles             ← 2-3 hours

Phase 3 (adds intelligence layer):
  3.1-3.3  Build agents.py (Agents A/B/C)                       ← 2-3 hours

Phase 4 (makes it visible):
  4.1  Upgrade streamlit_app.py for dual-source display          ← 2-3 hours
  4.2  Build signal_heatmap.py page                              ← 2-3 hours

Phase 5 (proves it works):
  5.1-5.4  Build evaluation scripts                              ← 3-4 hours
```

**Start with Phase 1. Each step builds on the previous one. Do not skip ahead.**

---

## Testing Each Step

After each step, verify it works:

- **Step 1.1:** Run the app, query "ibuprofen warnings" — confirm chunks from `adverse_reactions`, `contraindications`, `boxed_warning` etc. appear (not just the original 10 fields)
- **Step 1.2:** Call `resolve_drug_name("Advil")` — should return `{"generic_name": "ibuprofen", "rxcui": "5640", ...}`
- **Step 1.2 edge case:** Call `resolve_drug_name("Avdil")` (typo) — should still resolve via approximate match or spelling suggestion
- **Step 1.3:** Call `fetch_faers_summary("ibuprofen")` — should return aggregated reaction counts, seriousness data, demographics
- **Step 1.4:** Call `fetch_ndc_metadata("ibuprofen")` — should return manufacturer, dosage forms, pharm class
- **Step 1.5:** Call `build_unified_profile("What are the side effects of Advil?")` — should return a complete profile with sections from all sources
- **Step 2.1:** Run the full app — responses should now cite both `[label::...]` and `[faers::...]` sources

---

*This document is the single source of truth for implementation. When in doubt, refer to `TruPharma_Technical_Architecture_Guide.md` for deeper rationale.*
