# TruPharma Knowledge Graph — Implementation Plan

**Purpose:** This document is the single source of truth for implementing **Phase 1** of the TruPharma knowledge graph (KG). A new agent or developer should be able to execute the KG task using only this document and the existing codebase.

**Last updated:** 2026-02-20

---

## 1. Project Context (What TruPharma Is)

- **TruPharma** is a Drug Label Evidence RAG application. It answers drug-related questions using:
  - **openFDA Drug Label API** — official FDA labels (fields like `drug_interactions`, `warnings`, `dosage_and_administration`, etc.).
  - **RxNorm API** — entity resolution (brand ↔ generic, RxCUI, fuzzy/spell-check).
  - **OpenFDA FAERS** — real-world adverse event reports.
  - **OpenFDA NDC** — product metadata, active ingredients, brand/generic mapping.
- **Current stack:** Python, Streamlit, in-memory indexing (FAISS + BM25), no persistent database. Data is fetched in real time from APIs or from JSON/cache.
- **Deployment goal:** The app runs on **Streamlit Community Cloud** (or similar). It should **not** depend on the developer's laptop after deployment. All runtime resources are on the hosting platform; local disk is used only during development.
- **Relevant code:**
  - `src/ingestion/openfda_client.py` — `fetch_openfda_records()`, `pick_text_fields()`, `OPENFDA_BASE_URL`.
  - `src/ingestion/rxnorm.py` — `resolve_drug_name()`, `get_drug_info()`, `get_rxcui_by_name()`, `get_approximate_match()`, `get_generic_from_brand()`, `_get_all_related_rxcuis()`.
  - `src/ingestion/faers.py` — `fetch_faers_summary()`, FAERS count queries.
  - `src/ingestion/ndc.py` — `fetch_ndc_metadata()`, product/ingredient data.
  - `src/rag/drug_profile.py` — builds unified drug profile (labels + FAERS + NDC).
  - `src/rag/engine.py` — RAG pipeline (retrieve → generate → log).
  - `docs/TruPharma_Technical_Architecture_Guide.md` — full architecture, data sources, cross-dataset linking.
  - `docs/DataScienceProjectProposalTruPharma.pdf` — project vision and data source requirements.

---

## 2. Goal of This KG Task (Phase 1)

Implement a **multi-source** knowledge graph that:

1. **Uses all proposal data sources** (per `DataScienceProjectProposalTruPharma.pdf`):
   - **RxNorm** — drug identity (nodes)
   - **openFDA Drug Labels** — drug–drug interactions
   - **openFDA FAERS** — co-reported drugs, drug–reaction relationships
   - **openFDA NDC** — active ingredients, product metadata
   - **DrugBank** — drug–drug interactions (optional; requires academic license)

2. **Stores structured relationships** that are hard to get from text alone:
   - **Drug–drug interactions** (from labels, optionally DrugBank)
   - **Co-reported drugs** (from FAERS — drugs appearing together in adverse event reports)
   - **Drug–reaction links** (from FAERS — drug → MedDRA adverse reaction)
   - **Drug–ingredient links** (from NDC)
   - **Entity identity** (RxCUI ↔ generic name, brand names) for cross-linking

3. **Lives as a single artifact in the repo** (e.g. a SQLite file or node-link JSON) so that:
   - The **deployed app** (Streamlit) only **reads** this file at startup; no persistent write storage on the host.
   - The developer's laptop is only used to **build** the KG (or run a build script); after deployment, the laptop is not involved.

4. **Uses minimal resources:** No new cloud services, no paid DBs. SQLite and/or NetworkX + JSON are free and run in-process.

**Out of scope for Phase 1 (can be Phase 2):**  
Full label/FAERS text in the graph, explicit disparity edges (label vs FAERS), Signal Heatmap-specific graph queries. The RAG pipeline continues to handle free-text Q&A; the KG is an **add-on** for structured lookups and future multi-hop queries.

---

## 3. Preferred Stack (Decided)

- **Primary option:** **SQLite** — one file (e.g. `data/kg/trupharma_kg.db`), no server, minimal code. Good for persistence and portability; recursive CTEs for 1–2 hop traversals.
- **Alternative (optional):** **NetworkX + node-link JSON** — in-memory graph loaded from e.g. `data/kg/graph_node_link.json`; simpler graph algorithms, no SQL. Same "commit file to repo, read at startup" pattern.
- **Cost:** Both are **free** (SQLite is public domain; NetworkX is BSD). No hosting cost; the only "cost" is the size of the committed file (typically a few MB for Phase 1).

---

## 4. Schema (Phase 1 — Multi-Source)

### 4.1 Nodes

| Node type     | Primary key                    | Attributes (examples)                                                                 |
|---------------|--------------------------------|----------------------------------------------------------------------------------------|
| `Drug`        | `rxcui` (string) or `generic_name` (if no RxCUI) | `generic_name`, `brand_names` (JSON array), `rxcui`                                    |
| `Ingredient`  | ingredient name (string)       | —                                                                                     |
| `Reaction`    | MedDRA term (string)           | `reactionmeddrapt` (normalized)                                                        |

- Use **one row per drug**. Prefer `rxcui` as the unique id when available; for drugs without RxCUI, use `generic_name` as id.
- `Ingredient` and `Reaction` nodes are created when linked from NDC and FAERS, respectively.

### 4.2 Edges (Relationships)

| Edge type               | From  | To    | Source(s)       | Attributes (optional)                                              |
|-------------------------|-------|-------|----------------|--------------------------------------------------------------------|
| `INTERACTS_WITH`        | Drug A | Drug B | Labels, DrugBank | `source` ("label" or "drugbank"), `description` (short text)       |
| `CO_REPORTED_WITH`      | Drug A | Drug B | FAERS         | `source: "faers"`, `report_count` (aggregate count if available)   |
| `DRUG_CAUSES_REACTION`  | Drug   | Reaction | FAERS       | `source: "faers"`, `report_count` (optional)                       |
| `HAS_ACTIVE_INGREDIENT` | Drug   | Ingredient | NDC        | `source: "ndc"`                                                    |
| `HAS_PRODUCT`           | Drug   | NDC product (stored as node or in props) | NDC | `source: "ndc"`, `ndc_code`, product metadata                     |

- **Drug identity** (generic + brands) is stored in the `Drug` node props; `HAS_BRAND` is implicit in `brand_names`.
- Every edge should include `source` in props for provenance.

### 4.3 SQLite Table Layout (Concrete)

- **Table `nodes`:**  
  `id TEXT PRIMARY KEY, type TEXT NOT NULL, props TEXT`  
  (`props` = JSON: e.g. `{"generic_name": "ibuprofen", "brand_names": ["Advil", "Motrin"], "rxcui": "153008"}` for Drug; `{}` or minimal for Ingredient/Reaction.)

- **Table `edges`:**  
  `src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL, props TEXT, PRIMARY KEY (src, dst, type)`  
  (`src`/`dst` = node ids; `type` = e.g. `INTERACTS_WITH`, `CO_REPORTED_WITH`; `props` = JSON with `source`, `description`, etc.)

- **Indexes:** Index `edges(src, type)` and `edges(dst, type)` for fast lookups by drug.

---

## 5. Data Sources and Build Pipeline

Phase 1 uses **four primary data sources** (RxNorm + three openFDA APIs). DrugBank is optional if an academic license is obtained.

### 5.1 Data Source Overview

| Source            | API / Endpoint                       | KG Contribution                                                                 |
|-------------------|--------------------------------------|----------------------------------------------------------------------------------|
| **RxNorm**        | `https://rxnav.nlm.nih.gov/REST/`    | Drug nodes (rxcui, generic_name, brand_names); entity resolution for cross-linking |
| **openFDA Labels**| `https://api.fda.gov/drug/label.json`| `INTERACTS_WITH` edges from `drug_interactions` / `drug_interactions_table`      |
| **openFDA FAERS** | `https://api.fda.gov/drug/event.json`| `CO_REPORTED_WITH`, `DRUG_CAUSES_REACTION` edges                                 |
| **openFDA NDC**   | `https://api.fda.gov/drug/ndc.json`  | `HAS_ACTIVE_INGREDIENT`, `HAS_PRODUCT` edges; product metadata                    |
| **DrugBank**      | Academic license required            | Additional `INTERACTS_WITH` edges with `source: "drugbank"` (optional)           |

> **Note:** The RxNorm drug interaction API (`/REST/interaction/`) was **discontinued in January 2024**. Do not use it. openFDA labels and (if available) DrugBank are the interaction sources.

### 5.2 Build Order (Recommended)

Execute in this order so that each step can resolve entities using previously created nodes:

```
1. RxNorm           → Drug nodes (seed list; see §5.3)
2. openFDA NDC      → HAS_ACTIVE_INGREDIENT, HAS_PRODUCT; optionally extend Drug nodes
3. openFDA Labels   → INTERACTS_WITH edges (from drug_interactions)
4. openFDA FAERS    → CO_REPORTED_WITH, DRUG_CAUSES_REACTION edges
5. DrugBank         → INTERACTS_WITH edges (optional; if license obtained)
```

### 5.3 Drug Identity (Nodes) — RxNorm

- **Seed list:** Derive from openFDA for data-driven coverage:
  - Use openFDA label API with `count=openfda.generic_name.exact` to get top drugs by label volume.
  - Alternatively: fetch a batch of labels (e.g. limit=500) and collect distinct `openfda.generic_name` / `openfda.brand_name`, then resolve via RxNorm.
  - Target: top 200–500 drugs for Phase 1.
- **Resolution:** For each drug name, call `resolve_drug_name()` from `src/ingestion/rxnorm.py`; get `rxcui`, `generic_name`, `brand_names`.
- **Insert:** One `Drug` node per unique `rxcui` (or `generic_name` if no rxcui). Deduplicate by rxcui.

### 5.4 Drug–Drug Interactions — openFDA Labels

- **Fetch:** For each drug in the graph, fetch label records via `fetch_openfda_records()` using `search=openfda.generic_name:"{name}"` or `search=openfda.rxcui:"{rxcui}"`.
- **Parse interactions:**
  - **Prefer** `drug_interactions_table` (structured array) when present — easier to parse drug pairs.
  - **Fallback** to `drug_interactions` (prose text): extract drug names via:
    - **Option A:** Regex + keyword matching against a drug-name dictionary (resolved drugs from seed list).
    - **Option B (recommended):** Gemini API (free tier) — send prose to Gemini with a prompt: *"Extract all drug name pairs mentioned as interacting. Return JSON: [{drug1, drug2}, ...]"* — yields better recall than regex.
- **Resolve:** Each extracted name → `resolve_drug_name()` → rxcui or generic_name (node id).
- **Create:** `INTERACTS_WITH` edges with `source: "label"`, optional `description` snippet.
- **Avoid** duplicate edges (same pair, same type). Store both A→B and B→A if desired for symmetric queries.

### 5.5 Co-Reported Drugs & Drug–Reactions — openFDA FAERS

- **Co-reported drugs (`CO_REPORTED_WITH`):**  
  Drugs that appear together in the same adverse event report. Use FAERS count endpoint:
  - `search=patient.drug.openfda.generic_name:"{drug}"` with `count=patient.drug.medicinalproduct.exact` or similar to find co-reported drug pairs.
  - Or: fetch sample reports and extract `patient.drug[].medicinalproduct` / `patient.drug[].activesubstance.activesubstancename` for pairs; resolve via RxNorm; create edges.
- **Drug–reaction (`DRUG_CAUSES_REACTION`):**  
  - Use count endpoint: `search=patient.drug.openfda.generic_name:"{drug}"` + `count=patient.reaction.reactionmeddrapt.exact`.
  - Create `Reaction` nodes for MedDRA terms; link Drug → Reaction with `source: "faers"`, optional `report_count`.
- **Rate limits:** FAERS has same limits as other openFDA (240 req/min without key). Use count queries to minimize API calls; add short sleeps between batches.

### 5.6 Active Ingredients & Products — openFDA NDC

- **Fetch:** `search=openfda.rxcui:"{rxcui}"` or `search=generic_name:"{name}"` via NDC API.
- **Extract:** `active_ingredients` (or equivalent) from NDC records; product metadata (brand, dosage form, route).
- **Create:** `HAS_ACTIVE_INGREDIENT` edges (Drug → Ingredient); optionally `HAS_PRODUCT` or store product metadata in edge props.
- **Link:** Use `openfda.rxcui` from NDC when available to align with Drug nodes.

### 5.7 DrugBank (Optional)

- **If academic license obtained:** Parse DrugBank XML/JSON for drug–drug interactions; resolve drug names to node ids via RxNorm; create `INTERACTS_WITH` edges with `source: "drugbank"`.
- **Implementation:** Add a separate build step or flag (e.g. `--include-drugbank`) that runs only when DrugBank data is present.
- **Deprioritized** in Technical Architecture Guide; openFDA labels provide sufficient interaction coverage for Phase 1.

---

## 6. Where Things Live in the Repo

- **KG artifact (output of build):**  
  - `data/kg/trupharma_kg.db` (SQLite), **or**  
  - `data/kg/graph_node_link.json` (NetworkX node-link format).  
  One of these should be **committed to the repo** so the deployed app can read it without running the build.

- **Build script:**  
  - `scripts/build_kg.py` (or `src/kg/build_kg.py`).  
  - Connects to RxNorm, openFDA (labels, FAERS, NDC), and optionally DrugBank. Builds nodes and edges in the order in §5.2. Writes SQLite and/or JSON.  
  - Run **locally or in CI**, not in the Streamlit app at runtime.

- **KG loader / query helper (used by the app):**  
  - `src/kg/loader.py`:  
    - `load_kg(path="data/kg/trupharma_kg.db")` → returns a wrapper with:
      - `get_interactions(rxcui_or_name)` → List[dict]
      - `get_drug_identity(rxcui_or_name)` → Optional[dict]
      - `get_co_reported(rxcui_or_name)` → List[dict] (Phase 1)
      - `get_drug_reactions(rxcui_or_name)` → List[dict] (Phase 1)
      - `get_ingredients(rxcui_or_name)` → List[dict] (Phase 1)
  - Handle "file not found" by returning empty/None so the app runs without the KG.

- **Dependencies:**  
  - SQLite: none (stdlib).  
  - NetworkX (if used): add `networkx` to `requirements.txt`.  
  - Gemini (for extraction): existing `google-generativeai` if used; otherwise optional.

---

## 7. Integration with the Existing App

- **RAG / Drug profile:** When the RAG engine or drug profile needs structured data, call the **KG module**:
  - Load the KG once at startup (or on first use) from `data/kg/trupharma_kg.db`.
  - Use `get_interactions()`, `get_drug_identity()`, `get_co_reported()`, `get_drug_reactions()`, `get_ingredients()` to enrich responses.
- **Graceful degradation:** The app remains functional **without** the KG. If the file is missing, skip KG lookups and rely on existing API-based behavior. The KG is **additive**, not required for basic RAG.

---

## 8. Step-by-Step Implementation Tasks (For an Agent)

Execute in this order:

1. **Create directory and placeholder:**  
   - Ensure `data/kg/` exists. Add a `.gitkeep` or README.  
   - Decide: committed artifact = `trupharma_kg.db` or `graph_node_link.json` (or both). Document here.

2. **Implement schema in SQLite:**  
   - Create tables `nodes` and `edges` as in §4.3. Add indexes on `edges(src, type)` and `edges(dst, type)`. Support node types: Drug, Ingredient, Reaction.

3. **Implement node creation (RxNorm):**  
   - Build seed list: use openFDA label API `count=openfda.generic_name.exact` or batch fetch; collect top 200–500 drugs.  
   - For each, call `resolve_drug_name()`; deduplicate by rxcui.  
   - Insert Drug nodes. Optionally create Ingredient/Reaction nodes as placeholders for later steps.

4. **Implement edge creation (openFDA NDC):**  
   - For each Drug, fetch NDC records; extract `active_ingredients` and product metadata.  
   - Create Ingredient nodes if not exists; create `HAS_ACTIVE_INGREDIENT` and `HAS_PRODUCT` edges. Respect rate limits.

5. **Implement edge creation (openFDA Labels):**  
   - For each Drug, fetch labels; read `drug_interactions_table` first, then `drug_interactions` prose.  
   - Extract drug pairs (prefer table; for prose, use Gemini or regex + drug dictionary). Resolve to node ids; create `INTERACTS_WITH` edges with `source: "label"`. Avoid duplicates.

6. **Implement edge creation (openFDA FAERS):**  
   - Use count endpoints: for each Drug, get co-reported drugs and top reactions.  
   - Create Reaction nodes; create `CO_REPORTED_WITH` and `DRUG_CAUSES_REACTION` edges with `source: "faers"`. Respect rate limits.

7. **Implement DrugBank step (optional):**  
   - If DrugBank data available: parse, resolve names, add `INTERACTS_WITH` with `source: "drugbank"`. Make this step conditional.

8. **Write the build script:**  
   - Runnable as `python scripts/build_kg.py` (or `python -m src.kg.build_kg`).  
   - Accept optional flags: `--include-drugbank`, `--output-json`.  
   - Output: `data/kg/trupharma_kg.db` (and optionally `graph_node_link.json`).  
   - Add sleeps between API batches. Cache RxNorm lookups during build to reduce calls.

9. **Implement loader/query API:**  
   - `src/kg/loader.py`: load SQLite; expose `get_interactions`, `get_drug_identity`, `get_co_reported`, `get_drug_reactions`, `get_ingredients`. Return empty/None on file-not-found.

10. **Wire the loader into the app:**  
    - In RAG engine or drug profile, when a drug is resolved, call KG to enrich with interactions, co-reported drugs, reactions, ingredients. Keep changes minimal; disabling KG must not break the app.

11. **Run the build and commit the artifact:**  
    - Run `build_kg.py`; commit `data/kg/trupharma_kg.db` (and/or `graph_node_link.json`).  
    - Document: "To refresh the KG, run `python scripts/build_kg.py` and commit the updated file."

12. **Update docs:**  
    - Confirm paths in §6. Add "Knowledge Graph" section to main README with link to this plan and rebuild instructions.

---

## 9. Success Criteria

- A new agent can read this document and implement the KG without guessing the goal or the schema.  
- The KG uses **all four primary data sources** (RxNorm, openFDA Labels, FAERS, NDC) and optionally DrugBank.  
- The KG artifact is a single file (SQLite or JSON) in `data/kg/`, committed to the repo.  
- The deployed app (Streamlit) runs without the developer's laptop; it only reads the committed file.  
- The app works if the KG file is missing (graceful degradation).  
- Resource use remains minimal (no new paid services; small file size, e.g. few MB).  
- Edges include `source` for provenance (label, faers, ndc, drugbank).

---

## 10. References

- **TruPharma architecture:** `docs/TruPharma_Technical_Architecture_Guide.md`  
- **Project proposal & data sources:** `docs/DataScienceProjectProposalTruPharma.pdf`  
- **Cross-dataset linking (RxCUI):** Technical Architecture Guide, §4  
- **openFDA Drug Label API:** `https://api.fda.gov/drug/label.json`  
- **openFDA FAERS API:** `https://api.fda.gov/drug/event.json`  
- **openFDA NDC API:** `https://api.fda.gov/drug/ndc.json`  
- **RxNorm API base:** `https://rxnav.nlm.nih.gov/REST/`  
- **Existing ingestion:** `src/ingestion/openfda_client.py`, `src/ingestion/rxnorm.py`, `src/ingestion/faers.py`, `src/ingestion/ndc.py`  
- **Unified drug profile:** `src/rag/drug_profile.py`

---

*End of Knowledge Graph Implementation Plan — Phase 1*
