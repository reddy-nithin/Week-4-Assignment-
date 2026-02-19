# TruPharma: Technical Architecture & Implementation Guide

> **Project:** TruPharma — GenAI-Powered Verification Engine for Real-World Drug Safety
> **Team:** CiteRx — Nithin Songala, Salman Mirza, Amy Ngo
> **Course:** CS 5588 · Spring 2026
> **Document Purpose:** Technical reference for project reports, presentations, and implementation decisions

---

## Table of Contents

1. [Project Vision & Problem Statement](#1-project-vision--problem-statement)
2. [Current MVP State Assessment](#2-current-mvp-state-assessment)
3. [Data Sources — Complete API Analysis](#3-data-sources--complete-api-analysis)
4. [Cross-Dataset Linking Strategy](#4-cross-dataset-linking-strategy)
5. [Architecture — Snowflake-Free Design](#5-architecture--snowflake-free-design)
6. [Data Ingestion Pipeline](#6-data-ingestion-pipeline)
7. [Drug Label Field Expansion](#7-drug-label-field-expansion)
8. [FAERS Adverse Event Ingestion Strategy](#8-faers-adverse-event-ingestion-strategy)
9. [Unified Drug Profile Model](#9-unified-drug-profile-model)
10. [RAG Pipeline Architecture](#10-rag-pipeline-architecture)
11. [Agentic Orchestration Design](#11-agentic-orchestration-design)
12. [Frontend — Safety Chat & Signal Heatmap](#12-frontend--safety-chat--signal-heatmap)
13. [Evaluation Framework](#13-evaluation-framework)
14. [Implementation Phases & Roadmap](#14-implementation-phases--roadmap)
15. [Technology Decisions & Rationale](#15-technology-decisions--rationale)

---

## 1. Project Vision & Problem Statement

### The Latency Gap

There is a critical latency gap between the "official" safety profiles of drugs (derived from limited clinical trials) and "real-world" safety (what actually happens to millions of patients post-market). The FDA FAERS dataset contains over 15 million patient narratives describing adverse events, but this data is unstructured, noisy, and disconnected from the official National Drug Code (NDC) labeling.

### What TruPharma Does

TruPharma bridges this gap by implementing a Hybrid Retrieval-Augmented Generation (RAG) architecture that harmonizes:

- **Official "ground truth"** — structured regulatory drug labels (NDC/SPL)
- **Real-world patient evidence** — unstructured adverse event reports (FAERS)

The system functions as a **"Check Engine" light for personal health**, distilling millions of noisy reports into clear, data-backed assessments that answer: *"Is what patients actually experience different from what the official drug label says?"*

### Dual-Purpose System

| Interface | User | Purpose |
|-----------|------|---------|
| **Safety Chat** | Patients, Clinicians, Pharmacists | Natural language Q&A with dual-source validation |
| **Signal Heatmap** | Life Science Analysts, Pharmacovigilance Teams | Dashboard showing drugs with highest label-vs-reality disparity |

---

## 2. Current MVP State Assessment

### What Has Been Built (Week 4 Deliverable)

| Component | Status | Details |
|-----------|--------|---------|
| Streamlit Web App | Done | Deployed at `trupharm.streamlit.app` |
| Hybrid Retrieval (FAISS + BM25) | Done | Dense + sparse with reciprocal rank fusion |
| OpenFDA Drug Labels API Integration | Done | Real-time fetch from `/drug/label/` |
| Google Gemini 2.0 Flash LLM | Done | Optional; extractive fallback when no API key |
| Text Chunking & Indexing | Done | 250-word chunks, 40-word overlap |
| Confidence Scoring & Citations | Done | Heuristic 0–1 score, inline `[doc_id::field]` refs |
| CSV Logging | Done | 20+ interaction records in `product_metrics.csv` |
| Stress Test Page | Done | Edge-case scenario validation |
| Error Handling & Graceful Refusal | Done | Returns "Not enough evidence" on empty results |

### What Is Missing (vs. Full Proposal)

| Proposal Component | Status | Impact |
|--------------------|--------|--------|
| FAERS Adverse Event Data | **Not started** | Cannot show real-world patient evidence |
| NDC Directory Data | **Not started** | No structured product metadata |
| RxNorm Entity Resolution | **Not started** | No brand→generic mapping, no fuzzy matching |
| DrugBank Enrichment | **Deprioritized** | Requires license; alternatives available |
| Full Label Field Coverage | **Partial** | Only 10 of 70+ fields extracted |
| Agentic Orchestration (A/B/C) | **Not started** | No multi-agent reasoning chain |
| Signal Heatmap Dashboard | **Not started** | No analyst-facing disparity visualization |
| Dual-Source Validation | **Not started** | Cannot compare label vs. FAERS |
| A/B Testing Evaluation | **Not started** | No vanilla-LLM vs. RAG comparison |
| Grounding/Hallucination Audit | **Not started** | No citation precision measurement |
| BLUE Benchmark Evaluation | **Not started** | No semantic mapping accuracy test |

### Current Architecture (Week 4)

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit UI (Frontend)                │
│   Query Input  ·  Response  ·  Evidence  ·  Metrics/Logs │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                  RAG Engine (rag_engine.py)               │
│                                                          │
│  1. Build openFDA search query from user text            │
│  2. Fetch drug label records via openFDA API             │
│  3. Chunk text fields (10 selected label sections)  ◄── LIMITATION
│  4. Index: FAISS (dense) + BM25 (sparse)                │
│  5. Hybrid retrieval with reciprocal rank fusion         │
│  6. Generate answer (Gemini LLM or extractive fallback)  │
│  7. Log interaction to CSV                               │
└────────┬──────────────────────────┬──────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐    ┌───────────────────────────┐
│  openFDA API    │    │  Google Gemini 2.0 Flash   │
│  /drug/label    │    │  (Optional LLM grounding)  │
│  ONLY  ◄────────│────│── LIMITATION               │
└─────────────────┘    └───────────────────────────┘
```

**Key Limitation:** The current system can only answer "What does the drug label say?" but NOT the core proposal question: "Is what patients experience different from what the label says?"

---

## 3. Data Sources — Complete API Analysis

### 3.1 OpenFDA Drug Labels (`/drug/label/`)

| Attribute | Details |
|-----------|---------|
| **API Endpoint** | `https://api.fda.gov/drug/label.json` |
| **Authentication** | None required (API key optional for higher rate limits) |
| **Cost** | Free |
| **Rate Limits** | 240 req/min (no key), 120K req/day (with free key) |
| **Update Frequency** | Weekly |
| **Data Size** | All FDA-approved drug labels in SPL format |
| **Fields Available** | 70+ text fields covering all label sections |
| **Status in Project** | Already integrated (but only 10 of 70+ fields used) |

**What it provides:** The official FDA-approved drug labeling — indications, warnings, adverse reactions, contraindications, clinical pharmacology, dosing, and all other label sections.

**Role in TruPharma:** The "Official Ground Truth" — what the FDA says about a drug.

### 3.2 OpenFDA FAERS Adverse Events (`/drug/event/`)

| Attribute | Details |
|-----------|---------|
| **API Endpoint** | `https://api.fda.gov/drug/event.json` |
| **Authentication** | None required |
| **Cost** | Free |
| **Rate Limits** | Same as labels (240/min, 120K/day with key) |
| **Update Frequency** | Quarterly |
| **Data Size** | 15M+ adverse event reports (2004–present) |
| **Pagination** | `skip` + `limit` (max 26,000 per query) |
| **Count Endpoint** | Yes — pre-aggregated counts available server-side |
| **Status in Project** | **Not started — CRITICAL PRIORITY** |

**What it provides:** Real-world patient adverse event reports — what patients, doctors, and pharmacists actually observed after taking drugs.

**Key fields:**
- `patient.reaction[].reactionmeddrapt` — adverse reaction (MedDRA standardized term)
- `patient.reaction[].reactionoutcome` — outcome (recovered, fatal, unknown, etc.)
- `patient.drug[].medicinalproduct` — drug name (may be brand or generic)
- `patient.drug[].drugcharacterization` — suspect (1), concomitant (2), or interacting (3)
- `patient.drug[].activesubstance.activesubstancename` — active ingredient
- `patient.drug[].drugindication` — why they were taking the drug
- `patient.drug[].drugdosagetext` — dosage details
- `patient.drug[].drugadministrationroute` — route (oral, IV, topical, etc.)
- `patient.patientsex` — 0=Unknown, 1=Male, 2=Female
- `patient.patientonsetage` + `patientonsetageunit` — age at event onset
- `patient.patientweight` — weight in kg
- `serious` — 1=serious, 2=not serious
- `seriousnessdeath` — 1 if resulted in death
- `seriousnesshospitalization` — 1 if resulted in hospitalization
- `seriousnesslifethreatening` — 1 if life-threatening
- `seriousnessdisabling` — 1 if resulted in disability
- `receivedate` — date FDA received the report
- `primarysource.qualification` — who reported (1=Physician, 2=Pharmacist, 3=Other HCP, 5=Consumer)

**Role in TruPharma:** The "Real-World Evidence" — what patients actually experience.

**Important API Feature — Count Endpoint:**
Instead of fetching thousands of individual reports, the FAERS API supports pre-aggregated count queries:
```
/drug/event.json?search=patient.drug.openfda.generic_name:"ibuprofen"
    &count=patient.reaction.reactionmeddrapt.exact
```
This returns server-side aggregated reaction counts — dramatically faster than downloading and counting in Python.

### 3.3 OpenFDA NDC Directory (`/drug/ndc/`)

| Attribute | Details |
|-----------|---------|
| **API Endpoint** | `https://api.fda.gov/drug/ndc.json` |
| **Authentication** | None required |
| **Cost** | Free |
| **Rate Limits** | Same as other openFDA endpoints |
| **Update Frequency** | Daily |
| **Data Size** | 300K+ active and historical product listings |
| **Status in Project** | **Not started** |

**What it provides:** Structured product metadata — brand/generic names, active ingredients, dosage forms, routes, manufacturers, NDC codes, and pharmacological classifications.

**Key fields:**
- `brand_name` / `brand_name_base` / `brand_name_suffix` — trade names
- `generic_name` — generic drug name
- `active_ingredients[].name` / `.strength` — ingredients with strengths
- `dosage_form` — tablet, capsule, solution, etc.
- `route` — oral, topical, intravenous, etc.
- `marketing_category` — NDA, ANDA, BLA, OTC Monograph
- `application_number` — NDA/ANDA number
- `product_type` — HUMAN OTC DRUG, HUMAN PRESCRIPTION DRUG
- `dea_schedule` — controlled substance schedule (CI–CV)
- `pharm_class` — pharmacological class
- `openfda.pharm_class_epc` — Established Pharmacologic Class (e.g., "Nonsteroidal Anti-inflammatory Drug [EPC]")
- `openfda.pharm_class_moa` — Mechanism of Action (e.g., "Cyclooxygenase Inhibitors [MoA]")
- `openfda.pharm_class_cs` — Chemical Structure class
- `openfda.rxcui` — RxNorm Concept Unique Identifier (links to RxNorm)

**Key insight:** The NDC API already includes `openfda.rxcui`, which bridges directly to RxNorm without a separate lookup.

**Role in TruPharma:** Product identity and structured metadata — the "ID card" for each drug.

### 3.4 RxNorm API (NLM RxNav)

| Attribute | Details |
|-----------|---------|
| **API Base** | `https://rxnav.nlm.nih.gov/REST/` |
| **Authentication** | None required, no license needed |
| **Cost** | Completely free |
| **Rate Limits** | 20 requests/second |
| **Data Size** | 100K+ concepts with millions of relationships |
| **Status in Project** | **Not started** |

**What it provides:** Drug name normalization — resolves brand names to generics, provides spelling suggestions, and maps between drug identifier systems.

**Key endpoints for TruPharma:**

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `/drugs?name={name}` | Resolve drug name to ingredients | `name=advil` → ibuprofen 200 MG [Advil] |
| `/rxcui?name={name}` | Get RxCUI for a drug name | `name=ibuprofen` → rxcui: 5640 |
| `/rxcui/{id}/allrelated` | Get all related concepts | Full relationship graph |
| `/rxcui/{id}/related?tty=IN` | Get ingredient from product | Brand product → ingredient |
| `/approximateTerm?term={term}` | Fuzzy matching | Handles misspellings |
| `/spellingsuggestions?name={name}` | Spell-check | Typo suggestions |
| `/rxcui/{id}/ndcs` | Get NDCs for a concept | Links to NDC Directory |

**Example — Brand to Generic Resolution:**
```
GET https://rxnav.nlm.nih.gov/REST/drugs.json?name=advil

Response:
{
  "drugGroup": {
    "conceptGroup": [{
      "tty": "SBD",
      "conceptProperties": [{
        "rxcui": "153008",
        "name": "ibuprofen 200 MG Oral Tablet [Advil]",
        "synonym": "Advil 200 MG Oral Tablet"
      }]
    }]
  }
}
```

**Role in TruPharma:** Entity resolution layer — ensures "Advil," "advil," "Avdil" (typo), and "ibuprofen" all resolve to the same drug before querying other APIs.

### 3.5 DrugBank — Deprioritized

| Attribute | Details |
|-----------|---------|
| **Access** | Requires Academic License (free but needs application + approval) |
| **Data** | 14K+ drug entries — pharmacology, drug-drug interactions, biochemical pathways |
| **Status** | **Deprioritized** |

**Why deprioritized:** Between OpenFDA Drug Labels (which include `drug_interactions`, `clinical_pharmacology`, `mechanism_of_action`) and the NDC API (which includes pharmacological class and mechanism of action), the core pharmacological context DrugBank would provide is already available from free, no-auth APIs.

**Alternative for drug interactions:** The RxNorm Interaction API (`https://rxnav.nlm.nih.gov/REST/interaction/`) provides drug-drug interaction data for free.

### 3.6 BLUE Benchmark — Evaluation Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | `https://github.com/ncbi-nlp/BLUE_Benchmark` |
| **Access** | Free GitHub download |
| **Purpose** | Evaluation only — used in Phase 4, not during data ingestion |
| **Tasks** | 10 datasets across 5 biomedical NLP tasks (NER, relation extraction, etc.) |

### Summary: All Data Sources

| Dataset | API Available? | Auth Required? | Cost | Priority | Role |
|---------|---------------|----------------|------|----------|------|
| OpenFDA Drug Labels | Yes | No | Free | Already done (needs expansion) | Official ground truth |
| OpenFDA FAERS | Yes | No | Free | **Critical — #1 priority** | Real-world evidence |
| OpenFDA NDC | Yes | No | Free | High | Product identity |
| RxNorm | Yes | No | Free | High | Entity resolution |
| DrugBank | Licensed | Academic application | Free for students | Low — skip | Enrichment |
| BLUE Benchmark | N/A | No | Free download | Phase 4 | Evaluation |

**All core data sources are free REST APIs with no authentication required.**

---

## 4. Cross-Dataset Linking Strategy

### The Universal Linking Fields

OpenFDA places a harmonized `openfda` object inside **all three of its endpoints** (labels, events, NDC). These shared fields enable cross-referencing:

```
                        ┌─────────────────────────────┐
                        │       RxNorm API             │
                        │  rxcui ← drug name lookup    │
                        │  spelling suggestions        │
                        │  brand → generic resolution  │
                        └──────────┬──────────────────┘
                                   │ rxcui
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  /drug/label      │  │  /drug/event      │  │  /drug/ndc        │
│  (Drug Labels)    │  │  (FAERS)          │  │  (NDC Directory)  │
├───────────────────┤  ├───────────────────┤  ├───────────────────┤
│ openfda.rxcui  ◄──┼──► openfda.rxcui  ◄──┼──► openfda.rxcui    │
│ openfda.brand_ ◄──┼──► openfda.brand_ ◄──┼──► brand_name       │
│   name            │  │   name            │  │ generic_name      │
│ openfda.generic◄──┼──► openfda.generic◄──┼──► openfda.spl_     │
│   _name           │  │   _name           │  │   set_id          │
│ openfda.product◄──┼──► openfda.product◄──┼──► product_ndc       │
│   _ndc            │  │   _ndc            │  │ openfda.unii      │
│ openfda.spl_   ◄──┼──► openfda.spl_   ◄──┼──► openfda.pharm_   │
│   set_id          │  │   set_id          │  │   class_*         │
│ openfda.unii      │  │ openfda.unii      │  │ application_      │
│ openfda.substance │  │ openfda.substance │  │   number          │
│   _name           │  │   _name           │  │                   │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

### Join Key Hierarchy (Strongest to Weakest)

| Rank | Join Key | Present In | Description |
|------|----------|------------|-------------|
| 1 | **`rxcui`** | Labels, FAERS, NDC, RxNorm | Universal drug concept identifier — strongest link |
| 2 | **`generic_name` / `substance_name`** | Labels, FAERS, NDC | Active ingredient name — text-based fallback |
| 3 | **`product_ndc`** | Labels, FAERS, NDC | Product-level NDC code |
| 4 | **`spl_set_id`** | Labels, FAERS, NDC | Label document identifier (stable across versions) |
| 5 | **`application_number`** | Labels, FAERS, NDC | NDA/ANDA regulatory application number |
| 6 | **`unii`** | Labels, FAERS, NDC | Unique Ingredient Identifier (molecular structure) |
| 7 | **`brand_name`** | Labels, FAERS, NDC | Trade name (less reliable due to variations) |

### Linking Workflow

```
Step 1: User enters "Advil headache"
                │
Step 2: RxNorm resolves → rxcui: 153008, generic: "ibuprofen", brands: ["Advil", "Motrin", ...]
                │
Step 3: Use resolved identifiers to query all three openFDA endpoints:
        ├── /drug/label?search=openfda.rxcui:"153008"      → Official label
        ├── /drug/event?search=patient.drug.openfda.rxcui:"153008"  → FAERS reports
        └── /drug/ndc?search=openfda.rxcui:"153008"         → Product metadata
                │
Step 4: All results share openfda.rxcui="153008" → unified drug profile
```

---

## 5. Architecture — Snowflake-Free Design

### Design Decision: Why Snowflake-Free?

The original proposal specified a Snowflake-native architecture (Cortex Search, Cortex Complete, Snowpark Container Services). We adopted a **cloud-agnostic, open-source architecture** that achieves identical functionality:

| Proposal (Snowflake) | Replacement | Rationale |
|----------------------|-------------|-----------|
| Cortex Search (vector index) | **FAISS** (CPU) | Already proven in MVP; zero cost; same inner-product similarity |
| Snowflake relational tables | **In-memory + JSON cache** | No persistent DB needed for real-time API architecture |
| Cortex Complete (LLM) | **Google Gemini 2.0 Flash** | Already integrated; free tier available; excellent performance |
| Snowpark Container Services | **Python scripts + GitHub Actions** | Simpler, no cloud provisioning needed |
| Streamlit in Snowflake | **Streamlit Community Cloud** | Already deployed; free; same Streamlit framework |

### Advantages of This Approach

- **Zero cloud cost** — no Snowflake credits, all APIs are free
- **Faster iteration** — no cloud provisioning, everything runs locally
- **Simpler deployment** — Streamlit Community Cloud + GitHub auto-deploy
- **Portable** — can run on any machine with Python, not locked to Snowflake
- **Same capabilities** — FAISS + BM25 + Gemini delivers identical RAG functionality

### Target Architecture (Full Implementation)

```
┌──────────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI (Frontend)                     │
│                                                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐  │
│  │     SAFETY CHAT         │    │     SIGNAL HEATMAP          │  │
│  │  Consumer-facing Q&A    │    │  Analyst disparity dashboard │  │
│  │  Dual-source validation │    │  90-day rolling window      │  │
│  └────────────┬────────────┘    └──────────────┬──────────────┘  │
└───────────────┼────────────────────────────────┼─────────────────┘
                │                                │
                ▼                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                 AGENTIC ORCHESTRATION LAYER                      │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Agent A      │  │  Agent B      │  │  Agent C              │  │
│  │  (Regulator)  │  │  (Observer)   │  │  (Verifier)           │  │
│  │  Official     │  │  Real-world   │  │  Reconciles A + B     │  │
│  │  label data   │  │  FAERS data   │  │  Identifies signals   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘  │
└─────────┼─────────────────┼─────────────────────┼────────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                      RAG ENGINE (Upgraded)                        │
│                                                                  │
│  Unified Drug Profile → Smart Chunking → FAISS + BM25 Index     │
│  Hybrid Retrieval (Dense + Sparse + Reciprocal Rank Fusion)      │
│  Answer Generation (Gemini LLM / Extractive Fallback)            │
│  Confidence Scoring + Citation Enforcement                       │
└──────────┬────────────────────────────────────────┬──────────────┘
           │                                        │
           ▼                                        ▼
┌──────────────────────────────────────────────────────────────────┐
│              UNIFIED DRUG PROFILE ASSEMBLER                      │
│                                                                  │
│  Orchestrates parallel API calls → Merges into single document   │
│  Computes disparity analysis (label vs. FAERS)                   │
└──┬────────────┬────────────┬────────────┬────────────────────────┘
   │            │            │            │
   ▼            ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────────┐
│ RxNorm │  │ Drug   │  │ FAERS  │  │ NDC        │
│ API    │  │ Label  │  │ API    │  │ API        │
│        │  │ API    │  │        │  │            │
│ Entity │  │ ALL    │  │ Adverse│  │ Product    │
│ Resoln │  │ fields │  │ Events │  │ Metadata   │
└────────┘  └────────┘  └────────┘  └────────────┘
   FREE        FREE        FREE        FREE
   No Auth     No Auth     No Auth     No Auth
```

---

## 6. Data Ingestion Pipeline

### Three-Layer Architecture

```
LAYER 1: RESOLUTION              LAYER 2: PARALLEL FETCH              LAYER 3: UNIFIED ASSEMBLY
─────────────────────            ─────────────────────                ──────────────────────────

User: "Advil headache"           ┌── /drug/label ─────────┐          ┌──────────────────────────┐
       │                         │  ALL 70+ fields         │          │   Unified Drug Profile   │
       ▼                         │  (warnings, adverse_    │          │                          │
┌──────────────┐                 │   reactions, contrain-  │─────────►│  IDENTITY SECTION        │
│  RxNorm API  │                 │   dications, clinical_  │          │  (from NDC + RxNorm)     │
│              │                 │   pharmacology...)      │          │                          │
│  /drugs?     │                 └─────────────────────────┘          │  OFFICIAL LABEL SECTION  │
│  name=advil  │──► rxcui: 153008                                    │  (from /drug/label)      │
│              │    generic:      ┌── /drug/event ─────────┐         │  ALL fields, not just 10 │
│  /approximate│    ibuprofen     │  Adverse event reports  │         │                          │
│  Term (fuzzy)│    brands:       │  Reactions + counts     │────────►│  REAL-WORLD EVIDENCE     │
└──────────────┘    [Advil,       │  Seriousness data       │         │  (from /drug/event)      │
                     Motrin...]    │  Patient demographics   │         │  Aggregated statistics   │
                                  │  Sample narratives      │         │  + raw narratives        │
                                  └─────────────────────────┘         │                          │
                                                                      │  PRODUCT METADATA        │
                                  ┌── /drug/ndc ───────────┐         │  (from /drug/ndc)        │
                                  │  Brand/generic mapping  │────────►│  Pharm class, dosage,    │
                                  │  Dosage form, route     │         │  manufacturer            │
                                  │  Pharm class (MoA, EPC) │         │                          │
                                  │  Manufacturer           │         │  DISPARITY ANALYSIS      │
                                  └─────────────────────────┘         │  (computed)              │
                                                                      │  Label vs FAERS delta    │
                                                                      └──────────────────────────┘
```

### FAERS Ingestion — Using the Count Endpoint

Instead of downloading thousands of individual reports and aggregating in Python, the FAERS API supports server-side aggregation via the `count` parameter:

```
# Get top adverse reactions for ibuprofen (aggregated by FDA servers)
GET /drug/event.json
  ?search=patient.drug.openfda.generic_name:"ibuprofen"
  &count=patient.reaction.reactionmeddrapt.exact

# Get seriousness breakdown
GET /drug/event.json
  ?search=patient.drug.openfda.generic_name:"ibuprofen"+AND+serious:1
  &count=seriousnessdeath

# Get reporter type distribution
GET /drug/event.json
  ?search=patient.drug.openfda.generic_name:"ibuprofen"
  &count=primarysource.qualification

# Get patient sex distribution
GET /drug/event.json
  ?search=patient.drug.openfda.generic_name:"ibuprofen"
  &count=patient.patientsex

# Get age group distribution
GET /drug/event.json
  ?search=patient.drug.openfda.generic_name:"ibuprofen"
  &count=patient.patientagegroup

# Get reports over time (for 90-day window analysis)
GET /drug/event.json
  ?search=patient.drug.openfda.generic_name:"ibuprofen"
  &count=receivedate
```

This is dramatically faster and more efficient than downloading individual reports.

### Caching Strategy

The top ~200 drugs cover approximately 80% of user queries. Pre-fetching their unified profiles and caching them drops query latency from 10+ seconds to under 2 seconds.

```
Cache Layer:
├── Hot cache (in-memory): Last 50 queried drugs
├── Warm cache (JSON files): Top 200 pre-fetched drugs
└── Cold (real-time API): Uncommon drugs fetched on demand
```

---

## 7. Drug Label Field Expansion

### Current State: 10 Fields (Allowlist)

```python
FIELD_ALLOWLIST = [
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
```

### Target State: ALL Clinically Relevant Fields (Blocklist Approach)

Instead of an allowlist (opt-in), use a **blocklist** (opt-out) — include everything except noise:

```python
FIELD_BLOCKLIST = {
    "spl_product_data_elements",
    "spl_indexing_data_elements",
    "effective_time",
    "set_id",
    "id",
    "version",
    "openfda",                                  # structured metadata, handled separately
    "package_label_principal_display_panel",     # image/layout data, not useful for RAG
}
# Everything else gets included automatically
```

### Fields Gained by Switching to Blocklist

| Field | Clinical Importance |
|-------|-------------------|
| **`adverse_reactions`** | Known side effects — the #1 most important missing field |
| **`boxed_warning`** | FDA's strongest safety warning ("black box") |
| **`contraindications`** | When the drug MUST NOT be used |
| **`indications_and_usage`** | What the drug is approved to treat |
| **`clinical_pharmacology`** | How the drug works in the body |
| **`mechanism_of_action`** | Molecular-level drug action |
| **`pharmacokinetics`** | Absorption, distribution, metabolism, excretion |
| **`pharmacodynamics`** | Drug effects on the body |
| **`pregnancy`** / **`pregnancy_or_breast_feeding`** | Pregnancy risk categories |
| **`pediatric_use`** | Child-specific safety data |
| **`geriatric_use`** | Elderly-specific safety data |
| **`nursing_mothers`** | Breastfeeding safety |
| **`precautions`** | General precautions |
| **`drug_abuse_and_dependence`** | Addiction potential |
| **`controlled_substance`** | DEA scheduling |
| **`clinical_studies`** | Clinical trial results |
| **`dosage_forms_and_strengths`** | Available formulations |
| **`how_supplied`** | Packaging and storage |
| **`use_in_specific_populations`** | Special populations (renal/hepatic impairment) |
| **`patient_medication_information`** | Patient-facing guidance |
| **`spl_medguide`** | FDA medication guide |
| **`spl_patient_package_insert`** | Patient package insert |
| **`recent_major_changes`** | What changed recently on the label |
| **`laboratory_tests`** | Required monitoring tests |
| **`nonclinical_toxicology`** | Animal study data |
| **`carcinogenesis_and_mutagenesis_and_impairment_of_fertility`** | Cancer/genetic risk |
| **`do_not_use`** | Absolute contraindications (OTC) |
| **`ask_doctor`** / **`ask_doctor_or_pharmacist`** | When to consult (OTC) |
| **`keep_out_of_reach_of_children`** | Pediatric safety |
| **`purpose`** | Drug purpose (OTC) |
| **`inactive_ingredient`** | Non-active components (allergy relevance) |
| **`storage_and_handling`** | Proper storage conditions |
| **`references`** | Literature references |

This expands coverage from **10 fields to 40+ fields**, giving the LLM/RAG significantly more information to answer questions accurately.

---

## 8. FAERS Adverse Event Ingestion Strategy

### Why Raw Reports Don't Work for RAG

Individual FAERS reports are JSON objects with nested drug/reaction arrays. Dumping raw JSON into a RAG pipeline produces poor results because:
1. Each report covers ONE patient — not statistically meaningful alone
2. The text is structured data, not narrative prose the LLM can reason over
3. 15 million reports can't be indexed per-query

### The Aggregation Strategy

Instead, **aggregate FAERS data into readable summaries** that the RAG can use effectively:

```
══════════════════════════════════════════════════════════
REAL-WORLD ADVERSE EVENT SUMMARY FOR IBUPROFEN
Source: FDA FAERS (Adverse Event Reporting System)
Based on analysis of reports from 2004 to present
══════════════════════════════════════════════════════════

TOTAL REPORTS: 45,231

TOP REPORTED ADVERSE REACTIONS:
 1. Nausea — 3,201 reports (7.1%)
 2. Drug ineffective — 2,845 reports (6.3%)
 3. Headache — 2,102 reports (4.6%)
 4. Diarrhoea — 1,890 reports (4.2%)
 5. Vomiting — 1,654 reports (3.7%)
 6. Abdominal pain upper — 1,432 reports (3.2%)
 7. Fatigue — 1,298 reports (2.9%)
 8. Dizziness — 1,187 reports (2.6%)
 9. Rash — 1,056 reports (2.3%)
10. Dyspnoea — 987 reports (2.2%)

SERIOUSNESS BREAKDOWN:
- Serious events: 28,450 (62.9%)
  · Deaths: 1,203 (2.7%)
  · Hospitalizations: 15,678 (34.7%)
  · Life-threatening: 3,421 (7.6%)
  · Disability: 2,890 (6.4%)
  · Congenital anomaly: 156 (0.3%)
- Non-serious events: 16,781 (37.1%)

REPORTER TYPES:
- Physicians: 34%
- Consumers/Non-health professionals: 42%
- Pharmacists: 8%
- Other health professionals: 16%

PATIENT DEMOGRAPHICS:
- Female: 58% | Male: 35% | Unknown: 7%
- Most common age group: Adult (65%), Elderly (22%)

RECENT TRENDS (Last 90 days):
- [Count of new reports in the rolling window]
- [Any reactions showing unusual spikes]

REPRESENTATIVE INDIVIDUAL NARRATIVES:
[5-10 sample case summaries for context]
══════════════════════════════════════════════════════════
```

This aggregated text is what gets chunked and indexed — it's far more useful for RAG than raw JSON reports.

---

## 9. Unified Drug Profile Model

### The Document That Gets Chunked for RAG

All four data sources are assembled into **one coherent document per drug** before chunking:

```
═══════════════════════════════════════════════════════════════
TRUPHA RMA UNIFIED DRUG PROFILE
═══════════════════════════════════════════════════════════════
Drug: IBUPROFEN
Brand Names: Advil, Motrin, Nuprin
RxCUI: 153008 | NDC: 0573-0154
═══════════════════════════════════════════════════════════════

SECTION 1: PRODUCT IDENTITY
[Source: NDC Directory + RxNorm]
─────────────────────────────────
  Manufacturer: Haleon US Holdings LLC
  Pharmacologic Class: Nonsteroidal Anti-inflammatory Drug [EPC]
  Mechanism of Action: Cyclooxygenase Inhibitors [MoA]
  Chemical Class: Anti-Inflammatory Agents, Non-Steroidal [CS]
  Available Forms: Tablet (200mg), Capsule (200mg), Suspension (20mg/mL)
  Route: Oral
  Marketing Category: NDA (NDA018989)
  Product Type: HUMAN OTC DRUG

SECTION 2: OFFICIAL FDA LABEL
[Source: OpenFDA /drug/label — ALL available fields]
─────────────────────────────────
  2a. Indications and Usage: [full text]
  2b. Dosage and Administration: [full text]
  2c. Boxed Warning: [full text, if present]
  2d. Contraindications: [full text]
  2e. Warnings and Precautions: [full text]
  2f. Adverse Reactions: [full text]
  2g. Drug Interactions: [full text]
  2h. Clinical Pharmacology: [full text]
  2i. Mechanism of Action: [full text]
  2j. Pharmacokinetics: [full text]
  2k. Pregnancy / Nursing: [full text]
  2l. Pediatric Use: [full text]
  2m. Geriatric Use: [full text]
  2n. Use in Specific Populations: [full text]
  ... [ALL additional available sections]

SECTION 3: REAL-WORLD EVIDENCE
[Source: OpenFDA FAERS /drug/event — Aggregated]
─────────────────────────────────
  3a. Report Summary: [total count, date range]
  3b. Top Adverse Reactions: [aggregated counts with percentages]
  3c. Seriousness Breakdown: [deaths, hospitalizations, etc.]
  3d. Reporter Demographics: [physician vs consumer reporting]
  3e. Patient Demographics: [sex, age distribution]
  3f. Recent Trends: [90-day rolling window]
  3g. Representative Narratives: [5-10 sample cases]

SECTION 4: DISPARITY ANALYSIS
[Computed by TruPharma]
─────────────────────────────────
  4a. Reactions in FAERS but NOT on official label:
      [List of adverse events reported by patients but not in label warnings]
  4b. Reactions on label with HIGH FAERS counts:
      [Known side effects that are being reported at high rates]
  4c. Reactions on label with LOW/NO FAERS reports:
      [Label warnings that rarely appear in real-world reports]
  4d. Disparity Score: [quantified measure of label vs reality gap]
═══════════════════════════════════════════════════════════════
```

### Chunking Strategy

Each section is chunked independently, tagged with its source:

| Section | Chunking Method | Typical Chunk Count |
|---------|----------------|-------------------|
| Product Identity | Single chunk (short) | 1 |
| Label sections | Per-field, then 250-word sub-chunks | 20-60 per drug |
| FAERS summary | Per-subsection (reactions, seriousness, demographics) | 5-10 |
| Disparity analysis | Single chunk per subsection | 3-4 |

Each chunk carries metadata: `{source: "label"|"faers"|"ndc"|"disparity", field: "...", drug: "...", rxcui: "..."}`

---

## 10. RAG Pipeline Architecture

### Upgraded Retrieval Flow

```
User Query: "Does ibuprofen cause stomach bleeding?"
                │
                ▼
┌─────────────────────────────────────┐
│  1. ENTITY RESOLUTION (RxNorm)      │
│     "ibuprofen" → rxcui: 5640      │
│     brands: [Advil, Motrin, ...]    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. PARALLEL DATA FETCH             │
│     ├── /drug/label (ALL fields)    │
│     ├── /drug/event (counts + samples)│
│     └── /drug/ndc (metadata)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. UNIFIED PROFILE ASSEMBLY        │
│     Merge all sources → one document│
│     Compute disparity analysis      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. SMART CHUNKING                  │
│     Section-aware chunks with       │
│     source tags and metadata        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. HYBRID INDEXING                 │
│     FAISS (inner-product) + BM25   │
│     TF-IDF vectors (fast, CPU)     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. HYBRID RETRIEVAL                │
│     Dense + Sparse retrieval        │
│     Reciprocal Rank Fusion          │
│     Top-K evidence selection        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  7. ANSWER GENERATION               │
│     Gemini LLM (grounded)          │
│     OR extractive fallback          │
│     Citations: [source::field]      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  8. RESPONSE                        │
│     Answer with dual-source evidence│
│     "The label says X [label::      │
│      adverse_reactions], but FAERS  │
│      reports show Y [faers::        │
│      top_reactions]"                │
└─────────────────────────────────────┘
```

### Key Improvement: Dual-Source Evidence

The current system can only say: *"According to the drug label, ibuprofen may cause stomach pain."*

The upgraded system says: *"The official label lists gastrointestinal bleeding as a warning [label::warnings]. Real-world FAERS data confirms this with 3,201 reports of GI hemorrhage out of 45,231 total reports (7.1%) [faers::top_reactions]. Of these, 89% were classified as serious events [faers::seriousness]."*

---

## 11. Agentic Orchestration Design

### Three-Agent Chain (from Proposal)

```
┌────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
│              "Is ibuprofen safe for elderly?"               │
└──────────────────────┬─────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
│  AGENT A     │ │  AGENT B     │ │                      │
│  Regulator   │ │  Observer    │ │                      │
│              │ │              │ │                      │
│  Queries     │ │  Queries     │ │                      │
│  official    │ │  FAERS data  │ │                      │
│  drug label  │ │  for real-   │ │                      │
│              │ │  world       │ │                      │
│  Extracts:   │ │  evidence    │ │                      │
│  - geriatric │ │              │ │                      │
│    use info  │ │  Extracts:   │ │                      │
│  - warnings  │ │  - elderly   │ │                      │
│  - adverse   │ │    age group │ │                      │
│    reactions │ │    reports   │ │                      │
│  - contrain- │ │  - serious   │ │                      │
│    dications │ │    events in │ │                      │
│              │ │    elderly   │ │                      │
└──────┬───────┘ └──────┬───────┘ │                      │
       │                │         │                      │
       └────────┬───────┘         │                      │
                ▼                 │  AGENT C             │
       ┌────────────────┐        │  Verifier            │
       │  Combined      │        │                      │
       │  Evidence      │───────►│  Reconciles A + B    │
       │  from A + B    │        │  Identifies:         │
       └────────────────┘        │  - Confirmed risks   │
                                 │    (on label + FAERS)│
                                 │  - Emerging signals  │
                                 │    (FAERS only)      │
                                 │  - Unconfirmed label │
                                 │    warnings (label   │
                                 │    only, low FAERS)  │
                                 │                      │
                                 │  Generates final     │
                                 │  verified assessment │
                                 └──────────┬───────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │  FINAL RESPONSE      │
                                 │  with dual-source    │
                                 │  citations and       │
                                 │  disparity analysis  │
                                 └──────────────────────┘
```

### Implementation Approach

Using **LangChain agents** or **custom chain-of-thought prompting** with Gemini:

- **Agent A (Regulator):** Retrieves from label chunks only; extracts official safety statements
- **Agent B (Observer):** Retrieves from FAERS chunks only; identifies real-world patterns
- **Agent C (Verifier):** Takes outputs from A and B; performs reconciliation; identifies where label and reality diverge

---

## 12. Frontend — Safety Chat & Signal Heatmap

### Safety Chat (Consumer Interface) — Upgrade Plan

**Current:** Single-source answers from drug labels only.

**Target:** Dual-source validated answers with disparity analysis.

```
┌─────────────────────────────────────────────────────────────┐
│  TruPharma Safety Chat                                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  🔍 "Does ibuprofen cause stomach bleeding?"        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─── VERIFIED ASSESSMENT ──────────────────────────────┐  │
│  │                                                       │  │
│  │  ✅ CONFIRMED RISK                                    │  │
│  │                                                       │  │
│  │  The official FDA label warns about gastrointestinal  │  │
│  │  bleeding [label::warnings]. FAERS real-world data    │  │
│  │  confirms this with 3,201 GI hemorrhage reports out   │  │
│  │  of 45,231 total (7.1%) [faers::reactions].           │  │
│  │                                                       │  │
│  │  Risk is elevated in elderly patients (22% of reports)│  │
│  │  and those on concurrent anticoagulants               │  │
│  │  [label::drug_interactions].                          │  │
│  │                                                       │  │
│  │  ┌── Official Label Says ─┐  ┌── Real World Shows ─┐ │  │
│  │  │  Listed as warning     │  │  3,201 reports       │  │
│  │  │  "Risk of GI bleeding" │  │  89% serious         │  │
│  │  │  Precaution advised    │  │  34% hospitalized    │  │
│  │  └────────────────────────┘  └──────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── EVIDENCE ─────────────────────────────────────────┐  │
│  │  📋 Label Evidence (3 chunks)                         │  │
│  │  📊 FAERS Evidence (4 chunks)                         │  │
│  │  🏷️  NDC Metadata (1 chunk)                           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Signal Heatmap (Analyst Interface) — New Page

```
┌─────────────────────────────────────────────────────────────┐
│  TruPharma Signal Heatmap — 90-Day Disparity Analysis       │
│                                                             │
│  ┌─── TOP DISPARITY DRUGS (Last 90 Days) ───────────────┐  │
│  │                                                       │  │
│  │  Drug          | Label  | FAERS   | Disparity | Trend │  │
│  │  ──────────────|────────|─────────|───────────|────── │  │
│  │  Drug A        | 5 warn | 12 rxns | ⬆ HIGH    | ↑↑↑  │  │
│  │  Drug B        | 8 warn | 15 rxns | ⬆ HIGH    | ↑↑   │  │
│  │  Drug C        | 3 warn | 6 rxns  | ➡ MEDIUM  | →    │  │
│  │  Drug D        | 12 warn| 14 rxns | ➡ LOW     | ↓    │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── REACTION HEATMAP ─────────────────────────────────┐  │
│  │  [Plotly/Altair heatmap visualization]                │  │
│  │  X-axis: Drugs                                        │  │
│  │  Y-axis: Adverse Reactions                            │  │
│  │  Color: Disparity score (green=aligned, red=gap)      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Evaluation Framework

### Metrics (from Proposal)

| Metric | Description | How Measured |
|--------|-------------|-------------|
| **Answer Groundedness / Citation Precision** | % of responses where cited adverse events match actual FDA label or FAERS data | Audit sample of 100+ queries |
| **Hallucination Rate** | Frequency of false-positive side effects not in retrieved context | Manual + automated review |
| **Semantic Mapping Accuracy** | Success rate mapping colloquial → formal medical terms | BLUE benchmark tasks |
| **Latency per Query** | End-to-end response time | Automated measurement |
| **Cost per Query** | Compute cost (API calls, LLM tokens) | Tracked per query |

### A/B Testing Design

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Baseline 1: Vanilla LLM** | Gemini or LLaMA without RAG | Measure baseline hallucination risk |
| **Baseline 2: Vector-Only RAG** | Standard vector search, no agentic orchestration | Demonstrate necessity of hybrid approach |
| **TruPharma Full** | Hybrid RAG + agentic orchestration + dual-source | The complete system |

The **"RAG-Lift"** is quantified as the improvement of the full system over both baselines.

---

## 14. Implementation Phases & Roadmap

### Phase 1: Data Infrastructure (Expand + Ingest)

| Step | Task | Effort | Dependencies |
|------|------|--------|-------------|
| 1.1 | Expand drug labels to ALL fields (blocklist approach) | Small | None |
| 1.2 | Build RxNorm resolution module (`rxnorm_resolver.py`) | Medium | None |
| 1.3 | Build FAERS ingestion module (`faers_ingestion.py`) | Medium-Heavy | Step 1.2 |
| 1.4 | Build NDC ingestion module (`ndc_ingestion.py`) | Small-Medium | Step 1.2 |
| 1.5 | Build unified profile assembler (`drug_profile.py`) | Medium | Steps 1.1–1.4 |

### Phase 2: RAG Pipeline Upgrade

| Step | Task | Effort | Dependencies |
|------|------|--------|-------------|
| 2.1 | Upgrade `rag_engine.py` to use unified profiles | Medium | Phase 1 |
| 2.2 | Implement source-aware chunking | Medium | Step 2.1 |
| 2.3 | Update system prompts for dual-source evidence | Small | Step 2.1 |
| 2.4 | Add disparity analysis computation | Medium | Steps 1.3, 1.5 |

### Phase 3: Agentic Orchestration

| Step | Task | Effort | Dependencies |
|------|------|--------|-------------|
| 3.1 | Implement Agent A (Regulator — label retrieval) | Medium | Phase 2 |
| 3.2 | Implement Agent B (Observer — FAERS retrieval) | Medium | Phase 2 |
| 3.3 | Implement Agent C (Verifier — reconciliation) | Medium-Heavy | Steps 3.1, 3.2 |

### Phase 4: Frontend Upgrade

| Step | Task | Effort | Dependencies |
|------|------|--------|-------------|
| 4.1 | Upgrade Safety Chat for dual-source display | Medium | Phase 2 |
| 4.2 | Build Signal Heatmap page | Medium | Steps 1.3, 2.4 |
| 4.3 | Add RxNorm-powered drug name resolution in UI | Small | Step 1.2 |

### Phase 5: Evaluation

| Step | Task | Effort | Dependencies |
|------|------|--------|-------------|
| 5.1 | Set up A/B testing (vanilla LLM vs. RAG) | Medium | Phase 3 |
| 5.2 | Run grounding/hallucination audit | Medium | Phase 3 |
| 5.3 | Run BLUE benchmark evaluation | Medium | Phase 2 |
| 5.4 | Measure latency and cost metrics | Small | Phase 3 |

### Phase 6: Hardening & Delivery

| Step | Task | Effort | Dependencies |
|------|------|--------|-------------|
| 6.1 | Integration testing (end-to-end) | Medium | Phase 4 |
| 6.2 | Robustness testing (edge cases, failures) | Medium | Phase 4 |
| 6.3 | Performance optimization (caching, latency) | Medium | Phase 4 |
| 6.4 | Final demo preparation | Small | Phase 6.1–6.3 |
| 6.5 | Technical report and documentation | Medium | All phases |

---

## 15. Technology Decisions & Rationale

### Tech Stack Summary

| Layer | Technology | Why |
|-------|-----------|-----|
| **Language** | Python 3.11 | Ecosystem maturity for ML/NLP |
| **Frontend** | Streamlit | Rapid prototyping, free cloud hosting |
| **Vector Search** | FAISS (CPU, inner-product) | Proven in MVP, zero cost, fast |
| **Sparse Search** | BM25 (rank-bm25) | Keyword retrieval complement to dense |
| **Embeddings** | TF-IDF (default) / SentenceTransformers (optional) | TF-IDF is fast on CPU; ST for higher quality |
| **LLM** | Google Gemini 2.0 Flash | Free tier, fast, good grounding |
| **Data Sources** | OpenFDA APIs + RxNorm API | All free, no auth, comprehensive |
| **Deployment** | Streamlit Community Cloud | Free, auto-deploy from GitHub |
| **Logging** | CSV (MVP) → Cloud logging (production) | Simple to start, upgradeable |

### Why Each Technology Was Chosen Over Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Vector DB | FAISS | ChromaDB, Pinecone, Weaviate | Already working; no external service; sufficient for project scale |
| LLM | Gemini 2.0 Flash | OpenAI GPT-4o, Llama-3, Claude | Free tier; already integrated; good medical text handling |
| Data Storage | In-memory + JSON cache | PostgreSQL, SQLite, DuckDB | Real-time API architecture doesn't need persistent DB; cache handles repeat queries |
| Agentic Framework | Custom / LangChain | LlamaIndex, AutoGen | Lightweight; avoids heavy dependencies; more control over agent behavior |
| Frontend | Streamlit | Flask, FastAPI + React | Already deployed; fastest path to working demo; free hosting |

---

## Appendix A: API Quick Reference

### OpenFDA Drug Labels
```
Base: https://api.fda.gov/drug/label.json
Search: ?search=openfda.generic_name:"ibuprofen"&limit=10
Fields: 70+ text fields (see Section 7)
```

### OpenFDA FAERS Adverse Events
```
Base: https://api.fda.gov/drug/event.json
Search: ?search=patient.drug.openfda.generic_name:"ibuprofen"&limit=10
Counts: ?search=...&count=patient.reaction.reactionmeddrapt.exact
```

### OpenFDA NDC Directory
```
Base: https://api.fda.gov/drug/ndc.json
Search: ?search=brand_name:"advil"&limit=10
```

### RxNorm
```
Base: https://rxnav.nlm.nih.gov/REST/
Resolve: /drugs.json?name=advil
Fuzzy: /approximateTerm.json?term=advl
Spell: /spellingsuggestions.json?name=advl
RxCUI: /rxcui.json?name=ibuprofen
```

---

## Appendix B: Key File Changes Required

| File | Change | Priority |
|------|--------|----------|
| `src/rag_engine.py` | Replace `FIELD_ALLOWLIST` with `FIELD_BLOCKLIST` | Step 1.1 |
| `src/rag_engine.py` | Integrate unified profile instead of raw labels | Step 2.1 |
| `src/rag_engine.py` | Update system prompt for dual-source evidence | Step 2.3 |
| `src/openfda_rag.py` | Generalize to support multiple API endpoints | Step 1.3 |
| **NEW** `src/rxnorm_resolver.py` | RxNorm API client for entity resolution | Step 1.2 |
| **NEW** `src/faers_ingestion.py` | FAERS data fetching and aggregation | Step 1.3 |
| **NEW** `src/ndc_ingestion.py` | NDC metadata fetching and formatting | Step 1.4 |
| **NEW** `src/drug_profile.py` | Unified profile assembler + disparity analysis | Step 1.5 |
| `src/app/streamlit_app.py` | Dual-source UI, RxNorm resolution in search | Step 4.1 |
| **NEW** `src/app/pages/signal_heatmap.py` | Analyst disparity dashboard | Step 4.2 |
| `src/app/pages/stress_test.py` | Expand for multi-source scenarios | Step 6.2 |

---

*Document Version: 1.0 · Created: February 2026 · TruPharma / CiteRx Team*
