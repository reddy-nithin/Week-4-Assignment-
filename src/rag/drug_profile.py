"""
drug_profile.py · TruPharma Unified Drug Profile Assembler
==========================================================
Central orchestrator that resolves a drug via RxNorm, then fetches
Labels + FAERS + NDC in parallel, assembles a unified profile, and
computes label-vs-FAERS disparity analysis.

The output `text_sections` list produces TextChunk-compatible dicts
ready for the existing FAISS + BM25 indexing pipeline.
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure project root is importable ─────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.openfda_client import (
    clean_text,
    fetch_openfda_records,
    normalize_field_value,
    pick_text_fields,
)
from src.ingestion.rxnorm import resolve_drug_name
from src.ingestion.faers import fetch_faers_summary, format_faers_as_text
from src.ingestion.ndc import fetch_ndc_metadata, format_ndc_as_text

_FIELD_BLOCKLIST = {
    "spl_product_data_elements",
    "spl_indexing_data_elements",
    "effective_time",
    "set_id",
    "id",
    "version",
    "openfda",
    "package_label_principal_display_panel",
}

_LABEL_API = "https://api.fda.gov/drug/label.json"


# ══════════════════════════════════════════════════════════════
#  Drug-name extraction from natural language queries
# ══════════════════════════════════════════════════════════════

_STOP_WORDS = frozenset(
    "what are the is of for a an in on to and or how does do can "
    "side effects warnings interactions dosage dose drug about with "
    "tell me information info safety adverse reactions risk risks "
    "taking taken take should i my does it its this that".split()
)


def _extract_drug_name(query: str) -> str:
    """
    Heuristic extraction of a drug name from a natural-language query.
    Returns the longest non-stopword token sequence as a candidate drug name.
    """
    tokens = re.findall(r"[a-zA-Z0-9\-]+", query)
    candidates = [t for t in tokens if t.lower() not in _STOP_WORDS and len(t) > 2]
    if not candidates:
        return query.strip()
    return candidates[0]


# ══════════════════════════════════════════════════════════════
#  Label fetching (reuses openfda_client infrastructure)
# ══════════════════════════════════════════════════════════════

def _fetch_label_sections(
    generic_name: str,
    brand_name: Optional[str] = None,
    max_records: int = 3,
) -> Dict[str, str]:
    """
    Fetch drug label records from OpenFDA and return merged field->text dict.
    Tries generic name first, falls back to brand name.
    """
    queries = []
    if generic_name:
        queries.append(f'openfda.generic_name:"{generic_name}"')
    if brand_name:
        queries.append(f'openfda.brand_name:"{brand_name}"')
    if not queries:
        queries.append(generic_name)

    records: list = []
    for q in queries:
        try:
            records = fetch_openfda_records(
                search=q, base_url=_LABEL_API, limit=max_records, timeout_s=15,
            )
        except RuntimeError:
            continue
        if records:
            break

    if not records:
        try:
            records = fetch_openfda_records(
                search=generic_name, base_url=_LABEL_API, limit=max_records, timeout_s=15,
            )
        except RuntimeError:
            pass

    merged: Dict[str, str] = {}
    for rec in records:
        fields = pick_text_fields(rec, None, _FIELD_BLOCKLIST, False)
        for field, text in fields.items():
            if field not in merged or len(text) > len(merged[field]):
                merged[field] = text
    return merged


# ══════════════════════════════════════════════════════════════
#  Disparity analysis
# ══════════════════════════════════════════════════════════════

def _normalize_term(term: str) -> str:
    return re.sub(r"[^a-z ]", "", term.lower()).strip()


def compute_disparity(
    label_adverse_reactions: str,
    faers_top_reactions: List[dict],
) -> dict:
    """
    Compare label adverse_reactions text against FAERS top reported reactions.

    Returns dict with emerging signals, confirmed risks, over-warned items,
    and a 0-1 disparity score.
    """
    if not label_adverse_reactions or not faers_top_reactions:
        return {
            "reactions_in_faers_not_on_label": [],
            "reactions_on_label_with_high_faers": [],
            "reactions_on_label_with_low_faers": [],
            "disparity_score": 0.0,
        }

    label_lower = _normalize_term(label_adverse_reactions)

    in_faers_not_label = []
    on_label_high_faers = []
    total_faers = sum(r.get("count", 0) for r in faers_top_reactions) or 1

    for r in faers_top_reactions:
        term = r.get("term", "")
        count = r.get("count", 0)
        pct = count / total_faers * 100
        norm = _normalize_term(term)
        words = norm.split()

        found = any(w in label_lower for w in words if len(w) > 3)

        if found:
            on_label_high_faers.append({
                "term": term, "faers_count": count, "faers_pct": round(pct, 1)
            })
        else:
            in_faers_not_label.append({
                "term": term, "faers_count": count, "faers_pct": round(pct, 1)
            })

    total_top = len(faers_top_reactions) or 1
    disparity_score = round(len(in_faers_not_label) / total_top, 2)

    on_label_low_faers = []
    if on_label_high_faers:
        median_count = sorted(r["faers_count"] for r in on_label_high_faers)[
            len(on_label_high_faers) // 2
        ]
        on_label_low_faers = [
            r for r in on_label_high_faers if r["faers_count"] < median_count * 0.3
        ]

    return {
        "reactions_in_faers_not_on_label": in_faers_not_label,
        "reactions_on_label_with_high_faers": on_label_high_faers,
        "reactions_on_label_with_low_faers": on_label_low_faers,
        "disparity_score": disparity_score,
    }


def _format_disparity_text(disparity: dict) -> str:
    """Convert disparity analysis into readable text for RAG."""
    if not disparity or disparity.get("disparity_score", 0) == 0:
        return ""

    lines = [
        "LABEL vs REAL-WORLD DISPARITY ANALYSIS",
        f"Disparity Score: {disparity['disparity_score']:.2f} "
        f"(0 = fully aligned, 1 = fully divergent)",
        "",
    ]

    emerging = disparity.get("reactions_in_faers_not_on_label", [])
    if emerging:
        lines.append("EMERGING SIGNALS (reported in FAERS but NOT on drug label):")
        for r in emerging[:10]:
            lines.append(
                f"  - {r['term']}: {r['faers_count']:,} reports ({r['faers_pct']}%)"
            )
        lines.append("")

    confirmed = disparity.get("reactions_on_label_with_high_faers", [])
    if confirmed:
        lines.append("CONFIRMED RISKS (on label AND frequently reported in FAERS):")
        for r in confirmed[:10]:
            lines.append(
                f"  - {r['term']}: {r['faers_count']:,} reports ({r['faers_pct']}%)"
            )
        lines.append("")

    over_warned = disparity.get("reactions_on_label_with_low_faers", [])
    if over_warned:
        lines.append("POTENTIALLY OVER-WARNED (on label but rarely reported in FAERS):")
        for r in over_warned[:5]:
            lines.append(
                f"  - {r['term']}: {r['faers_count']:,} reports ({r['faers_pct']}%)"
            )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Master unified profile builder
# ══════════════════════════════════════════════════════════════

def build_unified_profile(user_query: str) -> dict:
    """
    Master function. Takes a user query, resolves the drug, fetches all
    data sources in parallel, and returns a unified profile with
    text_sections ready for RAG chunking.
    """
    drug_name = _extract_drug_name(user_query)
    rxnorm = resolve_drug_name(drug_name)

    generic = rxnorm.get("generic_name") or drug_name
    brand = (rxnorm.get("brand_names") or [None])[0]
    rxcui = rxnorm.get("rxcui")
    all_rxcuis = rxnorm.get("all_rxcuis", [])

    results: Dict[str, Any] = {}

    def _do_labels():
        return _fetch_label_sections(generic, brand_name=brand)

    def _do_faers():
        return fetch_faers_summary(generic, rxcuis=all_rxcuis[:5])

    def _do_ndc():
        return fetch_ndc_metadata(generic, brand_name=brand, rxcui=rxcui)

    tasks = {
        "labels": _do_labels,
        "faers": _do_faers,
        "ndc": _do_ndc,
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = {} if key != "faers" else {
                    "drug_name": generic, "total_reports": 0,
                    "top_reactions": [], "seriousness": {},
                    "reporter_types": {}, "patient_sex": {},
                    "patient_age_groups": {}, "sample_narratives": [],
                }

    label_data = results.get("labels", {})
    faers_summary = results.get("faers", {})
    ndc_metadata = results.get("ndc", {})

    label_adr_text = label_data.get("adverse_reactions", "")
    faers_reactions = faers_summary.get("top_reactions", [])
    disparity = compute_disparity(label_adr_text, faers_reactions)

    text_sections: List[dict] = []

    ndc_text = format_ndc_as_text(ndc_metadata)
    if ndc_text:
        text_sections.append({
            "source": "ndc",
            "field": "product_identity",
            "text": ndc_text,
        })

    for field, text in label_data.items():
        if len(text) >= 40:
            text_sections.append({
                "source": "label",
                "field": field,
                "text": text,
            })

    faers_text = format_faers_as_text(faers_summary)
    if faers_text:
        _add_faers_sections(text_sections, faers_summary, faers_text)

    disparity_text = _format_disparity_text(disparity)
    if disparity_text:
        text_sections.append({
            "source": "disparity",
            "field": "analysis",
            "text": disparity_text,
        })

    return {
        "drug_identity": {
            "input": user_query,
            "resolved_name": generic,
            "rxcui": rxcui,
            "brand_names": rxnorm.get("brand_names", []),
            "confidence": rxnorm.get("confidence", "none"),
        },
        "label_data": label_data,
        "faers_summary": faers_summary,
        "ndc_metadata": ndc_metadata,
        "disparity_analysis": disparity,
        "text_sections": text_sections,
    }


def _add_faers_sections(
    sections: list, summary: dict, full_text: str
) -> None:
    """Split FAERS data into separate section dicts for finer-grained retrieval."""
    total = summary.get("total_reports", 0)
    drug = summary.get("drug_name", "").upper()

    reactions = summary.get("top_reactions", [])
    if reactions:
        lines = [
            f"FAERS TOP ADVERSE REACTIONS FOR {drug} (Total reports: {total:,})"
        ]
        for i, r in enumerate(reactions[:15], 1):
            pct = r["count"] / total * 100 if total else 0
            lines.append(f"  {i}. {r['term']} — {r['count']:,} reports ({pct:.1f}%)")
        sections.append({
            "source": "faers", "field": "top_reactions", "text": "\n".join(lines),
        })

    ser = summary.get("seriousness", {})
    if ser and total:
        serious_pct = ser.get("serious", 0) / total * 100
        death_pct = ser.get("death", 0) / total * 100
        hosp_pct = ser.get("hospitalization", 0) / total * 100
        lt_pct = ser.get("life_threatening", 0) / total * 100
        dis_pct = ser.get("disabling", 0) / total * 100
        text = (
            f"FAERS SERIOUSNESS BREAKDOWN FOR {drug} (Total reports: {total:,})\n"
            f"Serious: {serious_pct:.1f}% ({ser.get('serious', 0):,})\n"
            f"Deaths: {death_pct:.1f}% ({ser.get('death', 0):,})\n"
            f"Hospitalizations: {hosp_pct:.1f}% ({ser.get('hospitalization', 0):,})\n"
            f"Life-threatening: {lt_pct:.1f}% ({ser.get('life_threatening', 0):,})\n"
            f"Disabling: {dis_pct:.1f}% ({ser.get('disabling', 0):,})"
        )
        sections.append({"source": "faers", "field": "seriousness", "text": text})

    sex = summary.get("patient_sex", {})
    age = summary.get("patient_age_groups", {})
    reporters = summary.get("reporter_types", {})
    if sex or age or reporters:
        lines = [f"FAERS DEMOGRAPHICS FOR {drug}"]
        if sex:
            lines.append("Patient Sex: " + ", ".join(f"{k}: {v}%" for k, v in sex.items()))
        if age:
            lines.append("Age Groups: " + ", ".join(f"{k}: {v}%" for k, v in age.items()))
        if reporters:
            lines.append("Reporter Types: " + ", ".join(f"{k}: {v}%" for k, v in reporters.items()))
        sections.append({
            "source": "faers", "field": "demographics", "text": "\n".join(lines),
        })

    samples = summary.get("sample_narratives", [])
    if samples:
        lines = [f"FAERS REPRESENTATIVE CASES FOR {drug}"]
        for i, s in enumerate(samples, 1):
            rxns = ", ".join(s.get("reactions", [])[:5]) or "unspecified"
            drugs = ", ".join(s.get("drugs", [])[:3]) or "unspecified"
            lines.append(
                f"  Case {i}: {s.get('age_group', '?')} {s.get('sex', '?')} — "
                f"reported {rxns} while taking {drugs}. Outcome: {s.get('outcome', '?')}."
            )
        sections.append({
            "source": "faers", "field": "sample_cases", "text": "\n".join(lines),
        })


# ══════════════════════════════════════════════════════════════
#  CLI smoke test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the side effects of Advil?"
    print(f"Building unified profile for: {query!r}\n")
    profile = build_unified_profile(query)

    print(f"Drug Identity: {profile['drug_identity']}")
    print(f"Label fields: {list(profile['label_data'].keys())}")
    print(f"FAERS total reports: {profile['faers_summary'].get('total_reports', 0):,}")
    print(f"NDC brand names: {profile['ndc_metadata'].get('brand_names', [])}")
    print(f"Disparity score: {profile['disparity_analysis'].get('disparity_score', 0)}")
    print(f"Text sections: {len(profile['text_sections'])}")
    print()
    for sec in profile["text_sections"]:
        print(f"  [{sec['source']}::{sec['field']}] ({len(sec['text'])} chars)")
