"""
faers_ingestion.py · TruPharma FAERS Adverse Event Ingestion
=============================================================
Fetches and aggregates adverse event data from the FDA FAERS API
into text summaries suitable for RAG chunking.

Uses the `count` endpoint for server-side aggregation (fast) and
only fetches a small number of individual reports for narrative samples.

API docs: https://open.fda.gov/apis/drug/event/
No authentication required (API key optional for higher rate limits).
"""

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

_BASE = "https://api.fda.gov/drug/event.json"
_TIMEOUT = 15
_UA = "TruPharma/2.0"

# FAERS qualification codes → human-readable labels
_QUALIFICATION_MAP = {
    "1": "physician",
    "2": "pharmacist",
    "3": "other_health_professional",
    "4": "lawyer",
    "5": "consumer",
}

# FAERS patient sex codes
_SEX_MAP = {
    "0": "unknown",
    "1": "male",
    "2": "female",
}

# FAERS patient age group codes
_AGE_GROUP_MAP = {
    "1": "neonate",
    "2": "infant",
    "3": "child",
    "4": "adolescent",
    "5": "adult",
    "6": "elderly",
}


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()


# ══════════════════════════════════════════════════════════════
#  Low-level HTTP helper
# ══════════════════════════════════════════════════════════════

def _api_get(url: str, timeout: int = _TIMEOUT) -> dict:
    """GET JSON from OpenFDA FAERS API. Returns {} on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, OSError):
        return {}


def _build_search(generic_name: str, rxcuis: Optional[List[str]] = None) -> str:
    """Build the FAERS search query string, preferring generic_name."""
    name = generic_name.strip().lower()
    clauses = [f'patient.drug.openfda.generic_name:"{name}"']
    if rxcuis:
        cui_clause = "+".join(f'patient.drug.openfda.rxcui:"{c}"' for c in rxcuis[:5])
        clauses.append(cui_clause)
    return "+OR+".join(clauses)


# ══════════════════════════════════════════════════════════════
#  Individual data fetchers
# ══════════════════════════════════════════════════════════════

def _fetch_total_count(search: str) -> int:
    url = f"{_BASE}?search={search}&limit=1"
    data = _api_get(url)
    return data.get("meta", {}).get("results", {}).get("total", 0)


def _fetch_top_reactions(search: str, limit: int = 25) -> List[dict]:
    url = f"{_BASE}?search={search}&count=patient.reaction.reactionmeddrapt.exact&limit={limit}"
    data = _api_get(url)
    results = data.get("results", [])
    return [{"term": r.get("term", ""), "count": r.get("count", 0)} for r in results]


def _fetch_seriousness(search: str) -> dict:
    """Fetch seriousness breakdown via separate filtered count queries."""
    fields = {
        "serious": "serious:1",
        "death": "seriousnessdeath:1",
        "hospitalization": "seriousnesshospitalization:1",
        "life_threatening": "seriousnesslifethreatening:1",
        "disabling": "seriousnessdisabling:1",
    }
    out: Dict[str, int] = {}
    for key, filt in fields.items():
        url = f"{_BASE}?search={search}+AND+{filt}&limit=1"
        data = _api_get(url)
        out[key] = data.get("meta", {}).get("results", {}).get("total", 0)
    return out


def _fetch_reporter_types(search: str) -> dict:
    url = f"{_BASE}?search={search}&count=primarysource.qualification"
    data = _api_get(url)
    results = data.get("results", [])
    total = sum(r.get("count", 0) for r in results) or 1
    out: Dict[str, float] = {}
    for r in results:
        code = str(r.get("term", ""))
        label = _QUALIFICATION_MAP.get(code, f"code_{code}")
        out[label] = round(r.get("count", 0) / total * 100, 1)
    return out


def _fetch_patient_sex(search: str) -> dict:
    url = f"{_BASE}?search={search}&count=patient.patientsex"
    data = _api_get(url)
    results = data.get("results", [])
    total = sum(r.get("count", 0) for r in results) or 1
    out: Dict[str, float] = {}
    for r in results:
        code = str(r.get("term", ""))
        label = _SEX_MAP.get(code, f"code_{code}")
        out[label] = round(r.get("count", 0) / total * 100, 1)
    return out


def _fetch_age_groups(search: str) -> dict:
    url = f"{_BASE}?search={search}&count=patient.patientagegroup"
    data = _api_get(url)
    results = data.get("results", [])
    total = sum(r.get("count", 0) for r in results) or 1
    out: Dict[str, float] = {}
    for r in results:
        code = str(r.get("term", ""))
        label = _AGE_GROUP_MAP.get(code, f"code_{code}")
        out[label] = round(r.get("count", 0) / total * 100, 1)
    return out


def _fetch_sample_reports(search: str, limit: int = 5) -> List[dict]:
    """Fetch recent individual reports for narrative summaries."""
    url = f"{_BASE}?search={search}&limit={limit}&sort=receivedate:desc"
    data = _api_get(url)
    reports = data.get("results", [])
    summaries = []
    for rpt in reports:
        patient = rpt.get("patient", {})
        drugs = patient.get("drug", [])
        reactions = patient.get("reaction", [])

        age = patient.get("patientagegroup", "")
        age_label = _AGE_GROUP_MAP.get(str(age), "")
        sex_code = str(patient.get("patientsex", ""))
        sex_label = _SEX_MAP.get(sex_code, "unknown")

        drug_names = []
        for d in drugs[:5]:
            dn = d.get("medicinalproduct", "")
            if dn:
                drug_names.append(dn)

        reaction_terms = []
        for rx in reactions[:10]:
            rt = rx.get("reactionmeddrapt", "")
            if rt:
                reaction_terms.append(rt)

        outcome_raw = rpt.get("serious", "")
        outcome_codes = []
        if rpt.get("seriousnessdeath"):
            outcome_codes.append("death")
        elif rpt.get("seriousnesshospitalization"):
            outcome_codes.append("hospitalization")
        elif rpt.get("seriousnesslifethreatening"):
            outcome_codes.append("life-threatening")
        elif rpt.get("seriousnessdisabling"):
            outcome_codes.append("disabling")
        outcome = ", ".join(outcome_codes) if outcome_codes else "non-serious"

        summaries.append({
            "age_group": age_label,
            "sex": sex_label,
            "drugs": drug_names,
            "reactions": reaction_terms,
            "outcome": outcome,
            "receive_date": rpt.get("receivedate", ""),
        })
    return summaries


# ══════════════════════════════════════════════════════════════
#  Master aggregation function
# ══════════════════════════════════════════════════════════════

def fetch_faers_summary(
    generic_name: str,
    rxcuis: Optional[List[str]] = None,
) -> dict:
    """
    Fetch aggregated FAERS data for a drug. Returns a summary dict with
    total_reports, top_reactions, seriousness breakdown, demographics,
    and sample narratives.  Returns empty summary on failure.
    """
    search = _build_search(generic_name, rxcuis)

    empty: Dict = {
        "drug_name": generic_name,
        "total_reports": 0,
        "top_reactions": [],
        "seriousness": {},
        "reporter_types": {},
        "patient_sex": {},
        "patient_age_groups": {},
        "sample_narratives": [],
    }

    # Run count + aggregation queries in parallel
    results: Dict = {}
    tasks = {
        "total": lambda: _fetch_total_count(search),
        "reactions": lambda: _fetch_top_reactions(search),
        "seriousness": lambda: _fetch_seriousness(search),
        "reporters": lambda: _fetch_reporter_types(search),
        "sex": lambda: _fetch_patient_sex(search),
        "age": lambda: _fetch_age_groups(search),
        "samples": lambda: _fetch_sample_reports(search),
    }

    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = None

    total = results.get("total", 0) or 0
    if total == 0:
        return empty

    return {
        "drug_name": generic_name,
        "total_reports": total,
        "top_reactions": results.get("reactions") or [],
        "seriousness": results.get("seriousness") or {},
        "reporter_types": results.get("reporters") or {},
        "patient_sex": results.get("sex") or {},
        "patient_age_groups": results.get("age") or {},
        "sample_narratives": results.get("samples") or [],
    }


# ══════════════════════════════════════════════════════════════
#  Text formatter for RAG chunking
# ══════════════════════════════════════════════════════════════

def format_faers_as_text(summary: dict) -> str:
    """
    Convert aggregated FAERS data into a readable text block
    suitable for RAG chunking.
    """
    if not summary or summary.get("total_reports", 0) == 0:
        return ""

    name = summary["drug_name"].upper()
    total = summary["total_reports"]
    lines = [
        f"REAL-WORLD ADVERSE EVENT SUMMARY FOR {name}",
        f"Source: FDA FAERS (Adverse Event Reporting System) | Total Reports: {total:,}",
        "",
    ]

    # Top reactions
    reactions = summary.get("top_reactions", [])
    if reactions:
        lines.append("Top Reported Adverse Reactions:")
        for i, r in enumerate(reactions[:15], 1):
            term = r["term"]
            count = r["count"]
            pct = count / total * 100 if total else 0
            lines.append(f"  {i}. {term} — {count:,} reports ({pct:.1f}%)")
        lines.append("")

    # Seriousness
    ser = summary.get("seriousness", {})
    if ser:
        serious = ser.get("serious", 0)
        ser_pct = serious / total * 100 if total else 0
        death_pct = ser.get("death", 0) / total * 100 if total else 0
        hosp_pct = ser.get("hospitalization", 0) / total * 100 if total else 0
        lt_pct = ser.get("life_threatening", 0) / total * 100 if total else 0
        dis_pct = ser.get("disabling", 0) / total * 100 if total else 0
        lines.append(
            f"Seriousness: {ser_pct:.1f}% serious | "
            f"{death_pct:.1f}% deaths | {hosp_pct:.1f}% hospitalizations | "
            f"{lt_pct:.1f}% life-threatening | {dis_pct:.1f}% disabling"
        )
        lines.append("")

    # Demographics
    sex = summary.get("patient_sex", {})
    if sex:
        sex_str = ", ".join(f"{k}: {v}%" for k, v in sex.items())
        lines.append(f"Patient Sex Distribution: {sex_str}")

    age = summary.get("patient_age_groups", {})
    if age:
        age_str = ", ".join(f"{k}: {v}%" for k, v in age.items())
        lines.append(f"Patient Age Groups: {age_str}")

    reporters = summary.get("reporter_types", {})
    if reporters:
        rep_str = ", ".join(f"{k}: {v}%" for k, v in reporters.items())
        lines.append(f"Reporter Types: {rep_str}")

    lines.append("")

    # Sample narratives
    samples = summary.get("sample_narratives", [])
    if samples:
        lines.append("Representative Cases:")
        for i, s in enumerate(samples, 1):
            age_g = s.get("age_group", "unknown age")
            sex_l = s.get("sex", "unknown")
            rxns = ", ".join(s.get("reactions", [])[:5]) or "unspecified"
            drugs = ", ".join(s.get("drugs", [])[:3]) or "unspecified"
            outcome = s.get("outcome", "unknown")
            date = s.get("receive_date", "")
            date_str = f" ({date[:4]}-{date[4:6]}-{date[6:]})" if len(date) == 8 else ""
            lines.append(
                f"  Case {i}: {age_g} {sex_l} — "
                f"reported {rxns} while taking {drugs}. "
                f"Outcome: {outcome}.{date_str}"
            )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Quick CLI smoke test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    drug = sys.argv[1] if len(sys.argv) > 1 else "ibuprofen"
    print(f"Fetching FAERS summary for '{drug}'...")
    summary = fetch_faers_summary(drug)
    print(json.dumps(summary, indent=2, default=str)[:2000])
    print("\n" + "=" * 60 + "\n")
    print(format_faers_as_text(summary))
