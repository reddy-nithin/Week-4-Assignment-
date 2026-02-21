"""
rxnorm_resolver.py · TruPharma RxNorm Entity Resolution
========================================================
Resolves arbitrary drug name inputs (brands, generics, misspellings) to
canonical identifiers via the free RxNorm REST API.

API docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
No authentication required.  Rate limit ≈ 20 req/s.
"""

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

_BASE = "https://rxnav.nlm.nih.gov/REST"
_TIMEOUT = 10
_UA = "TruPharma/2.0"


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context using certifi certs (macOS Python needs this)."""
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
    """GET JSON from RxNorm API.  Returns {} on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, OSError):
        return {}


# ══════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════

def get_rxcui_by_name(name: str) -> Optional[str]:
    """Exact-match lookup: returns the RxCUI for a drug name, or None."""
    encoded = urllib.parse.quote(name, safe="")
    data = _api_get(f"{_BASE}/rxcui.json?name={encoded}&search=2")
    group = data.get("idGroup", {})
    ids = group.get("rxnormId")
    if ids and isinstance(ids, list) and ids[0] != "":
        return ids[0]
    return None


def get_drug_info(name: str) -> dict:
    """
    /drugs endpoint — returns brand-name / generic concept groups.
    Result: {"brands": [...], "generics": [...], "rxcuis": [...]}
    """
    encoded = urllib.parse.quote(name, safe="")
    data = _api_get(f"{_BASE}/drugs.json?name={encoded}")
    groups = data.get("drugGroup", {}).get("conceptGroup", [])

    brands: List[str] = []
    generics: List[str] = []
    rxcuis: List[str] = []

    for grp in groups:
        tty = grp.get("tty", "")
        for prop in grp.get("conceptProperties", []):
            n = prop.get("name", "")
            rxcui = prop.get("rxcui", "")
            if rxcui:
                rxcuis.append(rxcui)
            if tty in ("BN", "BPCK", "SBD", "SBDF", "SBDG"):
                brands.append(n)
            elif tty in ("IN", "MIN", "PIN", "SCD", "SCDF", "SCDG"):
                generics.append(n)

    return {"brands": brands, "generics": generics, "rxcuis": rxcuis}


def get_rxcui_properties(rxcui: str) -> dict:
    """Get properties (name, tty, etc.) for an RxCUI."""
    data = _api_get(f"{_BASE}/rxcui/{rxcui}/properties.json")
    return data.get("properties", {})


def get_approximate_match(term: str, max_entries: int = 10) -> list:
    """Fuzzy match via /approximateTerm. Returns list of {rxcui, name, score}."""
    encoded = urllib.parse.quote(term, safe="")
    data = _api_get(
        f"{_BASE}/approximateTerm.json?term={encoded}&maxEntries={max_entries}"
    )
    group = data.get("approximateGroup", {})
    candidates = group.get("candidate", [])
    results = []
    seen_cuis: set = set()
    for c in candidates:
        rxcui = c.get("rxcui", "")
        name = c.get("name", "").strip()
        if not rxcui or rxcui in seen_cuis:
            continue
        seen_cuis.add(rxcui)
        # If name is missing, resolve it from properties
        if not name:
            props = get_rxcui_properties(rxcui)
            name = props.get("name", "")
        if not name:
            continue
        results.append({
            "rxcui": rxcui,
            "name": name,
            "score": c.get("score", ""),
        })
    return results


def get_spelling_suggestions(name: str) -> list:
    """Returns a list of spelling-corrected drug names."""
    encoded = urllib.parse.quote(name, safe="")
    data = _api_get(f"{_BASE}/spellingsuggestions.json?name={encoded}")
    group = data.get("suggestionGroup", {})
    return group.get("suggestionList", {}).get("suggestion", []) or []


def get_related_brands(rxcui: str) -> list:
    """Get brand names related to an RxCUI (via /related?tty=BN)."""
    data = _api_get(f"{_BASE}/rxcui/{rxcui}/related.json?tty=BN")
    groups = data.get("relatedGroup", {}).get("conceptGroup", [])
    names = []
    for grp in groups:
        for prop in grp.get("conceptProperties", []):
            n = prop.get("name", "")
            if n and n not in names:
                names.append(n)
    return names


def get_generic_from_brand(rxcui: str) -> Optional[str]:
    """Resolve a brand-product RxCUI to its generic ingredient name."""
    data = _api_get(f"{_BASE}/rxcui/{rxcui}/related.json?tty=IN+MIN")
    groups = data.get("relatedGroup", {}).get("conceptGroup", [])
    for grp in groups:
        for prop in grp.get("conceptProperties", []):
            n = prop.get("name", "")
            if n:
                return n
    return None


def _get_all_related_rxcuis(rxcui: str) -> List[str]:
    """Collect all RxCUIs related to the given one (brands, generics, forms)."""
    data = _api_get(f"{_BASE}/rxcui/{rxcui}/allrelated.json")
    groups = data.get("allRelatedGroup", {}).get("conceptGroup", [])
    cuis: List[str] = []
    for grp in groups:
        for prop in grp.get("conceptProperties", []):
            c = prop.get("rxcui", "")
            if c and c not in cuis:
                cuis.append(c)
    return cuis


# ══════════════════════════════════════════════════════════════
#  Master resolution function
# ══════════════════════════════════════════════════════════════

def resolve_drug_name(name: str) -> dict:
    """
    Takes any drug name input (brand, generic, misspelled) and returns a
    canonical resolution dict:

        {
            "input":               original input string,
            "resolved_name":       best canonical name,
            "rxcui":               primary RxCUI (str or None),
            "brand_names":         [str, ...],
            "generic_name":        str or None,
            "all_rxcuis":          [str, ...],
            "spelling_suggestion": str or None,
            "confidence":          "exact" | "approximate" | "spelling_corrected" | "none",
        }
    """
    result: Dict = {
        "input": name,
        "resolved_name": name,
        "rxcui": None,
        "brand_names": [],
        "generic_name": None,
        "all_rxcuis": [],
        "spelling_suggestion": None,
        "confidence": "none",
    }

    clean = name.strip()
    if not clean:
        return result

    # ── 1. Exact match ────────────────────────────────────────
    rxcui = get_rxcui_by_name(clean)
    if rxcui:
        result["rxcui"] = rxcui
        result["confidence"] = "exact"
        _enrich(result, rxcui, clean)
        return result

    # ── 2. /drugs endpoint (brand/generic concept lookup) ─────
    info = get_drug_info(clean)
    if info["rxcuis"]:
        rxcui = info["rxcuis"][0]
        result["rxcui"] = rxcui
        result["confidence"] = "exact"
        if info["generics"]:
            result["generic_name"] = info["generics"][0]
            result["resolved_name"] = info["generics"][0]
        if info["brands"]:
            result["brand_names"] = list(dict.fromkeys(info["brands"]))
        result["all_rxcuis"] = list(dict.fromkeys(info["rxcuis"]))
        _enrich(result, rxcui, clean)
        return result

    # ── 3. Approximate (fuzzy) match ──────────────────────────
    approx = get_approximate_match(clean)
    if approx:
        best = approx[0]
        rxcui = best["rxcui"]
        result["rxcui"] = rxcui
        result["resolved_name"] = best["name"]
        result["confidence"] = "approximate"
        _enrich(result, rxcui, clean)
        return result

    # ── 4. Spelling suggestions ───────────────────────────────
    suggestions = get_spelling_suggestions(clean)
    if suggestions:
        corrected = suggestions[0]
        result["spelling_suggestion"] = corrected
        result["confidence"] = "spelling_corrected"
        # Recurse with the corrected spelling (one level only)
        sub = resolve_drug_name(corrected)
        if sub["rxcui"]:
            result["rxcui"] = sub["rxcui"]
            result["resolved_name"] = sub["resolved_name"]
            result["generic_name"] = sub["generic_name"]
            result["brand_names"] = sub["brand_names"]
            result["all_rxcuis"] = sub["all_rxcuis"]
        return result

    return result


def _enrich(result: dict, rxcui: str, original_input: str) -> None:
    """Fill in brand names, generic name, and related RxCUIs from a known RxCUI."""
    if not result.get("generic_name"):
        generic = get_generic_from_brand(rxcui)
        if generic:
            result["generic_name"] = generic
            result["resolved_name"] = generic

    if not result.get("brand_names"):
        if result.get("generic_name"):
            parent_cui = get_rxcui_by_name(result["generic_name"])
            if parent_cui:
                result["brand_names"] = get_related_brands(parent_cui)
        if not result.get("brand_names"):
            result["brand_names"] = get_related_brands(rxcui)

    if not result.get("all_rxcuis"):
        result["all_rxcuis"] = _get_all_related_rxcuis(rxcui)
    if rxcui not in result["all_rxcuis"]:
        result["all_rxcuis"].insert(0, rxcui)

    if not result.get("resolved_name") or result["resolved_name"] == original_input:
        if result.get("generic_name"):
            result["resolved_name"] = result["generic_name"]


# ══════════════════════════════════════════════════════════════
#  Quick CLI smoke test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    term = sys.argv[1] if len(sys.argv) > 1 else "Advil"
    res = resolve_drug_name(term)
    print(json.dumps(res, indent=2))
