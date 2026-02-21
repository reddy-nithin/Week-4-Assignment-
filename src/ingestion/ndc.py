"""
ndc_ingestion.py · TruPharma NDC Product Metadata
==================================================
Fetches structured product metadata from the OpenFDA NDC Directory API
and formats it for RAG consumption.

API docs: https://open.fda.gov/apis/drug/ndc/
No authentication required.
"""

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

_BASE = "https://api.fda.gov/drug/ndc.json"
_TIMEOUT = 15
_UA = "TruPharma/2.0"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()


def _api_get(url: str, timeout: int = _TIMEOUT) -> dict:
    """GET JSON from OpenFDA NDC API. Returns {} on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, OSError):
        return {}


def _search_ndc(query: str, limit: int = 10) -> List[dict]:
    """Run a search against the NDC endpoint and return result records."""
    encoded = urllib.parse.quote(query, safe='":+')
    url = f"{_BASE}?search={encoded}&limit={limit}"
    data = _api_get(url)
    return data.get("results", [])


def _unique(items: list) -> list:
    """Deduplicate while preserving order."""
    seen: set = set()
    out = []
    for x in items:
        key = x.upper() if isinstance(x, str) else str(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


# ══════════════════════════════════════════════════════════════
#  Main fetch + merge
# ══════════════════════════════════════════════════════════════

def fetch_ndc_metadata(
    generic_name: str,
    brand_name: Optional[str] = None,
    rxcui: Optional[str] = None,
) -> dict:
    """
    Fetch NDC product metadata.  Tries rxcui first, then brand, then
    generic name.  Merges multiple NDC entries into a single dict.
    """
    empty: Dict = {
        "brand_names": [],
        "generic_name": generic_name,
        "manufacturer": None,
        "active_ingredients": [],
        "dosage_forms": [],
        "routes": [],
        "marketing_category": None,
        "application_number": None,
        "product_type": None,
        "pharm_class_epc": [],
        "pharm_class_moa": [],
        "pharm_class_cs": [],
        "dea_schedule": None,
        "product_ndcs": [],
        "rxcuis": [],
    }

    records: List[dict] = []

    # Search strategy: try most specific first
    if rxcui:
        records = _search_ndc(f'openfda.rxcui:"{rxcui}"')

    if not records and brand_name:
        records = _search_ndc(f'brand_name:"{brand_name}"')

    if not records and generic_name:
        records = _search_ndc(f'generic_name:"{generic_name}"')

    if not records:
        return empty

    return _merge_records(records, generic_name)


def _merge_records(records: List[dict], generic_name: str) -> dict:
    """Merge multiple NDC product records into a single metadata dict."""
    brand_names: List[str] = []
    manufacturers: List[str] = []
    ingredients: List[dict] = []
    dosage_forms: List[str] = []
    routes: List[str] = []
    marketing_cats: List[str] = []
    app_numbers: List[str] = []
    product_types: List[str] = []
    epc: List[str] = []
    moa: List[str] = []
    cs: List[str] = []
    dea: List[str] = []
    ndcs: List[str] = []
    rxcuis: List[str] = []
    resolved_generic = generic_name

    for rec in records:
        bn = rec.get("brand_name", "")
        if bn:
            brand_names.append(bn)

        gn = rec.get("generic_name", "")
        if gn and not resolved_generic:
            resolved_generic = gn

        lab = rec.get("labeler_name", "")
        if lab:
            manufacturers.append(lab)

        for ai in rec.get("active_ingredients", []):
            name = ai.get("name", "")
            strength = ai.get("strength", "")
            if name:
                ingredients.append({"name": name, "strength": strength})

        for prod in rec.get("packaging", []):
            ndc = prod.get("package_ndc", "")
            if ndc:
                ndcs.append(ndc)

        pndc = rec.get("product_ndc", "")
        if pndc:
            ndcs.append(pndc)

        df = rec.get("dosage_form", "")
        if df:
            dosage_forms.append(df)

        for r in rec.get("route", []):
            if r:
                routes.append(r)

        mc = rec.get("marketing_category", "")
        if mc:
            marketing_cats.append(mc)

        an = rec.get("application_number", "")
        if an:
            app_numbers.append(an)

        pt = rec.get("product_type", "")
        if pt:
            product_types.append(pt)

        ds = rec.get("dea_schedule", "")
        if ds:
            dea.append(ds)

        openfda = rec.get("openfda", {})
        for c in openfda.get("rxcui", []):
            rxcuis.append(c)
        for c in openfda.get("pharm_class_epc", []):
            epc.append(c)
        for c in openfda.get("pharm_class_moa", []):
            moa.append(c)
        for c in openfda.get("pharm_class_cs", []):
            cs.append(c)

    # Deduplicate ingredients by name (keep first strength seen)
    seen_ing: set = set()
    unique_ing = []
    for ing in ingredients:
        key = ing["name"].upper()
        if key not in seen_ing:
            seen_ing.add(key)
            unique_ing.append(ing)

    return {
        "brand_names": _unique(brand_names),
        "generic_name": resolved_generic,
        "manufacturer": _unique(manufacturers)[0] if manufacturers else None,
        "active_ingredients": unique_ing,
        "dosage_forms": _unique(dosage_forms),
        "routes": _unique(routes),
        "marketing_category": _unique(marketing_cats)[0] if marketing_cats else None,
        "application_number": _unique(app_numbers)[0] if app_numbers else None,
        "product_type": _unique(product_types)[0] if product_types else None,
        "pharm_class_epc": _unique(epc),
        "pharm_class_moa": _unique(moa),
        "pharm_class_cs": _unique(cs),
        "dea_schedule": _unique(dea)[0] if dea else None,
        "product_ndcs": _unique(ndcs),
        "rxcuis": _unique(rxcuis),
    }


# ══════════════════════════════════════════════════════════════
#  Text formatter for RAG chunking
# ══════════════════════════════════════════════════════════════

def format_ndc_as_text(metadata: dict) -> str:
    """Format NDC metadata into readable text for RAG consumption."""
    if not metadata or not metadata.get("generic_name"):
        return ""

    gn = metadata["generic_name"]
    brands = metadata.get("brand_names", [])
    brand_str = ", ".join(brands[:5]) if brands else "N/A"

    header = f"PRODUCT IDENTITY: {brand_str} ({gn})"
    lines = [header]

    mfr = metadata.get("manufacturer")
    if mfr:
        lines.append(f"Manufacturer: {mfr}")

    epc = metadata.get("pharm_class_epc", [])
    if epc:
        lines.append(f"Pharmacologic Class: {'; '.join(epc)}")

    moa = metadata.get("pharm_class_moa", [])
    if moa:
        lines.append(f"Mechanism of Action: {'; '.join(moa)}")

    cs = metadata.get("pharm_class_cs", [])
    if cs:
        lines.append(f"Chemical Structure: {'; '.join(cs)}")

    forms = metadata.get("dosage_forms", [])
    if forms:
        lines.append(f"Dosage Forms: {'; '.join(forms)}")

    routes = metadata.get("routes", [])
    if routes:
        lines.append(f"Route: {'; '.join(routes)}")

    ings = metadata.get("active_ingredients", [])
    if ings:
        ing_strs = []
        for ai in ings:
            s = ai["name"]
            if ai.get("strength"):
                s += f" {ai['strength']}"
            ing_strs.append(s)
        lines.append(f"Active Ingredients: {'; '.join(ing_strs)}")

    pt = metadata.get("product_type")
    if pt:
        lines.append(f"Product Type: {pt}")

    mc = metadata.get("marketing_category")
    an = metadata.get("application_number")
    if mc:
        mc_str = mc
        if an:
            mc_str += f" ({an})"
        lines.append(f"Marketing: {mc_str}")

    dea = metadata.get("dea_schedule")
    if dea:
        lines.append(f"DEA Schedule: {dea}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Quick CLI smoke test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    drug = sys.argv[1] if len(sys.argv) > 1 else "ibuprofen"
    print(f"Fetching NDC metadata for '{drug}'...")
    meta = fetch_ndc_metadata(drug)
    print(json.dumps(meta, indent=2, default=str)[:2000])
    print("\n" + "=" * 60 + "\n")
    print(format_ndc_as_text(meta))
