"""
src.ingestion — Data source modules for TruPharma.

Public API:
    resolve_drug_name     (rxnorm)
    fetch_faers_summary   (faers)
    format_faers_as_text  (faers)
    fetch_ndc_metadata    (ndc)
    format_ndc_as_text    (ndc)
    TextChunk, SubChunk   (openfda_client)
    build_artifacts       (openfda_client)
"""

from src.ingestion.rxnorm import resolve_drug_name
from src.ingestion.faers import fetch_faers_summary, format_faers_as_text
from src.ingestion.ndc import fetch_ndc_metadata, format_ndc_as_text
from src.ingestion.openfda_client import (
    TextChunk,
    SubChunk,
    build_artifacts,
    build_openfda_query,
    fetch_openfda_records,
    tokenize,
)

__all__ = [
    "resolve_drug_name",
    "fetch_faers_summary",
    "format_faers_as_text",
    "fetch_ndc_metadata",
    "format_ndc_as_text",
    "TextChunk",
    "SubChunk",
    "build_artifacts",
    "build_openfda_query",
    "fetch_openfda_records",
    "tokenize",
]
