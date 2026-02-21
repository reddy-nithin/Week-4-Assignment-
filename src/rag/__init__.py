"""
src.rag — RAG pipeline modules for TruPharma.

Public API:
    run_rag_query          (engine)
    read_logs              (engine)
    build_unified_profile  (drug_profile)
    compute_disparity      (drug_profile)
"""

from src.rag.engine import run_rag_query, read_logs
from src.rag.drug_profile import build_unified_profile, compute_disparity

__all__ = [
    "run_rag_query",
    "read_logs",
    "build_unified_profile",
    "compute_disparity",
]
