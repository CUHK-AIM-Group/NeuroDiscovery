"""Phase 1: structured data ingestion.

Build the KG skeleton from NeuroNames, MeSH, DisGeNET, and CognitiveAtlas.
UMLS alignment keeps entities consistent across sources.

Usage:
    python -m neurooracle.phase1
"""

from .src.ingest_pipeline import run_full_ingestion
from .src.umls_integration import align_graph_to_umls

if __name__ == "__main__":
    from .src.ingest_pipeline import main
    main()
