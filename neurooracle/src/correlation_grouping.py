"""Opt-in symmetric correlation indexing; scientific record orientation stays put."""
from .kg_bulk_cleanup import symmetric_correlation_key
from .relation_evidence import relation_key

POLICY = dict(version='kg.correlation_orientation.v1',symmetric_predicates=['correlates_with'],
              original_claim_and_edge_orientation_preserved=True)


def enabled(metadata):
    declaration=metadata.get('relation_grouping')
    if declaration is None:return False
    if declaration!=POLICY:raise ValueError('unsupported relation grouping policy')
    return True


class IndexTerms:
    def __init__(self,base=None):self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    def relation_key(self,claim):
        key=self.base.relation_key(claim) if self.base is not None else relation_key(claim)
        return symmetric_correlation_key(key)
