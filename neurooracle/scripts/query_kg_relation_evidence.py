"""Read current shared relation evidence with original sentences and conditions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neurooracle.src.relation_evidence_dossier import find_relation_dossiers
from neurooracle.src.claim_evidence_query import query_claim_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ids = parser.add_mutually_exclusive_group()
    ids.add_argument('--claim-id', help='Original CLM ID; return every paper in its current shared claim')
    ids.add_argument('--relation-id', help='Shared REL ID; return its complete paper evidence')
    parser.add_argument('--subject')
    parser.add_argument('--predicate')
    parser.add_argument('--object', dest='object_name')
    parser.add_argument('--minimum-papers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    campaign = Path(__file__).resolve().parents[1] / 'data/umls_mapping/umls_2026AA_atomic_mentions_v1_20260906/kg_overnight_20260907/CAMPAIGN.json'
    if args.claim_id or args.relation_id:
        if args.subject or args.predicate or args.object_name:
            parser.error('ID lookup cannot be combined with text filters')
        result = query_claim_evidence(campaign, claim_id=args.claim_id, relation_id=args.relation_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    results = find_relation_dossiers(campaign, subject=args.subject, predicate=args.predicate,
                                    object_name=args.object_name, minimum_papers=args.minimum_papers)
    results.sort(key=lambda x: (-x['evidence']['verified_article_count'], x['evidence']['relation_id']))
    print(json.dumps(dict(matching_relations=len(results), displayed=results[:args.limit],
        scope='current_shared_relations_only_not_complete_literature_recall',
        warning='Distinct articles are not independent studies. Review context and source roles; no consensus inferred.'),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
