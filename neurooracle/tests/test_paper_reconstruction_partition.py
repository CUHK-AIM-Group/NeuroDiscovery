import hashlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from partition_paper_reconstruction import classify, graph_records, HashingTextReader


class PartitionBoundaries(unittest.TestCase):
    def test_fixed_map_with_paper_citation_is_protected(self):
        self.assertEqual(classify('edge', 0, {'source': 'HansenReceptor2022',
            'metadata': {'pmid': '123'}, 'weight': 0.41})[0], 'fixed')

    def test_mixed_paper_entity_stays_protected(self):
        self.assertEqual(classify('node', 'x', {'source_vocab': 'claim_extraction',
            'domain_tags': ['claim', 'dataset']})[0], 'protected_unclassified')

    def test_unrecognized_bridge_is_not_removed(self):
        self.assertEqual(classify('edge', 0, {'source': 'curated-new-bridge'})[0],
                         'protected_unclassified')

    def test_explicit_legacy_claim_source_is_partitioned(self):
        self.assertEqual(classify('edge', 0, {'source': 'claim:123'})[0], 'legacy_literature')

    def test_single_pass_preserves_parallel_selfloop_unicode_and_headers(self):
        raw = ('{"meta":{"中文":"λ"},"concepts":{"a":{"id":"a"}},'
               '"edges":[{"source_id":"a","target_id":"a","weight":0.3},'
               '{"source_id":"a","target_id":"a","weight":0.4}]}  \n').encode()
        r = HashingTextReader(io.BytesIO(raw))
        rows = list(graph_records(r))
        self.assertEqual(len(rows), 4)
        self.assertEqual([x[1] for x in rows if x[0] == 'edge'], [0, 1])
        self.assertEqual(rows[-1][2]['weight'], 0.4)
        self.assertEqual(r.hash.hexdigest(), hashlib.sha256(raw).hexdigest())
        self.assertEqual(r.bytes, len(raw))

    def test_rejects_incomplete_graph_and_trailing_data(self):
        for value in [b'{"concepts":{}}', b'{"concepts":{},"edges":[]} junk']:
            with self.assertRaises(ValueError):
                list(graph_records(HashingTextReader(io.BytesIO(value))))


if __name__ == '__main__':
    unittest.main()
