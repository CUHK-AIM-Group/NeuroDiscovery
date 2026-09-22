import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

script=Path(__file__).resolve().parents[1]/'scripts/prepare_paper_reconstruction.py'
spec=importlib.util.spec_from_file_location('paper_reconstruction_scope',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PaperReconstructionScope(unittest.TestCase):
    def config(self):
        return {'execution':{'output_directory':'tmp/paper-scope-test'},'old_claims_are_extraction_input':False,
                'fixed_neuroscience_layer':{'allow_writes':False}}
    def test_rejects_fixed_layer_writes(self):
        config=self.config();config['fixed_neuroscience_layer']['allow_writes']=True
        with patch.object(module,'read',return_value=config),self.assertRaises(ValueError):module.settings()
    def test_rejects_old_claims_as_answers(self):
        config=self.config();config['old_claims_are_extraction_input']=True
        with patch.object(module,'read',return_value=config),self.assertRaises(ValueError):module.settings()
    def test_output_cannot_target_fixed_data(self):
        config=self.config();config['execution']['output_directory']='neurooracle/data/full_v2'
        with patch.object(module,'read',return_value=config),self.assertRaises(ValueError):module.settings()
    def test_packet_excludes_cached_claim_and_review_annotations(self):
        raw={'title':'Example','abstract_text':'A literal result; n = 17.','pmid':'123',
             'claims':[{'subject':'OLD'}],'seed_claim':{'id':'CLM:old'},'review_notes':'approved'}
        self.assertEqual(module.source_fields(raw),{'title':'Example','abstract_text':'A literal result; n = 17.','pmid':'123'})
        self.assertIn('claims',raw)
    def test_source_statistics_and_notice_text_unchanged(self):
        raw={'abstract_text':'β = −0.3; p < 0.05; 95% CI [−0.6, −0.1].','notices':[{'type':'correction','pmid':'456'}],
             'own_article_ids':{'pmid':'123','doi':'10.example/value'},'publication_types':['Review']}
        self.assertEqual(module.source_fields(raw),raw)
    def test_nested_bibliography_preserves_notices_but_excludes_answers(self):
        raw={'paper':{'pmid':'123','authors':['A'],'year':2000,'claims':['old answer']},
             'comments_corrections':[{'type':'ErratumFor','pmid':'456'}]}
        result=module.source_fields(raw)
        self.assertNotIn('claims',result['paper'])
        self.assertEqual(result['paper']['authors'],['A'])
        self.assertEqual(result['comments_corrections'],raw['comments_corrections'])

if __name__=='__main__':unittest.main(verbosity=2)
