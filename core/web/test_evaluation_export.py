"""Isolated synthetic participant handoff tests; no experimental or expert data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.web.discovery_study import DiscoveryStudy, register_discovery_routes, StudyError
from core.web.evaluation_export import build_evaluation_export, ranking_export
from core.web.test_user_study import UserStudyService


class EvaluationExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        pack = Path(__file__).parent / 'study_materials/cs1_discovery_pilot_v10.json'
        self.discovery = DiscoveryStudy(pack, self.root / 'discovery', record_kind='test')
        self.ranking = UserStudyService(self.root / 'ranking')
        self.one = self.discovery.create({'code': 'TEST-EXPORT', 'experience': '3-5', 'assignment_id': 'P01'})
        self.bank = self.root / 'candidates.json'
        self.bank.write_text(json.dumps({'session_protocol': {'required_sessions': 6, 'active_seconds_per_session': 600}, 'hypotheses': [
            {'id': 'h1', 'title': 'Hypothesis one', 'source_name': 'A', 'target_name': 'R1', 'composite_score': 0.9, 'path': []},
            {'id': 'h2', 'title': 'Hypothesis two', 'source_name': 'B', 'target_name': 'R2', 'composite_score': 0.2, 'path': []},
        ]}), encoding='utf-8')
        self.two = self.new_ranking()
        self.payload = {'participant_code': 'TEST-EXPORT', 'discovery_sessions': [
            {'id': self.one['session']['id'], 'token': self.one['session_token']}],
            'ranking_session_ids': [self.two['session_id']]}

    def new_ranking(self, code='TEST-EXPORT'):
        return self.ranking.create_session(study_id='test-study', participant_id=code,
                                          condition='manual', candidate_path=self.bank, random_seed=0)

    def answer(self, session, choice='left', event_id='first'):
        pair = session['pairs'][0]
        event = {'type': 'pairwise_choice', 'elapsed_ms': 1500, 'payload': {
            '_event_id': event_id, 'pair_id': pair['pair_id'], 'left_id': pair['left_id'],
            'right_id': pair['right_id'], 'choice': choice,
            'winner_id': pair['left_id'] if choice == 'left' else pair['right_id'],
            'decision_ms': 1500, 'changed_answer': event_id != 'first',
        }}
        self.ranking.append_events(session['session_id'], [event])
        return event

    def export(self, payload=None):
        return build_evaluation_export(payload or self.payload, self.discovery, self.ranking)

    def test_partial_bundle_keeps_raw_answers_and_explicit_incomplete_status(self):
        self.answer(self.two)
        result = self.export()
        self.assertEqual(result['completion_status'], 'partial')
        self.assertEqual(result['human_evaluation_1']['status'], 'in_progress')
        two = result['human_evaluation_2']['sessions'][0]
        self.assertEqual(two['summary']['answered'], 1)
        self.assertEqual(two['questions'][0]['answer'], 'left')
        self.assertIsNone(two['timing']['completed_at'])
        self.assertIsNone(two['final_ranking'])
        self.assertTrue(result['human_evaluation_1']['sessions'][0]['question_results'])

    def test_all_previous_sessions_for_same_expert_only(self):
        self.ranking.submit_session(self.two['session_id'], ranking=[item['id'] for item in self.two['candidates']], active_seconds=600, wall_seconds=610)
        second = self.new_ranking()
        stranger = self.new_ranking('TEST-OTHER')
        self.payload['ranking_session_ids'] = [second['session_id']]
        result = self.export()['human_evaluation_2']
        ids = {item['session']['session_id'] for item in result['sessions']}
        self.assertEqual(ids, {self.two['session_id'], second['session_id']})
        self.assertNotIn(stranger['session_id'], ids)
        self.assertEqual(result['completed_sessions'], 1)

    def test_mixed_participant_references_are_rejected(self):
        other = self.new_ranking('TEST-OTHER')
        payload = copy.deepcopy(self.payload)
        payload['ranking_session_ids'] = [other['session_id']]
        with self.assertRaises(ValueError): self.export(payload)
        payload = copy.deepcopy(self.payload)
        payload['participant_code'] = 'TEST-OTHER'
        with self.assertRaises(ValueError): self.export(payload)

    def test_private_resume_token_still_required(self):
        payload = copy.deepcopy(self.payload)
        payload['discovery_sessions'][0]['token'] = 'invalid'
        with self.assertRaises(StudyError): self.export(payload)

    def test_no_tokens_hidden_scores_or_result_labels(self):
        serialized = json.dumps(self.export())
        for forbidden in ('session_token', 'secret_hash', 'composite_score', 'execution_results', 'candidate_source', 'organizer'):
            self.assertNotIn('"' + forbidden + '"', serialized)
        self.assertNotIn(self.one['session_token'], serialized)

    def test_retry_deduplication_and_latest_answer(self):
        event = self.answer(self.two)
        self.ranking.append_events(self.two['session_id'], [event])
        self.answer(self.two, 'right', 'second')
        question = self.export()['human_evaluation_2']['sessions'][0]['questions'][0]
        self.assertEqual(question['answer'], 'right')
        self.assertEqual(question['answer_submission_count'], 2)
        self.assertEqual(question['modification_count'], 1)
        self.assertEqual(question['total_decision_time_ms'], 3000)

    def test_checkpoint_does_not_override_real_choices(self):
        self.answer(self.two)
        pair_id = self.two['pairs'][0]['pair_id']
        self.ranking.append_events(self.two['session_id'], [{'type': 'evaluation_checkpoint', 'payload': {
            'active_ms': 8000, 'open_ms': 9000, 'language': 'en', 'question_stats': {
                pair_id: {'choice': 'right', 'first_presented_at': '2026-09-20T10:00:00Z', 'view_count': 3}}}}])
        result = self.export()['human_evaluation_2']['sessions'][0]
        self.assertEqual(result['questions'][0]['answer'], 'left')
        self.assertEqual(result['questions'][0]['view_count'], 3)
        self.assertEqual(result['timing']['active_answering_time_ms'], 8000)

    def test_http_route_and_workflow_revision(self):
        app = FastAPI()
        register_discovery_routes(app, self.discovery, self.ranking)
        with TestClient(app) as client:
            response = client.post('/api/studies/evaluations/export', json=self.payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            self.assertEqual(response.json()['schema_version'], 'neurodiscovery-human-evaluations-v1')
            self.assertEqual(client.get('/api/studies/discovery/config').json()['evaluation_workflow_revision'], 'unified-evaluation-export-v1')

    def test_export_after_reopening_without_current_ranking_session(self):
        self.new_ranking('TEST-OTHER')
        result = self.export({'participant_code': 'TEST-EXPORT', 'ranking_study_ids': ['test-study']})
        self.assertEqual(result['human_evaluation_2']['session_count'], 1)
        self.assertEqual(result['human_evaluation_1']['status'], 'not_included')

    def test_read_only_export_and_duplicate_references(self):
        before = self.discovery.get(self.one['session']['id'], self.one['session_token'])
        payload = copy.deepcopy(self.payload)
        payload['discovery_sessions'] *= 2
        payload['ranking_session_ids'] *= 2
        first, second = self.export(payload), self.export(payload)
        self.assertEqual(first['human_evaluation_1']['session_count'], 1)
        self.assertEqual(first['human_evaluation_2']['session_count'], 1)
        self.assertEqual(first['human_evaluation_1']['sessions'][0]['question_results'], second['human_evaluation_1']['sessions'][0]['question_results'])
        self.assertEqual(first['human_evaluation_1']['sessions'][0]['session'], second['human_evaluation_1']['sessions'][0]['session'])
        self.assertEqual(before, self.discovery.get(self.one['session']['id'], self.one['session_token']))


if __name__ == '__main__':
    unittest.main()
