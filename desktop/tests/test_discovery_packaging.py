"""Participant-pack staging is isolated, exact, versioned and non-destructive."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('discovery_stager', ROOT / 'desktop/scripts/stage-discovery-study.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiscoveryPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Path(self.temp.name) / 'backend'
        self.source = ROOT / 'core/web/study_materials'
        self.raw = (self.source / module.SOURCE_NAME).read_bytes()
        self.original = json.loads(self.raw)

    def test_participant_fields_and_all_science_values_remain_identical(self):
        pack = module.participant_pack(self.raw)
        for key in module.PUBLIC_FIELDS:
            self.assertEqual(pack[key], self.original[key], key)
        self.assertNotIn('organizer', pack)
        self.assertNotEqual(pack['pack_id'], self.original['pack_id'])
        self.assertEqual(pack['distribution']['author_pack_sha256'], hashlib.sha256(self.raw).hexdigest())

    def test_projection_does_not_mutate_author(self):
        pack = module.participant_pack(self.raw)
        pack['cards'][0]['pre']['title'] = 'synthetic'
        self.assertEqual((self.source / module.SOURCE_NAME).read_bytes(), self.raw)

    def test_only_four_distribution_assets_are_staged_and_translation_is_exact(self):
        module.stage(ROOT, self.backend)
        target = self.backend / 'core/web/study_materials'
        self.assertEqual({p.name for p in target.iterdir()},
                         {module.TARGET_NAME, module.CATALOG_NAME, module.ASSIGNMENTS_NAME,
                          module.REFERENCE_NOTES_NAME, module.SIGNIFICANCE_NAME})
        old = json.loads((self.source / module.CATALOG_NAME).read_bytes())
        new = json.loads((target / module.CATALOG_NAME).read_bytes())
        self.assertEqual(old['strings'], new['strings'])
        self.assertEqual(new['source_sha256'], hashlib.sha256((target / module.TARGET_NAME).read_bytes()).hexdigest())

    def test_assignment_table_is_staged_byte_identical_and_bound_to_the_author_pack(self):
        module.stage(ROOT, self.backend)
        target = self.backend / 'core/web/study_materials' / module.ASSIGNMENTS_NAME
        self.assertEqual(target.read_bytes(), (self.source / module.ASSIGNMENTS_NAME).read_bytes())
        table = json.loads(target.read_bytes())
        self.assertEqual(table['pack_id'], self.original['pack_id'])
        self.assertEqual(table['pack_sha256'], hashlib.sha256(self.raw).hexdigest())

    def test_reference_notes_are_staged_byte_identical_and_bound_to_the_author_pack(self):
        module.stage(ROOT, self.backend)
        target = self.backend / 'core/web/study_materials' / module.REFERENCE_NOTES_NAME
        self.assertEqual(target.read_bytes(), (self.source / module.REFERENCE_NOTES_NAME).read_bytes())
        notes = json.loads(target.read_bytes())
        self.assertEqual(notes['pack_id'], self.original['pack_id'])
        self.assertEqual(notes['pack_sha256'], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(notes['note_count'], sum(len(card['pre']['references']) for card in self.original['cards']))

    def test_significance_notes_are_staged_byte_identical_and_bound_to_the_author_pack(self):
        module.stage(ROOT, self.backend)
        target = self.backend / 'core/web/study_materials' / module.SIGNIFICANCE_NAME
        self.assertEqual(target.read_bytes(), (self.source / module.SIGNIFICANCE_NAME).read_bytes())
        significance = json.loads(target.read_bytes())
        self.assertEqual(significance['pack_id'], self.original['pack_id'])
        self.assertEqual(significance['pack_sha256'], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(len(significance['significance']), 10)

    def test_exact_repeat_is_idempotent(self):
        self.assertEqual(module.stage(ROOT, self.backend), module.stage(ROOT, self.backend))

    def test_never_stages_into_author_directory(self):
        with self.assertRaisesRegex(ValueError, 'author'):
            module.stage(ROOT, ROOT)

    def test_unexpected_existing_files_are_not_deleted(self):
        target = self.backend / 'core/web/study_materials'
        target.mkdir(parents=True)
        existing = target / 'private-organizer.json'
        existing.write_text('preserve')
        with self.assertRaises(ValueError):
            module.stage(ROOT, self.backend)
        self.assertEqual(existing.read_text(), 'preserve')

    def test_different_prior_material_is_not_overwritten(self):
        module.stage(ROOT, self.backend)
        target = self.backend / 'core/web/study_materials' / module.TARGET_NAME
        target.write_text('preserve')
        with self.assertRaises(ValueError):
            module.stage(ROOT, self.backend)
        self.assertEqual(target.read_text(), 'preserve')

    def test_both_platforms_exclude_author_packs_and_stage_participant_projection(self):
        for name in ['prepare-bundled-runtime.ps1', 'prepare-bundled-runtime-mac.sh']:
            text = (ROOT / 'desktop/scripts' / name).read_text()
            self.assertIn('study_materials', text)
            self.assertIn('stage-discovery-study.py', text)

    def test_live_default_remains_author_pack_until_a_distribution_pack_exists(self):
        from core.web.discovery_study import DEFAULT_PACK, PACKAGED_PACK
        self.assertFalse(PACKAGED_PACK.exists())
        self.assertEqual(DEFAULT_PACK.name, module.SOURCE_NAME)


if __name__ == '__main__':
    unittest.main()
