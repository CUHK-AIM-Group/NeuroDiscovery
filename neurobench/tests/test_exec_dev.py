"""Input-boundary checks not covered by numerical case qualification."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from neurobench.runner.exec_dev.common import Case, contained, file_map, finish, load_case, write_json
from neurobench.runner.exec_dev.oracle import close, evaluate, Rejected


class ExecDevelopmentBoundaries(unittest.TestCase):
    def test_absolute_and_parent_artifact_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('../outside.txt', str(Path(d).resolve() / 'outside.txt')):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    contained(d, name)

    def test_symlink_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); original = root / 'original.txt'; original.write_text('input')
            out = root / 'output'; out.mkdir()
            try:
                (out / 'link.txt').symlink_to(original)
            except OSError:
                self.skipTest('This account cannot create symlinks')
            with self.assertRaisesRegex(ValueError, 'linked'):
                file_map(out)

    def test_nan_never_passes_numerical_comparison(self):
        with self.assertRaises(Rejected):
            close([float('nan')], [0], 'nan')

    def test_wrong_shape_never_broadcasts_into_a_pass(self):
        with self.assertRaises(Rejected):
            close([[1, 1]], [1, 1], 'shape')

    def test_packet_pin_rejects_changed_index_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d) / 'INDEX.json', {'cases': []})
            with self.assertRaisesRegex(ValueError, 'index_binding'):
                load_case(d, 'E01', '0' * 64)

    def test_malformed_candidate_nifti_is_task_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); inp = root / 'case/inputs'; inp.mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), inp / 'map.nii.gz')
            write_json(inp / 'parameters.json', {'threshold': .5})
            case = Case(inp.parent, {'case_id': 'E03', 'input_sha256': file_map(inp),
                                    'required_outputs': ['mask.nii.gz', 'stats.json']})
            out = root / 'out'; out.mkdir()
            (out / 'mask.nii.gz').write_bytes(b'not a NIfTI payload')
            write_json(out / 'stats.json', {'voxel_count': 8, 'volume_ml': .008})
            finish(case, out)
            self.assertEqual(evaluate(case, out)['verdict'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
