"""Stage participant-only, versioned discovery materials for desktop builds.

The source pack is an author/audit artifact. Do not distribute its private
organizer mapping, copy collected answers, or regenerate scientific evidence.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_NAME = 'cs1_discovery_pilot_v16.json'
TARGET_NAME = 'cs1_discovery_pilot_v16_desktop_v1.json'
CATALOG_NAME = 'cs1_discovery_en_v15.json'
ASSIGNMENTS_NAME = 'cs1_discovery_assignments_v12.json'
REFERENCE_NOTES_NAME = 'cs1_discovery_reference_notes_v11.json'
SIGNIFICANCE_NAME = 'cs1_discovery_significance_v11.json'
PUBLIC_FIELDS = ('schema_version', 'protocol_version', 'status', 'questions',
                 'issues', 'public_meta', 'common_pre', 'common_post', 'cards', 'scoring')


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def participant_pack(raw):
    author = json.loads(raw)
    if (author['status'] != 'pilot_only' or len(author['cards']) != 10 or len(author['questions']) != 5
            or author['pack_id'] != 'neurodiscovery-multitopic-20260922-v16'):
        raise ValueError('Expected the reviewed v16 ten-case, five-question panel.')
    if author['public_meta'].get('review_flow') != 'single_round':
        raise ValueError('Expected the single-round material version.')
    result = {key: copy.deepcopy(author[key]) for key in PUBLIC_FIELDS}
    result['pack_id'] = author['pack_id'] + '-desktop-v1'
    result['distribution'] = {
        'version': 'participant-only-v1', 'author_pack_id': author['pack_id'],
        'author_pack_sha256': hashlib.sha256(raw).hexdigest(),
        'participant_fields_unchanged': True,
    }
    return result


def stage(source_root, backend_root):
    source = Path(source_root).resolve() / 'core/web/study_materials'
    target = Path(backend_root).resolve() / 'core/web/study_materials'
    if source == target:
        raise ValueError('Refusing to stage into the author material directory.')
    if not (source / SOURCE_NAME).is_file():
        public = Path(source_root).resolve() / 'desktop/evaluation-materials'
        manifest = json.loads((public / 'manifest.json').read_bytes())
        expected = {TARGET_NAME, CATALOG_NAME, ASSIGNMENTS_NAME, REFERENCE_NOTES_NAME, SIGNIFICANCE_NAME}
        if set(manifest['files']) != expected:
            raise ValueError('Unexpected public material inventory')
        for name, digest in manifest['files'].items():
            if hashlib.sha256((public / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Public material hash mismatch: {name}')
        pack = json.loads((public / TARGET_NAME).read_bytes())
        if set(pack) != set(PUBLIC_FIELDS) | {'pack_id', 'distribution'}:
            raise ValueError('Unexpected participant fields')
        if target.exists() and any(target.iterdir()):
            raise ValueError('Use an empty material target')
        target.mkdir(parents=True, exist_ok=True)
        for name in expected:
            shutil.copyfile(public / name, target / name)
        return manifest
    raw = (source / SOURCE_NAME).read_bytes()
    pack = participant_pack(raw)
    pack_bytes = serialize(pack)
    catalog = json.loads((source / CATALOG_NAME).read_bytes())
    if catalog['source_pack'] != SOURCE_NAME or catalog['source_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('English catalog is not bound to the exact reviewed source.')
    # Only metadata changes; the entire translation dictionary remains identical.
    catalog['version'] += '-desktop-v1'
    catalog['source_pack'] = TARGET_NAME
    catalog['source_sha256'] = hashlib.sha256(pack_bytes).hexdigest()
    # The assignment table deals public packet IDs only and must be bound to the
    # exact author pack; it ships unchanged next to the participant projection.
    assignments_raw = (source / ASSIGNMENTS_NAME).read_bytes()
    assignments = json.loads(assignments_raw)
    if (assignments.get('pack_id') != pack['distribution']['author_pack_id']
            or assignments.get('pack_sha256') != pack['distribution']['author_pack_sha256']):
        raise ValueError('Assignment table is not bound to the exact author pack.')
    # Reference notes are display annotations bound to the exact author pack;
    # they ship unchanged like the assignment table.
    notes_raw = (source / REFERENCE_NOTES_NAME).read_bytes()
    notes = json.loads(notes_raw)
    if (notes.get('pack_id') != pack['distribution']['author_pack_id']
            or notes.get('pack_sha256') != pack['distribution']['author_pack_sha256']):
        raise ValueError('Reference notes are not bound to the exact author pack.')
    # Significance sentences are display annotations bound to the exact author
    # pack; they ship unchanged like the reference notes.
    significance_raw = (source / SIGNIFICANCE_NAME).read_bytes()
    significance = json.loads(significance_raw)
    if (significance.get('pack_id') != pack['distribution']['author_pack_id']
            or significance.get('pack_sha256') != pack['distribution']['author_pack_sha256']):
        raise ValueError('Significance notes are not bound to the exact author pack.')
    files = {TARGET_NAME: pack_bytes, CATALOG_NAME: serialize(catalog),
             ASSIGNMENTS_NAME: assignments_raw, REFERENCE_NOTES_NAME: notes_raw,
             SIGNIFICANCE_NAME: significance_raw}
    if target.exists():
        extra = {p.name for p in target.iterdir()} - set(files)
        if extra:
            raise ValueError('Material target contains non-distribution files; refusing to overwrite or delete.')
        for name, data in files.items():
            if (target / name).exists() and (target / name).read_bytes() != data:
                raise ValueError('Different material already staged; use a new build directory.')
    target.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (target / name).write_bytes(data)
    return {'pack_id': pack['pack_id'], 'participant_fields_unchanged': True,
            'files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--backend', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.backend), indent=2))
