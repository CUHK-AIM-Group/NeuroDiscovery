"""Complete, size-bounded UTF-8 acceptance packets; no provider dispatch."""

import hashlib
import json
from pathlib import Path


MAX_REVIEW_ARTIFACTS = 12
MAX_REVIEW_CONTENT_BYTES = 512 * 1024
MAX_REVIEW_PACKET_BYTES = 1024 * 1024


class ReviewPacketError(ValueError):
    pass


def build_review_packet(run):
    paths = run.state.get('artifacts', [])
    if not paths or len(paths) > MAX_REVIEW_ARTIFACTS:
        raise ReviewPacketError('Submit 1–12 complete reviewable artifacts; no files were omitted or reviewed.')
    artifacts = []
    bindings = []
    remaining = MAX_REVIEW_CONTENT_BYTES
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_relative_to(run.workspace) or not path.is_file():
            raise ReviewPacketError('A review artifact is missing or outside the workspace.')
        with path.open('rb') as handle:
            content = handle.read(remaining + 1)
        if len(content) > remaining:
            raise ReviewPacketError('Complete artifact content exceeds the 512 KiB review limit. Submit a focused '
                                   'final report with sufficient source evidence, not redundant drafts or an entire '
                                   'unrelated corpus. Preserve originals; no partial packet was sent.')
        if not content:
            raise ReviewPacketError('An empty artifact cannot be reviewed.')
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ReviewPacketError('Review artifacts must be valid UTF-8 text. Supply a complete text report '
                                   'for binary results with source bindings and limitations.') from exc
        remaining -= len(content)
        binding = {'path': str(path), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        bindings.append(binding)
        artifacts.append({**binding, 'content': text, 'partial': False})
    packet = {'objective': run.state.get('objective'), 'claim': run.state.get('summary'),
              'steering_instructions': run.state.get('steering_instructions', []),
              'validation_claim': run.state.get('validation'), 'evidence': run.state['evidence'],
              'reading_ledger': run.state.get('reading_ledger', {}),
              'analysis_notes': run.state.get('analysis_notes', []),
              'artifacts': artifacts, 'artifact_count': len(artifacts),
              'review_contract': {'complete_artifact_content': True,
                                  'review_owner': 'runtime', 'scientific_validation': False}}
    serialized = json.dumps(packet, ensure_ascii=False)
    if len(serialized.encode('utf-8')) > MAX_REVIEW_PACKET_BYTES:
        raise ReviewPacketError('Complete review packet exceeds 1 MiB including metadata. No review call was '
                               'made; reduce the submission without removing evidence necessary for acceptance.')
    if bindings != run.verify_artifacts():
        raise ReviewPacketError('Artifacts changed while preparing review; no packet was sent.')
    return serialized, bindings
