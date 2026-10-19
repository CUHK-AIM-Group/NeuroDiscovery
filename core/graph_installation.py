"""One portable, hash-bound graph and evidence installation for app consumers.

Frozen campaign/acceptance bytes are copied, never rebased or re-approved.
Only physical file lookup changes, inside a context-local storage binding.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import threading
import zipfile

PACKAGE_SCHEMA = 'neurodiscovery.graph-package.v1'
INSTALL_SCHEMA = 'neurodiscovery.graph-installation.v1'
RELEASE_SCHEMA = 'neurodiscovery.graph-download.v1'
REPO_ROOT = Path(__file__).resolve().parents[1]
_packages = {}
_lock = threading.RLock()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Graph installation metadata must be an object')
    return value


def _key(value):
    text = str(value).replace('\\', '/')
    return text.casefold() if re.match(r'^[a-zA-Z]:/', text) else text


def _within(root, relative):
    if not isinstance(relative, str):
        raise ValueError('Graph package file must have a relative path')
    parts = PurePosixPath(relative)
    if (not relative or '\\' in relative or ':' in relative or parts.is_absolute()
            or '..' in parts.parts):
        raise ValueError('Graph package path must remain inside its installation')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Graph package path escapes its installation')
    return path


class GraphPackage:
    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.root = self.manifest_path.parent
        if self.manifest_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('Graph package manifest is too large')
        raw = self.manifest_path.read_bytes()
        self.signature = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if not isinstance(self.manifest, dict) or self.manifest.get('schema') != PACKAGE_SCHEMA:
            raise ValueError('Unsupported graph package schema')
        self._files, self._verified = {}, {}
        self._lock = threading.RLock()
        for row in self.manifest.get('files', []):
            source = row['source']
            if (not isinstance(source.get('path'), str)
                    or type(source.get('bytes')) is not int or source['bytes'] < 0
                    or type(source.get('mtime_ns')) is not int
                    or not re.fullmatch('[a-f0-9]{64}', source.get('sha256', ''))):
                raise ValueError('Invalid graph package source fingerprint')
            key = _key(source['path'])
            if key in self._files:
                raise ValueError('Duplicate graph package source')
            self._files[key] = (source, _within(self.root, row['file']))
        for name in ('graph', 'state' if self.is_formal else 'campaign'):
            if _key(self.manifest.get(name, '')) not in self._files:
                raise ValueError('Graph package is missing its ' + name)

    def resolve(self, source):
        row = self._files.get(_key(source))
        if row is None:
            raise ValueError('Graph package is missing a required dependency: ' + str(source))
        expected, path = row
        # Resolve again so replacing a parent with a symlink cannot escape.
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Graph package dependency escaped its installation')
        with self._lock:
            info = path.stat()
            signature = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
            if info.st_size != expected['bytes']:
                raise ValueError('Graph package dependency size changed')
            if self._verified.get(_key(source)) != signature:
                if sha256(path) != expected['sha256']:
                    raise ValueError('Graph package dependency hash changed')
                after = path.stat()
                if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                    raise ValueError('Graph package dependency changed during verification')
                self._verified[_key(source)] = signature
        return path

    def matches(self, fingerprint, physical=None):
        row = self._files.get(_key(fingerprint['path']))
        if row is None:
            return False
        expected, _ = row
        # Original timestamps bind receipt identity. Physical timestamps can
        # differ after unzip/copy; exact content is checked by resolve().
        if any(fingerprint.get(k) != expected[k] for k in ('bytes', 'mtime_ns', 'sha256')):
            return False
        path = self.resolve(fingerprint['path'])
        return physical is None or Path(physical).resolve() == path

    @contextmanager
    def activate(self):
        from neurooracle.src.kg_storage import installed_storage
        with installed_storage(self.resolve, self.matches):
            yield self

    @property
    def is_formal(self):
        return self.manifest.get('source_kind') == 'formal_current'

    @property
    def state(self):
        return self.resolve(self.manifest['state'])

    @property
    def campaign(self):
        return None if self.is_formal else self.resolve(self.manifest['campaign'])

    @property
    def graph(self):
        return self.resolve(self.manifest['graph'])

    @property
    def graph_revision(self):
        return self._files[_key(self.manifest['graph'])][0]['sha256']

    def validate(self):
        from core.web.claim_layer_v8 import AcceptedClaimLayer
        for original in self._files:
            self.resolve(original)
        if self.is_formal:
            from core.formal_graph_evidence import FormalGraphEvidence
            service = FormalGraphEvidence(self.state, graph_path=self.graph)
            state = service._read_current()
            graph = state['canonical_files']['knowledge_graph']
            row = self._files[_key(self.manifest['graph'])][0]
            if any(graph.get(k) != row[k] for k in ('bytes', 'sha256')):
                raise ValueError('Formal graph and CURRENT_STATE use different revisions')
            service._index()  # Complete source count/locations before activation.
            result = service.status()
            return {**result, 'release_id': self.manifest.get('release_id'),
                    'package_sha256': self.signature, 'graph_path': str(self.graph)}
        campaign = _json(self.campaign)
        graph_row = self._files[_key(self.manifest['graph'])][0]
        if campaign.get('current_graph') != graph_row:
            raise ValueError('Graph and evidence in the installation use different revisions')
        with self.activate():
            result = AcceptedClaimLayer(self.campaign).status()
        if not result.get('available'):
            raise ValueError('Graph package evidence is unavailable')
        return {**result, 'release_id': self.manifest.get('release_id'),
                'package_sha256': self.signature, 'graph_path': str(self.graph)}


def installed_package(repo_root=None):
    root = Path(repo_root or REPO_ROOT).resolve()
    override = os.environ.get('NEUROCLAW_GRAPH_INSTALLATION')
    pointer = (Path(override).expanduser() if override else
               root / 'neurooracle/data/graph-installation.json')
    if override and not pointer.is_absolute():
        pointer = root / pointer
    if not pointer.exists():
        if override:
            raise ValueError('The selected graph installation does not exist')
        return None
    if pointer.is_dir():
        pointer = pointer / 'manifest.json'
    value = _json(pointer)
    if value.get('schema') == INSTALL_SCHEMA:
        pointer = _within(pointer.parent, value['package'])
    raw = pointer.read_bytes()
    key = (str(pointer.resolve()), hashlib.sha256(raw).hexdigest())
    with _lock:
        if key not in _packages:
            if len(_packages) >= 8:
                _packages.clear()
            _packages[key] = GraphPackage(pointer)
        return _packages[key]


def configured_campaign_path(repo_root):
    package = installed_package(repo_root)
    if package is not None:
        return package.campaign
    root = Path(repo_root)
    configured = os.environ.get('NEUROCLAW_KG_CAMPAIGN')
    # A formal release replaces the obsolete development selector. Explicit
    # experiment overrides remain version-pinned and must still validate.
    from core.formal_graph_evidence import formal_state_path
    if configured is None and formal_state_path(root) is not None:
        return None
    config = root / 'neurooracle/configs/graph_explorer.json'
    if configured is None and config.exists():
        payload = _json(config)
        if payload.get('version') != 1 or not isinstance(payload.get('campaign'), str):
            raise ValueError('Invalid graph explorer configuration')
        configured = payload['campaign']
    if configured is None:
        return None
    if not configured.strip():
        raise ValueError('Empty accepted graph campaign path')
    path = Path(configured)
    return (path if path.is_absolute() else root / path).resolve()


def formal_source(repo_root=None):
    from core.formal_graph_evidence import FormalGraphEvidence, formal_state_path
    from core.web.claim_evidence import EvidenceUnavailable
    root = Path(repo_root or REPO_ROOT).resolve()
    package = installed_package(root)
    if package is not None:
        return (package.state, package.graph) if package.is_formal else None
    if os.environ.get('NEUROCLAW_KG_CAMPAIGN') is not None:
        return None
    state = formal_state_path(root)
    if state is None:
        return None
    service = FormalGraphEvidence(state)
    try:
        service._read_current()
    except EvidenceUnavailable as exc:
        raise ValueError(str(exc)) from exc
    return state, service.graph


def selected_graph_path(repo_root=None):
    root = Path(repo_root or REPO_ROOT).resolve()
    package = installed_package(root)
    if package is not None:
        return package.graph
    formal = formal_source(root)
    if formal is not None:
        return formal[1]
    if configured_campaign_path(root) is not None:
        # An explicitly selected research source must also own the concept
        # viewer. Missing or stale acceptance never falls back to a demo.
        from neurooracle.src.graph_paths import current_graph_path
        return current_graph_path(root)
    path = root / 'neurooracle/data/knowledge_graph.json'
    compressed = path.with_suffix('.json.gz')
    return path if path.exists() or not compressed.exists() else compressed


class InstalledClaimLayer:
    """Late-bound shared service; a newly installed package needs no restart."""
    def __init__(self, repo_root):
        self.root = Path(repo_root)
        self._identity, self._service = None, None
        self._selection_lock = threading.RLock()

    def _selected(self):
        from core.web.claim_layer_v8 import AcceptedClaimLayer
        from core.formal_graph_evidence import FormalGraphEvidence
        package = installed_package(self.root)
        formal = formal_source(self.root)
        campaign = package.campaign if package else configured_campaign_path(self.root)
        identity = (package.signature if package else None, str(campaign), str(formal))
        with self._selection_lock:
            if self._service is None or identity != self._identity:
                self._service = (FormalGraphEvidence(formal[0], graph_path=formal[1]) if formal
                                 else AcceptedClaimLayer(campaign))
                self._identity = identity
            return package, self._service

    def __getattr__(self, name):
        package, service = self._selected()
        value = getattr(service, name)
        if not callable(value):
            return value
        def invoke(*args, **kwargs):
            current, selected = self._selected()
            with current.activate() if current else nullcontext():
                return getattr(selected, name)(*args, **kwargs)
        return invoke

    @contextmanager
    def read_snapshot(self):
        package, service = self._selected()
        with package.activate() if package else nullcontext():
            with service.read_snapshot() as revision:
                yield revision
            # An installation switch cannot produce a mixed-revision result.
            after, after_service = self._selected()
            if (after_service is not service or
                    (after.signature if after else None) != (package.signature if package else None)):
                raise ValueError('Graph installation changed during evidence retrieval')


def export_formal_package(state_path, destination, *, release_id):
    """Publish exact formal graph/state bytes; no experimental campaign needed."""
    from core.formal_graph_evidence import FormalGraphEvidence
    state_path, destination = Path(state_path).resolve(), Path(destination).resolve()
    service = FormalGraphEvidence(state_path)
    before = state_path.read_bytes()
    service.status()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for source, relative in ((state_path, 'files/CURRENT_STATE.json'),
                             (service.graph, 'files/knowledge_graph.json')):
        info = source.stat()
        fingerprint = dict(path=str(source), bytes=info.st_size, mtime_ns=info.st_mtime_ns, sha256=sha256(source))
        target = _within(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != fingerprint['sha256']:
            raise ValueError('Formal source changed during graph export')
        rows.append(dict(source=fingerprint, file=relative))
    if state_path.read_bytes() != before:
        raise ValueError('Formal state changed during graph export')
    manifest = dict(schema=PACKAGE_SCHEMA, source_kind='formal_current', release_id=release_id,
                    graph=str(service.graph), state=str(state_path), files=rows)
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return GraphPackage(destination / 'manifest.json').validate()


def export_package(campaign_path, destination, *, release_id):
    """Copy the complete active fingerprint closure without altering receipts."""
    from core.web.claim_layer_v8 import AcceptedClaimLayer
    from neurooracle.src.shared_relation_catalog import check_file
    campaign_path, destination = Path(campaign_path).resolve(), Path(destination).resolve()
    before = _json(campaign_path)
    if before.get('status') != 'COMPLETED' or before.get('active_process') is not None:
        raise ValueError('Only a settled accepted campaign can be exported')
    source_layer = AcceptedClaimLayer(campaign_path)
    source_status = source_layer.status()
    info = campaign_path.stat()
    campaign_fp = dict(path=str(campaign_path), bytes=info.st_size,
                       mtime_ns=info.st_mtime_ns, sha256=sha256(campaign_path))
    pending = [campaign_fp]
    inputs, physical = {}, {}

    def references(value):
        if isinstance(value, dict):
            if all(k in value for k in ('path', 'bytes', 'mtime_ns', 'sha256')):
                pending.append(value)
            else:
                for child in value.values():
                    references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)

    def graph_metadata(path):
        # save_graph writes metadata before concepts; never load a multi-GB
        # graph just to find its explicitly bound sibling registries.
        with path.open('rb') as stream:
            prefix = stream.read(4 * 1024 * 1024).decode('utf-8', errors='replace')
        match = re.search(r'"metadata"\s*:\s*', prefix)
        if not match:
            raise ValueError('Graph metadata header is missing')
        return json.JSONDecoder().raw_decode(prefix[match.end():])[0]

    while pending:
        fp = pending.pop()
        key = _key(fp['path'])
        if key in inputs:
            if inputs[key] != fp:
                raise ValueError('Conflicting source fingerprints in graph package')
            continue
        path = check_file(fp, full_hash=True)
        inputs[key], physical[key] = fp, path
        if key == _key(campaign_fp['path']):
            for name in (*source_layer.INPUTS, 'current_claim_layer'):
                if before.get(name):
                    references(before[name])
        elif key == _key(before['current_graph']['path']):
            metadata = graph_metadata(path)
            for name in ('entity_identity', 'paper_identity', 'relation_evidence'):
                binding = metadata.get(name) or {}
                if binding.get('registry'):
                    sibling = _within(path.parent, binding['registry'])
                    record = sibling.stat()
                    pending.append(dict(path=str(Path(fp['path']).parent / binding['registry']),
                                        bytes=record.st_size, mtime_ns=record.st_mtime_ns,
                                        sha256=binding['sha256']))
        elif path.suffix.casefold() in {'.json', '.jsonl', '.ndjson'}:
            with path.open(encoding='utf-8') as stream:
                if path.suffix.casefold() == '.json':
                    references(json.load(stream))
                else:
                    for line in stream:
                        if line.strip():
                            references(json.loads(line))

    # Preserve sibling relationships as well as exact bytes. The common root
    # is removed; no machine-specific absolute target is used for installation.
    common = Path(os.path.commonpath([str(path.parent) for path in physical.values()]))
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for key, fp in inputs.items():
        source = physical[key]
        relative = 'files/' + source.relative_to(common).as_posix()
        target = _within(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != fp['sha256']:
            raise ValueError('Source changed while building graph package')
        rows.append(dict(source=fp, file=relative))
    if _json(campaign_path) != before or source_layer.status() != source_status:
        raise ValueError('Accepted source changed during graph export')
    value = dict(schema=PACKAGE_SCHEMA, release_id=release_id,
                 graph=before['current_graph']['path'], campaign=campaign_fp['path'],
                 files=sorted(rows, key=lambda row: _key(row['source']['path'])))
    (destination / 'manifest.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return GraphPackage(destination / 'manifest.json').validate()


def install_archive(archive, data_directory, *, expected_sha256=None, expected_package_sha256=None):
    """Validate in a fresh directory, then atomically switch one app pointer."""
    archive, data = Path(archive).resolve(), Path(data_directory).resolve()
    if expected_sha256 and sha256(archive) != expected_sha256:
        raise ValueError('Downloaded graph package hash does not match its release')
    data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.graph-install-', dir=data) as temporary:
        staging = Path(temporary) / 'package'
        staging.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            seen = set()
            for entry in zipped.infolist():
                target = _within(staging, entry.filename)
                key = str(target).casefold() if os.name == 'nt' else str(target)
                if key in seen or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError('Duplicate or linked graph archive entry')
                seen.add(key)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(entry) as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
        package = GraphPackage(staging / 'manifest.json')
        if expected_package_sha256 and package.signature != expected_package_sha256:
            raise ValueError('Graph package manifest differs from its release')
        return _activate_package(staging, data)


def _activate_package(staging, data):
    package = GraphPackage(staging / 'manifest.json')
    result = package.validate()
    relative = 'graph-packages/' + package.signature
    target = _within(data, relative)
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        GraphPackage(target / 'manifest.json').validate()
    else:
        os.replace(staging, target)
    pointer = data / 'graph-installation.json'
    descriptor, name = tempfile.mkstemp(prefix='.graph-pointer-', dir=data)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(dict(schema=INSTALL_SCHEMA, package=relative + '/manifest.json'), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, pointer)
    finally:
        if Path(name).exists():
            Path(name).unlink()
    return {**result, 'graph_path': str(GraphPackage(target / 'manifest.json').graph),
            'installation': str(pointer)}


def install_formal_snapshot(publication_bytes, compressed, data_directory):
    """Install the existing public full graph, retaining its publication receipt."""
    import zstandard
    publication = json.loads(publication_bytes)
    if (publication.get('schema_version') != 'neurodiscovery.graph-publication.v1' or
            publication.get('source_status') != 'canonical_current'):
        raise ValueError('Invalid formal graph publication')
    graph = publication['graph']
    if graph.get('compression') != 'zstandard':
        raise ValueError('Unsupported formal graph compression')
    compressed, data = Path(compressed), Path(data_directory)
    if compressed.stat().st_size != graph['bytes'] or sha256(compressed) != graph['sha256']:
        raise ValueError('Downloaded formal graph hash or size differs from its publication')
    data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.formal-install-', dir=data) as temporary:
        staging = Path(temporary) / 'package'
        files = staging / 'files'
        files.mkdir(parents=True)
        state_path = files / 'PUBLICATION.json'
        state_path.write_bytes(publication_bytes)
        target = files / 'knowledge_graph.json'
        digest, size = hashlib.sha256(), 0
        with compressed.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as decoded, target.open('xb') as output:
            for block in iter(lambda: decoded.read(4 * 1024 * 1024), b''):
                size += len(block)
                if size > graph['uncompressed_bytes']:
                    raise ValueError('Formal graph decompression exceeds its published size')
                digest.update(block)
                output.write(block)
        if size != graph['uncompressed_bytes'] or digest.hexdigest() != graph['uncompressed_sha256']:
            raise ValueError('Decompressed formal graph differs from its publication')
        rows = []
        for path in (state_path, target):
            info = path.stat()
            rows.append(dict(source=dict(path=path.name, bytes=info.st_size,
                                         mtime_ns=info.st_mtime_ns, sha256=sha256(path)), file='files/' + path.name))
        manifest = dict(schema=PACKAGE_SCHEMA, source_kind='formal_current',
                        release_id=publication['release_id'], graph=target.name, state=state_path.name, files=rows)
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return _activate_package(staging, data)


def pack_archive(manifest, archive, *, remote_path):
    package = GraphPackage(manifest)
    package.validate()
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves earlier releases. Zip64 handles the full KG.
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as zipped:
        members = {'manifest.json'} | {row['file'] for row in package.manifest['files']}
        for relative in sorted(members):
            zipped.write(_within(package.root, relative), relative)
    release = dict(schema=RELEASE_SCHEMA, release_id=package.manifest['release_id'],
                   package_sha256=package.signature,
                   archive=dict(path=remote_path, sha256=sha256(archive), bytes=archive.stat().st_size))
    with archive.with_suffix(archive.suffix + '.release.json').open('x', encoding='utf-8') as stream:
        json.dump(release, stream, ensure_ascii=False, indent=2)
    return release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument('--campaign', type=Path)
    source.add_argument('--state', type=Path)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--release-id', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('manifest', type=Path)
    pack = commands.add_parser('pack')
    pack.add_argument('manifest', type=Path)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--remote-path', required=True)
    install = commands.add_parser('install')
    install.add_argument('archive', type=Path)
    install.add_argument('--data-directory', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        result = (export_formal_package(args.state, args.output, release_id=args.release_id) if args.state
                  else export_package(args.campaign, args.output, release_id=args.release_id))
    elif args.command == 'verify':
        result = GraphPackage(args.manifest).validate()
    elif args.command == 'pack':
        result = pack_archive(args.manifest, args.output, remote_path=args.remote_path)
    else:
        result = install_archive(args.archive, args.data_directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
