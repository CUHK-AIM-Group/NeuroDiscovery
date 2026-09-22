"""Verified physical relocation, without rewriting frozen scientific receipts.

Legacy paths remain usable through directory junctions. The relocation catalog
only translates an exact, hashed source fingerprint to its verified target;
unrelated stale fingerprints still fail normal integrity checks.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'neurooracle/configs/kg_storage.json'


def _key(path):
    return os.path.normcase(os.path.abspath(path)).replace('\\', '/')


@lru_cache(maxsize=4)
def _read_config(path, size, stamp):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('version') != 1:
        raise ValueError('Unsupported KG storage configuration')
    if not value.get('active'):
        return None
    repository = Path(value['repository'])
    destination = Path(value['destination_root'])
    if _key(repository) != _key(ROOT):
        raise ValueError('KG storage configuration belongs to another repository')
    for row in value['roots']:
        src, dst = Path(row['source']), Path(row['destination'])
        if not src.is_absolute() or not dst.is_absolute():
            raise ValueError('KG storage roots must be absolute')
        if not src.is_relative_to(repository) or not dst.is_relative_to(destination):
            raise ValueError('KG storage root outside authorized scope')
        if src.relative_to(repository) != dst.relative_to(destination):
            raise ValueError('KG storage mapping changes the relative data path')
    return value


def configuration():
    if not CONFIG.exists():
        return None
    info = CONFIG.stat()
    return _read_config(str(CONFIG), info.st_size, info.st_mtime_ns)


def _mapped(path, config):
    path = Path(os.path.abspath(path))
    if config:
        for row in config['roots']:
            source = Path(row['source'])
            if path.is_relative_to(source):
                return Path(row['destination']) / path.relative_to(source)
    return None


def storage_path(path):
    """Resolve current physical storage for a logical or already moved path."""
    path = Path(path)
    return (_mapped(path, configuration()) or path).resolve()


@lru_cache(maxsize=4)
def _verified_catalog(path, size, stamp, sha):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    info = Path(path).stat()
    if (info.st_size,info.st_mtime_ns)!=(size,stamp) or digest.hexdigest()!=sha:
        raise ValueError('KG relocation catalog integrity check failed')
    return path


def stat_matches(fingerprint, path=None):
    """Accept native metadata or an exact cryptographically verified relocation."""
    physical = Path(path) if path is not None else storage_path(fingerprint['path'])
    info = physical.stat()
    if (info.st_size,info.st_mtime_ns)==(fingerprint['bytes'],fingerprint['mtime_ns']):
        return True
    config = configuration()
    if config is None:
        return False
    binding = config['catalog']
    catalog = Path(binding['path'])
    current = catalog.stat()
    if (current.st_size,current.st_mtime_ns)!=(binding['bytes'],binding['mtime_ns']):
        raise ValueError('KG relocation catalog changed')
    _verified_catalog(str(catalog),current.st_size,current.st_mtime_ns,binding['sha256'])
    with sqlite3.connect(catalog.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        row = db.execute('''select destination,bytes,source_mtime_ns,destination_mtime_ns,sha256
          from files where source_key=? or destination_key=?''',
          (_key(fingerprint['path']),_key(fingerprint['path']))).fetchone()
    if row is None:
        return False
    destination,size,old_stamp,new_stamp,sha = row
    return (physical.resolve()==Path(destination).resolve()
            and (fingerprint['bytes'],fingerprint['mtime_ns'])==(size,old_stamp)
            and (info.st_size,info.st_mtime_ns)==(size,new_stamp)
            and ('sha256' not in fingerprint or fingerprint['sha256']==sha))


def staging_directory(value, root):
    """Keep writes in the logical staging scope and its registered physical move."""
    logical = Path(os.path.abspath(Path(root) / value))
    staging = Path(os.path.abspath(Path(root) / 'tmp'))
    if not logical.is_relative_to(staging):
        raise ValueError('Output must stay in the dedicated staging workspace')
    config = configuration()
    mapped = _mapped(logical, config)
    physical = (mapped or logical).resolve()
    if mapped is not None:
        # Only the registered destination, with no further junction escape.
        if physical != Path(os.path.abspath(mapped)):
            raise ValueError('Relocated staging path escapes its registered destination')
    elif not physical.is_relative_to(staging.resolve()):
        raise ValueError('Output must stay in the dedicated staging workspace')
    return physical
