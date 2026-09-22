"""Relocation must preserve frozen identity and fail closed on unrelated drift."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from neurooracle.src import kg_storage
from neurooracle.src.shared_relation_catalog import check_file


def fp(path):
    info=path.stat()
    return dict(path=str(path),bytes=info.st_size,mtime_ns=info.st_mtime_ns,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def moved(tmp_path,monkeypatch):
    repo=tmp_path/'repo';target=tmp_path/'target'
    source=repo/'tmp/papers/input.json';dest=target/'tmp/papers/input.json'
    source.parent.mkdir(parents=True);dest.parent.mkdir(parents=True)
    source.write_bytes(b'unchanged scientific source')
    binding=fp(source)
    dest.write_bytes(source.read_bytes())
    stamp=(binding['mtime_ns']//10_000_000-1)*10_000_000
    os.utime(dest,ns=(stamp,stamp))
    catalog=target/'RELOCATION.sqlite'
    with sqlite3.connect(catalog) as c:
        c.execute('''create table files(source_key text primary key,destination_key text unique,
          destination text,bytes integer,source_mtime_ns integer,destination_mtime_ns integer,sha256 text)''')
        c.execute('insert into files values(?,?,?,?,?,?,?)',
          (kg_storage._key(source),kg_storage._key(dest),str(dest),binding['bytes'],binding['mtime_ns'],dest.stat().st_mtime_ns,binding['sha256']))
    config=repo/'storage.json'
    config.write_text(json.dumps(dict(version=1,active=True,repository=str(repo),destination_root=str(target),
      roots=[dict(source=str(source.parent),destination=str(dest.parent))],catalog=fp(catalog))))
    monkeypatch.setattr(kg_storage,'ROOT',repo)
    monkeypatch.setattr(kg_storage,'CONFIG',config)
    return source,dest,binding,catalog,repo,config


def test_old_receipt_resolves_identical_moved_content(moved):
    source,dest,binding,*_=moved
    assert check_file(binding,full_hash=True)==dest
    assert kg_storage.storage_path(source)==dest
    assert kg_storage.stat_matches({**binding,'path':str(dest)})


def test_stale_original_fingerprint_is_not_repaired_by_relocation(moved):
    _,_,binding,*_=moved
    for change in ({'mtime_ns':binding['mtime_ns']+100},{'bytes':binding['bytes']+1},{'sha256':'0'*64}):
        with pytest.raises(ValueError,match='fingerprint changed'):
            check_file({**binding,**change})


def test_later_destination_modification_rejected(moved):
    _,dest,binding,*_=moved
    dest.write_bytes(b'a changed source')
    with pytest.raises(ValueError,match='fingerprint changed'):
        check_file(binding)


def test_catalog_change_rejected(moved):
    _,_,binding,catalog,*_=moved
    with catalog.open('ab') as handle:handle.write(b'tampered')
    with pytest.raises(ValueError,match='catalog changed'):
        check_file(binding)


def test_same_stat_catalog_tampering_rejected_before_first_load(moved):
    _,_,binding,catalog,*_=moved
    before=catalog.stat();raw=bytearray(catalog.read_bytes());raw[-1]^=1
    catalog.write_bytes(raw);os.utime(catalog,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='catalog integrity'):
        check_file(binding)


def test_staging_allows_registered_move_but_not_fixed_data(moved):
    _,dest,_,_,repo,_=moved
    assert kg_storage.staging_directory('tmp/papers',repo)==dest.parent
    for value in ('neurooracle/data','tmp/../neurooracle/data',str(dest.parent)):
        with pytest.raises(ValueError,match='dedicated staging'):
            kg_storage.staging_directory(value,repo)


def test_no_catalog_does_not_relax_native_checks(tmp_path,monkeypatch):
    monkeypatch.setattr(kg_storage,'CONFIG',tmp_path/'absent.json')
    source=tmp_path/'a';source.write_bytes(b'original');binding=fp(source)
    assert check_file(binding,full_hash=True)==source
    with pytest.raises(ValueError,match='fingerprint changed'):
        check_file({**binding,'mtime_ns':binding['mtime_ns']+1})
