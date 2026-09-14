"""Regression against lexical false-positive diagnosis and over-clearance."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_kg_all_gene_endpoints import classify


@pytest.mark.parametrize('name,labels,category,matched',[
    ('greater number of depressive episodes',['NUMB','NUMB gene'],'word_interior_alias_only',['numb']),
    ('major depressive disorder symptom severity',['SORD'],'word_interior_alias_only',['sord']),
    ('whole-brain and nucleus accumbens myelin',['LARS','LEUS'],'word_interior_alias_only',['leus']),
    ('cortical surface area',['FANCE','FACE'],'word_interior_alias_only',['face']),
    ('frontotemporal alterations',['VCP','TERA'],'word_interior_alias_only',['tera']),
    ('VCP',['VCP','p97'],'exact_label_context_still_required',['vcp']),
    (' VCP ',['vcp'],'exact_label_context_still_required',['vcp']),
    ('VCP treatment',['VCP'],'symbol_phrase_requires_scope_review',['vcp']),
    ('APOE ε4 carrier status',['APOE'],'symbol_phrase_requires_scope_review',['apoe']),
    ('cortical volume',['NUMB'],'no_label_alignment',[]),
    ('', ['NUMB'],'missing_name',[]),
    (None,['NUMB'],'missing_name',[]),
    ('p97-related assay',['p97'],'symbol_phrase_requires_scope_review',['p97']),
    ('NUMB2',['NUMB'],'word_interior_alias_only',['numb']),
    ('protein X',['[X]'],'no_label_alignment',[]),
])
def test_diagnostic_only_categories(name,labels,category,matched):
    assert classify(name,labels)==(category,matched)


def test_label_list_is_not_modified():
    labels=['VCP','vcp','',None]
    assert classify('VCP',labels)==('exact_label_context_still_required',['vcp'])
    assert labels==['VCP','vcp','',None]


def test_exact_symbol_never_automatically_clears_species_or_homonyms():
    assert 'required' in classify('VCP',['VCP'])[0]
    assert classify('NUMB mutation',['NUMB'])[0]=='symbol_phrase_requires_scope_review'
