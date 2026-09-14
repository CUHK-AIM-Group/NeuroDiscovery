"""Four source-reviewed research intentions incorrectly flattened into findings.

This is an exact bounded retirement, not a classifier for reviews, hypotheses,
weak associations, or any other article. It never synthesizes a null finding.
"""
from .kg_identity_pilot import digest
from .kg_literal_endpoint_repair import edge_owner, reviewed_edges
from .kg_scoped_structure import source_documents
from .kg_source_scope_resolution import compacted_ordinal, original_ordinal

VERSION = 'kg.research_statement_retirement.v1'
REVIEWS = {
    'CLM:15337253ab285fa6': dict(
        sha='cccbce10ae39f3df81d5e38d9a1edc2cd8694baa78af580f9f7e3632506bf82e',
        pmid='12788246', doi='10.1016/s0006-3223(03)00069-6', original='requires',
        subject='pediatric bipolar disorder',
        object='models of state switching across positive negative and irritable affect',
        raw='The review identifies marked state fluctuations, rapid cycles, irritability and ADHD comorbidity as developmental features requiring cortico-limbic-striatal and attention-emotion regulation research.',
        fragment='Potential foci for research on the pathophysiology of pediatric BPD include',
        reason='research_requirement_not_empirical_disease_model_association'),
    'CLM:26b3837bbea8a29f': dict(
        sha='108db2f1b2007d0b6ccca82cdde4d5aec738aa95e412d7a77734b6d56ffe0220',
        pmid='12716235', doi='10.4088/jcp.v64n0402', original='is_designed_to_test',
        subject='algorithm-driven disease management',
        object='clinical and economic outcomes in schizophrenia bipolar disorder and major depression',
        raw='TMAP-3 compared medication algorithms plus education, physician training, documentation and coordinator prompts versus treatment as usual across public clinics.',
        fragment='Analyses were based on hierarchical linear models designed to test',
        reason='study_design_not_reported_comparative_outcome'),
    'CLM:a56168cd836df3ce': dict(
        sha='16639c5d95f030020557ac157a5a444c88b5e161ecf97693e09bffa1a159c4b2',
        pmid='14601038', doi='10.1002/ajmg.c.20015', original='may_explain',
        subject='epigenetic mechanisms',
        object='non-Mendelian features and inconsistent genetic findings in bipolar disorder',
        raw='The review argues imprinting, tissue-specific effects, epigenetic polymorphism and X-inactivation regulation fit twin discordance, age susceptibility, sex differences and phase fluctuation.',
        fragment='systematic large-scale epiG studies of BD have to be initiated',
        reason='proposed_explanatory_hypothesis_not_established_association'),
    'CLM:cf7e3ac19a7c4adc': dict(
        sha='ee33bdb499142c23450395d9cd2dac9a1ead779142c73407ee9a37e9ba3c8d49',
        pmid='12534655', doi='10.1046/j.1440-1614.2003.01098.x', original='may_reduce',
        subject='schema-focused cognitive therapy',
        object='bipolar relapse vulnerability through changes in illness adaptation and self-concept',
        raw='The proposed model targets temperament, developmental experiences and cognitive vulnerabilities to improve adjustment and functional recovery alongside pharmacotherapy.',
        fragment='This proposed treatment, combined with pharmacotherapy, may offer new psychotherapeutic options for the future.',
        reason='proposed_treatment_model_not_tested_relapse_reduction'),
}


def public_source_proof(xml):
    docs = source_documents([xml]); proof = {}
    for cid, review in REVIEWS.items():
        d = docs.get(review['pmid'])
        if not d or review['doi'] not in {s.casefold() for s in d['dois']}:
            raise ValueError('own explicit PMID/DOI source differs')
        if review['fragment'] not in d['abstract']:
            raise ValueError('reviewed complete-source context differs')
        proof[cid] = dict(pmid=review['pmid'], doi=review['doi'], title=d['title'],
            abstract_sha256=digest(d['abstract']), fragment_sha256=digest(review['fragment']),
            reason=review['reason'], source_review_only_not_generic_genre_exclusion=True)
    return proof


def review_science(cid, sha, md, proof):
    r = REVIEWS.get(cid)
    if r is None or sha != r['sha']:
        raise ValueError('unreviewed or changed claim')
    paper = md.get('source_paper') or {}; inner = md.get('metadata') or {}
    if str(paper.get('pmid')) != r['pmid'] or str(paper.get('doi') or '').casefold() != r['doi']:
        raise ValueError('claim own paper differs')
    if cid not in proof or (proof[cid]['pmid'], proof[cid]['doi'], proof[cid]['reason']) != (r['pmid'], r['doi'], r['reason']):
        raise ValueError('missing reviewed source proof')
    if (md.get('subject_name'), md.get('object_name'), md.get('raw_text')) != (r['subject'], r['object'], r['raw']):
        raise ValueError('reviewed statement scope differs')
    if md.get('predicate') != 'is_associated_with' or md.get('negated') is not False:
        raise ValueError('not the reviewed affirmative flattening')
    if inner.get('original_predicate') != r['original']:
        raise ValueError('original research-intention expression differs')
    evidence = md.get('evidence') or {}
    if evidence.get('legacy_value') != md['raw_text'] or any(evidence.get(k) is not None for k in ('p_value','effect_size','sample_size')):
        raise ValueError('reviewed saved evidence differs')
    return r


def reviewed_claim(row, owned, proof):
    cid = row['id']; md = row['metadata']
    r = review_science(cid, digest(row), md, proof)
    if len(owned) != 3 or any(edge_owner(e) != cid for _, e in owned):
        raise ValueError('not the reviewed three-edge owned closure')
    if reviewed_edges(cid, row, row, owned) != []:
        raise ValueError('unchanged source closure differs')
    if sum(e['relation_type'] == 'about' for _, e in owned) != 2:
        raise ValueError('expected one science edge and two about edges')
    return dict(claim_id=cid, claim_sha256=digest(row), pmid=r['pmid'], doi=r['doi'],
        reason=r['reason'], source_proof=proof[cid],
        owned_edges=[dict(ordinal=o, edge_sha256=digest(e)) for o,e in sorted(owned)],
        no_null_or_replacement_claim_synthesized=True, source_document_not_deleted=True,
        record_preimages_saved=False)
