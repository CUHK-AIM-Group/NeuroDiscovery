'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../study.html'), 'utf8');
const code = source.slice(source.indexOf('    function relationLabel('), source.indexOf('    function evidenceDirectionPhrase('));

function fixture(language) {
  const labels = {
    relationUnknown: ['Not reported', '未报告'],
    relationExact: ['Same', '相同'],
    diseaseRelation: ['Disease', '疾病'],
    regionRelation: ['Brain region', '脑区'],
    measureRelation: ['Measure', '指标'],
    relationToHypothesis: ['Relationship to the hypothesis', '与假设的关系'],
  };
  const context = {
    lang: language,
    t: key => labels[key]?.[language === 'zh' ? 1 : 0] || key,
    candidateDirection: () => 'difference',
    candidateChain: () => ({}),
    detectPaperDirection: () => 'increase',
    escapeHtml: value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;'),
  };
  vm.createContext(context);
  vm.runInContext(code, context);
  return context;
}

test('direction-free hypotheses show both increases and decreases in green', () => {
  for (const language of ['en', 'zh']) {
    const context = fixture(language);
    for (const [direction, english, chinese] of [['increase', 'Increase', '升高'], ['decrease', 'Decrease', '降低']]) {
      const paper = {kgMatchFacets: {disease: 'matched', region: 'matched', measure: 'matched', direction}};
      const original = JSON.stringify(paper);
      const analysis = context.paperRelationAnalysis({}, paper);
      const html = context.paperRelationHtml(analysis);
      assert.equal(analysis.directionRelation, 'unknown');
      assert.equal(analysis.reportedResult, language === 'zh' ? chinese : english);
      assert.ok(analysis.summary.includes(context.paperResultLabel()));
      assert.ok(html.includes(analysis.reportedResult));
      assert.doesNotMatch(html, /Direction|方向|Not reported|未报告/);
      assert.equal(analysis.reportedResultTone, 'exact');
      assert.ok(html.includes(`<span class="paper-relation exact"><span class="paper-relation-key">${context.paperResultLabel()}</span>`));
      assert.equal(JSON.stringify(paper), original);
    }
  }
});

test('missing or unsupported structured findings stay missing, without text fallback', () => {
  for (const language of ['en', 'zh']) {
    const context = fixture(language);
    context.detectPaperDirection = () => { throw new Error('Must not override structured missingness'); };
    for (const direction of [undefined, null, '', 'not_reported', 'unrecognized', '<script>']) {
      const paper = {kgMatchFacets: {direction}, title: 'Increased activity'};
      assert.equal(context.paperReportedResult({}, paper), language === 'zh' ? '未报告' : 'Not reported');
      assert.equal(context.paperReportedResultTone({}, paper), 'unmatched');
    }
  }
});

test('legacy findings retain null, mixed and association categories in both languages', () => {
  for (const [direction, english, chinese] of [
    ['no-difference', 'No significant difference', '无显著差异'],
    ['mixed', 'Mixed findings', '混合结果'],
    ['association', 'Association', '相关性'],
    ['unspecified', 'Not reported', '未报告'],
  ]) {
    for (const language of ['en', 'zh']) {
      const context = fixture(language);
      context.detectPaperDirection = () => direction;
      assert.equal(context.paperReportedResult({}, {}), language === 'zh' ? chinese : english);
      assert.equal(context.paperReportedResultTone({}, {}), 'unmatched');
    }
  }
});

test('directional hypotheses retain agreement and opposite-result colors', () => {
  const context = fixture('en');
  for (const direction of ['increase', 'decrease']) {
    context.candidateDirection = () => direction;
    for (const result of ['increase', 'decrease']) {
      const paper = {kgMatchFacets: {direction: result}};
      assert.equal(context.paperReportedResultTone({}, paper), direction === result ? 'exact' : 'opposite');
    }
  }
});

test('finding color does not change scope matching or manufacture overall support', () => {
  const context = fixture('en');
  const analysis = context.paperRelationAnalysis({}, {kgMatchFacets: {direction: 'decrease'}});
  assert.equal(analysis.reportedResultTone, 'exact');
  assert.equal(analysis.diseaseRelation, 'none');
  assert.equal(analysis.regionRelation, 'none');
  assert.equal(analysis.featureRelation, 'none');
  context.detectPaperDirection = () => 'decrease';
  assert.equal(context.paperReportedResultTone({}, {}), 'exact');
});

test('all inline study scripts parse', () => {
  for (const match of source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
    new vm.Script(match[1]);
  }
});
