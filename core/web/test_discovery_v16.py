import copy
import json
import shutil
import subprocess
import tempfile
import unittest

from fastapi.testclient import TestClient

from core.web.build_discovery_pilot_v16 import HERE, ROOT, PARENT, PACK_ID, REVISION, build
from core.web.discovery_study import DEFAULT_PACK, DiscoveryStudy, StudyError, create_preview_app


class RuntimeProcessQuestionnaireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = build()
        cls.pack = json.loads(cls.outputs["cs1_discovery_pilot_v16.json"])

    def test_frozen_parent_results_and_repeatable_release(self):
        self.assertEqual(build(), self.outputs)
        for name, raw in self.outputs.items():
            self.assertEqual((HERE / name).read_bytes(), raw)
        parent = json.loads(PARENT.read_bytes())
        for old, new in zip(parent["cards"], self.pack["cards"]):
            restored = copy.deepcopy(new)
            self.assertEqual(len(restored["pre"]["reading"].pop("runtime_process")), 4)
            self.assertEqual(old, restored)

    def test_questions_anchors_translations_and_five_item_scoring(self):
        self.assertEqual(DEFAULT_PACK.name, "cs1_discovery_pilot_v16.json")
        questions = {question["id"]: question for question in self.pack["questions"]}
        self.assertEqual(set(questions), {"grounding", "novelty", "design", "validation", "value"})
        self.assertEqual(questions["validation"]["title"], "系统是否成功完成了实验完整过程，包括数据预处理、模型选择、实验记录和分析")
        self.assertIn("此前的真实实验反馈", questions["grounding"]["title"])
        translations = json.loads(self.outputs["cs1_discovery_en_v15.json"])["strings"]
        for question in questions.values():
            for text in [question["title"], question["help"], question["capability"], *[option["label"] for option in question["options"]]]:
                self.assertTrue(translations[text])
        for items in self.pack["scoring"]["applicable_items_by_card"].values():
            self.assertEqual(set(items), set(questions))
        self.assertEqual(self.pack["scoring"]["version"], REVISION)

    def test_old_scores_are_not_merged_or_reinterpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            old = DiscoveryStudy(PARENT, directory, record_kind="test")
            created = old.create({"code": "TEST-OLD-SCORES", "experience": "3-5", "assignment_id": "P01"})
            saved = old.mutate(created["session"]["id"], created["session_token"], "save", {
                "request_id": "old-scores", "revision": 0, "card_id": "packet-01",
                "answers": {"grounding": "2", "feedback": "5", "validation": "4"}, "note": "preserve"})
            old_export = old.export(created["session"]["id"], created["session_token"])
            new = DiscoveryStudy(DEFAULT_PACK, directory, record_kind="test")
            restored = new.get(created["session"]["id"], created["session_token"])
            for key in ("session", "cards", "questions", "scoring"):
                self.assertEqual(saved[key], restored[key])
                self.assertEqual(old_export[key], new.export(created["session"]["id"], created["session_token"])[key])

    def test_http_save_submit_export_all_assignments(self):
        with tempfile.TemporaryDirectory() as directory, TestClient(create_preview_app(DEFAULT_PACK, directory, "test")) as client:
            self.assertEqual(client.get("/api/studies/discovery/config").json()["pack_id"], PACK_ID)
            for expert in range(1, 11):
                response = client.post("/api/studies/discovery/sessions", json={
                    "code": f"TEST-V16-{expert}", "experience": "3-5", "assignment_id": f"P{expert:02d}"})
                self.assertEqual(response.status_code, 200, response.text)
                view = response.json()
                endpoint = "/api/studies/discovery/sessions/" + view["session"]["id"]
                headers = {"X-Discovery-Session-Token": view["session_token"]}
                self.assertEqual(len(view["cards"]), 10)
                self.assertNotIn("organizer", view)
                invalid = client.post(endpoint + "/save", headers=headers, json={
                    "request_id": "old-feedback-rejected", "revision": 0, "card_id": view["cards"][0]["id"], "answers": {"feedback": "5"}})
                self.assertEqual(invalid.status_code, 400)
                for index, card in enumerate(view["cards"]):
                    response = client.post(endpoint + "/save", headers=headers, json={
                        "request_id": f"save-{index}", "revision": view["session"]["revision"], "card_id": card["id"],
                        "answers": {question["id"]: "4" for question in view["questions"]}, "note": "SYNTHETIC QA"})
                    self.assertEqual(response.status_code, 200, response.text)
                    view = response.json()
                response = client.post(endpoint + "/submit", headers=headers, json={"request_id": "submit", "revision": view["session"]["revision"]})
                self.assertEqual(response.status_code, 200, response.text)
                view = response.json()
                self.assertEqual(view["score_summary"]["composite_mean"], 4)
                self.assertEqual(len(view["score_summary"]["dimensions"]), 5)
                exported = client.get(endpoint + "/export", headers=headers)
                self.assertEqual(exported.status_code, 200, exported.text)
                exported = exported.json()
                self.assertEqual(exported["scoring"]["version"], REVISION)
                self.assertEqual(exported["questions"], view["questions"])
                self.assertTrue(all(len(card["pre"]["reading"]["runtime_process"]) == 4 for card in exported["cards"]))

    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_actual_renderer_and_desktop_rejects_old_backend(self):
        program = r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const pack=JSON.parse(fs.readFileSync('core/web/study_materials/cs1_discovery_pilot_v16.json'));
const source=fs.readFileSync('core/web/static/discovery-study.js','utf8');
const renderer=source.slice(source.indexOf('function readableMaterialHTML('),source.indexOf('function renderMaterial('));
const context={display:{language:'zh'},current:0,data:pack,h:value=>String(value).replaceAll('<','&lt;'),safeUrl:String,
 terminologyHTML:()=>'',partHTML:()=>'',significanceHTML:()=>'',researchChainHTML:()=>'',readableResultTablesHTML:()=>'<div>TABLES</div>'};
vm.createContext(context);vm.runInContext(renderer,context);
for(const language of ['zh','en']) {
 context.display.language=language;
 for(const card of pack.cards) {
  const html=context.readableMaterialHTML(card);
  assert.equal((html.match(/data-runtime-process/g)||[]).length,1);
  for(const step of card.pre.reading.runtime_process) assert.ok(html.includes(step.text[language]));
  assert.ok(html.indexOf('data-runtime-process')<html.indexOf('TABLES'));
  const historical=structuredClone(card);delete historical.pre.reading.runtime_process;
  assert.ok(!context.readableMaterialHTML(historical).includes('data-runtime-process'));
 }
}
const malicious=structuredClone(pack.cards[0]);malicious.pre.reading.runtime_process[0].text.zh='<script>bad</script>';
context.display.language='zh';assert.ok(!context.readableMaterialHTML(malicious).includes('<script>'));
const main=fs.readFileSync('desktop/main.js','utf8');
const check=main.slice(main.indexOf('function requestMultiTopicStudy('),main.indexOf('async function requestDesktopCompatible('));
const payload={meta:structuredClone(pack.public_meta),evaluation_workflow_revision:'unified-evaluation-export-v1',
 assignments:{allocation_revision:'he1-topic40-success-led-v7'},experimental_results_revision:'visible-results-v3',scoring:pack.scoring};
const http={get:(_url,_options,callback)=>{const request={on:()=>{},destroy:()=>{}};
 queueMicrotask(()=>callback({statusCode:200,on:(event,handler)=>{if(event==='data')handler(JSON.stringify(payload));if(event==='end')handler();}}));return request;}};
const transport={http};vm.createContext(transport);vm.runInContext(check,transport);
(async()=>{assert.equal(await transport.requestMultiTopicStudy('http://fixture'),true);
 delete payload.meta.questionnaire_revision;assert.equal(await transport.requestMultiTopicStudy('http://fixture'),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result = subprocess.run([shutil.which("node"), "-e", program], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
