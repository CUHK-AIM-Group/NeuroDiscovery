import json
import shutil
import subprocess
import tempfile
import unittest

from fastapi.testclient import TestClient

from core.web.build_discovery_pilot_v15 import HERE, PACK_ID, ROOT
from core.web.discovery_study import create_preview_app


class RelatedLiteratureIntegrationTests(unittest.TestCase):
    def test_http_config_and_new_session(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_preview_app(HERE / "cs1_discovery_pilot_v15.json", directory, "test")) as client:
                config = client.get("/api/studies/discovery/config").json()
                self.assertEqual(config["pack_id"], PACK_ID)
                self.assertEqual(config["meta"]["related_literature_revision"], "five-reviewed-references-v1")
                response = client.post("/api/studies/discovery/sessions", json={
                    "code": "TEST-V15-HTTP", "experience": "3-5", "assignment_id": "P01"})
                self.assertEqual(response.status_code, 200, response.text)
                view = response.json()
                self.assertTrue(all(len(card["pre"]["references"]) == 5 for card in view["cards"]))
                self.assertNotIn("organizer", view)

    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_actual_renderer_and_desktop_compatibility(self):
        program = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const pack = JSON.parse(fs.readFileSync('core/web/study_materials/cs1_discovery_pilot_v15.json'));
const source = fs.readFileSync('core/web/static/discovery-study.js', 'utf8');
const renderer = source.slice(source.indexOf('function readableMaterialHTML('), source.indexOf('function renderMaterial('));
const context = {display:{language:'zh'}, current:0, data:pack, h:String, safeUrl:String,
 terminologyHTML:()=>'', partHTML:()=>'', significanceHTML:()=>'', researchChainHTML:()=>'', readableResultTablesHTML:()=>''};
vm.createContext(context); vm.runInContext(renderer, context);
for (const language of ['zh', 'en']) {
 context.display.language = language;
 for (const card of pack.cards) {
  const html = context.readableMaterialHTML(card);
  assert.equal((html.match(/class="related-study"/g)||[]).length,5);
  assert.equal((html.match(/class="rationale-refs"/g)||[]).length,1);
  assert.ok(html.includes(language==='en'?'Related studies (5)':'相关研究 (5)'));
  for (const reference of card.pre.references) assert.ok(html.includes(`PMID ${reference.pmid}`));
 }
}
const main = fs.readFileSync('desktop/main.js', 'utf8');
const check = main.slice(main.indexOf('function requestMultiTopicStudy('),main.indexOf('async function requestDesktopCompatible('));
const payload = {meta:pack.public_meta, evaluation_workflow_revision:'unified-evaluation-export-v1',
 assignments:{allocation_revision:'he1-topic40-success-led-v7'}, experimental_results_revision:'visible-results-v3', scoring:pack.scoring};
const http = {get:(_url,_options,callback)=>{
 const request = {on:()=>{},destroy:()=>{}};
 queueMicrotask(()=>callback({statusCode:200,on:(event,handler)=>{
  if(event==='data') handler(JSON.stringify(payload)); if(event==='end') handler();
 }})); return request;
}};
const transport = {http}; vm.createContext(transport); vm.runInContext(check,transport);
(async()=>{
 assert.equal(await transport.requestMultiTopicStudy('http://fixture'),false);
 delete payload.meta.related_literature_revision;
 assert.equal(await transport.requestMultiTopicStudy('http://fixture'),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result = subprocess.run([shutil.which("node"), "-e", program], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
