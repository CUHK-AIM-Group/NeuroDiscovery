"""Approved PubMed search with bounded dependency recovery and stdlib fallback."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET


PUBMED_TOOL_NAME = "search_pubmed"
PUBMED_TOOL = {"type": "function", "function": {
    "name": PUBMED_TOOL_NAME,
    "description": "Search PubMed and save source-linked JSON. If Bio is missing, install biopython once into "
                   "task-local dependencies, verify import with the SAME active Python, then retry. If installation "
                   "fails, use stdlib HTTP. This tool requires normal execution approval (network, install, output write). "
                   "Use it after missing-biopython errors instead of directory probes. Never use runtime receipts as output.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "output": {"type": "string"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
    }, "required": ["query", "output"]},
}}


def _process(arguments, env, cancel, timeout=90):
    if cancel.is_set():
        raise RuntimeError("cancelled")
    bootstrap = (
        "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); mode=sys.argv.pop(1); "
        "target=sys.argv.pop(1); sys.argv[0]=target; "
        "exec(target) if mode=='-c' else (runpy.run_module(target,run_name='__main__',alter_sys=True) "
        "if mode=='-m' else runpy.run_path(target,run_name='__main__'))"
    )
    mode, target, *rest = arguments if arguments[0] in {"-c", "-m"} else ["script", *arguments]
    command = [sys.executable, "-c", bootstrap, env.get("PYTHONPATH", ""), mode, target, *rest]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0) as process:
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set() or time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                raise RuntimeError("cancelled" if cancel.is_set() else "dependency_timeout")
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                return process.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                continue


def pubmed_http(query, limit, cancel):
    def fetch(endpoint, parameters):
        if cancel.is_set():
            raise RuntimeError("cancelled")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" + endpoint + "?" + urlencode(parameters)
        with urlopen(url, timeout=25) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("PubMed response exceeds bounded size")
        if cancel.is_set():
            raise RuntimeError("cancelled")
        return data

    payload = json.loads(fetch("esearch.fcgi", {"db": "pubmed", "term": query, "retmax": limit, "retmode": "json"}))
    search = payload.get("esearchresult")
    if payload.get("error") or not isinstance(search, dict) or search.get("ERROR") or search.get("errorlist"):
        raise ValueError("PubMed search returned an error or invalid response")
    identifiers = search.get("idlist")
    if not isinstance(identifiers, list) or any(not str(identifier).isdigit() for identifier in identifiers):
        raise ValueError("Invalid PubMed identifiers")
    if not identifiers:
        return []
    root = ET.fromstring(fetch("efetch.fcgi", {"db": "pubmed", "id": ",".join(identifiers[:limit]), "retmode": "xml"}))
    if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
        raise ValueError("PubMed fetch returned an invalid response")
    papers = []
    for article in root.findall("PubmedArticle"):
        pmid = article.findtext("MedlineCitation/PMID", "")
        title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
        if not pmid or not title:
            raise ValueError("PubMed article lacks PMID or title")
        papers.append({"pmid": pmid, "title": title,
                       "abstract": "\n".join("".join(part.itertext()) for part in article.findall(".//AbstractText")),
                       "doi": next((part.text for part in article.findall(".//ArticleId") if part.get("IdType") == "doi"), ""),
                       "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    if not papers:
        raise ValueError("PubMed fetch returned no articles for nonempty search IDs")
    return papers


def search_with_recovery(workspace, dependency_dir, arguments, cancel, recovery_state):
    steps = []
    try:
        query, output = arguments.get("query"), arguments.get("output")
        limit = arguments.get("max_results", 20)
        if not isinstance(query, str) or not query.strip() or len(query) > 2000 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Use a nonempty query and max_results between 1 and 50")
        if not isinstance(output, str) or not output:
            raise ValueError("Use a workspace JSON output path")
        workspace = Path(workspace).resolve()
        target = (workspace / output).resolve()
        dependencies = Path(dependency_dir).resolve()
        if not target.is_relative_to(workspace) or not dependencies.is_relative_to(workspace) or target.suffix.lower() != ".json" or ".neurodiscovery" in target.relative_to(workspace).parts:
            raise ValueError("Output must be workspace JSON outside runtime bookkeeping")
        if target.exists():
            raise ValueError("Output exists; use a new path to preserve prior research")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(dependencies))
        verified = False
        try:
            verified = _process(["-c", "from Bio import Entrez; print('import_verified')"], env, cancel)[0] == 0
            if not verified and not recovery_state.get("install_attempted"):
                recovery_state["install_attempted"] = True
                dependencies.mkdir(parents=True, exist_ok=True)
                installed = _process(["-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--retries", "0",
                                      "--timeout", "20", "--only-binary=:all:", "--target", str(dependencies), "biopython"], env, cancel)[0] == 0
                steps.append("install_succeeded" if installed else "install_failed")
                verified = installed and _process(["-c", "from Bio import Entrez; print('import_verified')"], env, cancel)[0] == 0
            steps.append("import_verified" if verified else "import_unavailable")
        except (OSError, RuntimeError):
            if cancel.is_set():
                raise RuntimeError("cancelled")
            steps.append("install_or_import_failed")
        if cancel.is_set():
            raise RuntimeError("cancelled")
        papers = None
        if verified:
            script = Path(__file__).resolve().parents[1] / "skills/academic-research-hub/scripts/research.py"
            try:
                code, stdout, stderr = _process([str(script), "pubmed", query, "--max-results", str(limit), "--format", "json"], env, cancel)
                candidate = json.loads(stdout) if code == 0 else None
                if isinstance(candidate, list) and all(isinstance(row, dict) and row.get("pmid") and row.get("title") for row in candidate):
                    papers = candidate
                    steps.append("retried_same_interpreter")
            except (OSError, ValueError, RuntimeError):
                if cancel.is_set():
                    raise RuntimeError("cancelled")
        if papers is None:
            steps.append("stdlib_http_fallback")
            papers = pubmed_http(query, limit, cancel)
        if cancel.is_set():
            raise RuntimeError("cancelled")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(papers, handle, ensure_ascii=False, indent=2)
        return {"success": True, "path": str(target), "count": len(papers), "recovery_steps": steps,
                "interpreter": sys.executable, "message": f"Saved {len(papers)} PubMed records to {target}",
                "recovery_hint": "Read this JSON with read_workspace_file. Empty results are not proof that related research does not exist."}
    except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
        return {"success": False, "error_type": "cancelled" if cancel.is_set() else "pubmed_recovery_failed",
                "error": str(exc), "recovery_steps": steps,
                "recovery_hint": "Inspect access/query problems or report the blocker; do not replace recovery with directory listings."}
