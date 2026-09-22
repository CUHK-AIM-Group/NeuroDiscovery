"""Versioned expert pilot; preserve historical two-stage sessions unchanged.

The standalone server is deliberately loopback-only. The main application keeps
its existing study authentication in addition to the per-session secret below.
"""
import argparse
import copy
import getpass
import hashlib
import json
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse

from core.web.discovery_scoring import score_session, applicable_item_ids
from core.web.discovery_localization import presentation

HERE = Path(__file__).resolve().parent
# Installers contain only the participant projection, not the author's private
# source mapping. Its separate ID preserves existing author-preview sessions.
PACKAGED_PACK = HERE / "study_materials" / "cs1_discovery_pilot_v16_desktop_v1.json"
DEFAULT_PACK = PACKAGED_PACK if PACKAGED_PACK.is_file() else HERE / "study_materials" / "cs1_discovery_pilot_v16.json"
DEFAULT_DATA = Path.home() / ".neurodiscovery" / "discovery-expert-study"
ASSIGNMENTS_NAMES = ("cs1_discovery_assignments_v10.json", "cs1_discovery_assignments_v9.json", "cs1_discovery_assignments_v8.json", "cs1_discovery_assignments_v7.json", "cs1_discovery_assignments_v6.json", "cs1_discovery_assignments_v5.json", "cs1_discovery_assignments_v4.json", "cs1_discovery_assignments_v3.json", "cs1_discovery_assignments_v2.json", "cs1_discovery_assignments_v1.json")
REFERENCE_NOTES_NAMES = ("cs1_discovery_reference_notes_v9.json", "cs1_discovery_reference_notes_v8.json", "cs1_discovery_reference_notes_v7.json", "cs1_discovery_reference_notes_v6.json", "cs1_discovery_reference_notes_v5.json", "cs1_discovery_reference_notes_v4.json", "cs1_discovery_reference_notes_v3.json", "cs1_discovery_reference_notes_v2.json", "cs1_discovery_reference_notes_v1.json")
SIGNIFICANCE_NAMES = ("cs1_discovery_significance_v9.json", "cs1_discovery_significance_v8.json", "cs1_discovery_significance_v7.json", "cs1_discovery_significance_v6.json", "cs1_discovery_significance_v5.json", "cs1_discovery_significance_v4.json", "cs1_discovery_significance_v3.json", "cs1_discovery_significance_v2.json", "cs1_discovery_significance_v1.json")
# Backward-compatible names used by v6 tooling and tests that stage a single
# legacy sidecar next to an explicitly selected historical pack.
ASSIGNMENTS_NAME = ASSIGNMENTS_NAMES[-1]
REFERENCE_NOTES_NAME = REFERENCE_NOTES_NAMES[-1]
SIGNIFICANCE_NAME = SIGNIFICANCE_NAMES[-1]
ASSIGNMENTS_NAMES = ("cs1_discovery_assignments_v11.json",) + ASSIGNMENTS_NAMES
REFERENCE_NOTES_NAMES = ("cs1_discovery_reference_notes_v10.json",) + REFERENCE_NOTES_NAMES
SIGNIFICANCE_NAMES = ("cs1_discovery_significance_v10.json",) + SIGNIFICANCE_NAMES
ASSIGNMENTS_NAMES = ("cs1_discovery_assignments_v12.json",) + ASSIGNMENTS_NAMES
REFERENCE_NOTES_NAMES = ("cs1_discovery_reference_notes_v11.json",) + REFERENCE_NOTES_NAMES
SIGNIFICANCE_NAMES = ("cs1_discovery_significance_v11.json",) + SIGNIFICANCE_NAMES
PREVIEW_ALL = "ALL"
SPECIALTIES = ["神经影像", "临床神经科学/精神病学", "计算神经科学", "AI4S/机器学习", "统计/研究方法", "影像遗传学", "其他交叉学科"]
EXPOSURES = ["no", "yes", "unsure"]


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _matching_sidecar(pack, pack_path, names):
    """Return the newest sidecar naming this pack, skipping other versions."""
    distribution = pack.get("distribution") or {}
    pack_ids = {pack["pack_id"], distribution.get("author_pack_id")}
    for name in names:
        candidate = Path(pack_path).resolve().parent / name
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if payload.get("pack_id") in pack_ids:
            return candidate, payload
    return None, None


def load_assignments(pack, pack_path):
    """Optional expert assignment table, bound to the exact pack it deals from.

    A table naming another pack is simply ignored so older packs keep working;
    a table naming this pack but failing its hash or content checks is a real
    packaging error and raises.
    """
    candidate, table = _matching_sidecar(pack, pack_path, ASSIGNMENTS_NAMES)
    if candidate is None:
        return None
    distribution = pack.get("distribution") or {}
    hashes = {hashlib.sha256(Path(pack_path).read_bytes()).hexdigest(), distribution.get("author_pack_sha256")}
    if table.get("version") not in {1, 2} or table.get("pack_sha256") not in hashes:
        raise ValueError("The assignment table is not bound to this exact material pack.")
    cards = {card["id"] for card in pack["cards"]}
    experts = table.get("experts") or {}
    if sorted(experts) != [f"P{index:02d}" for index in range(1, 11)]:
        raise ValueError("The assignment table must deal to exactly P01–P10.")
    coverage = {card: 0 for card in cards}
    for expert, dealt in experts.items():
        if not dealt or len(dealt) != len(set(dealt)) or any(card not in cards for card in dealt):
            raise ValueError(f"Assignment {expert} contains unknown or duplicated cards.")
        for card in dealt:
            coverage[card] += 1
    if any(count == 0 for count in coverage.values()):
        raise ValueError("The assignment table leaves cards without any reviewer.")
    declared = table.get("coverage") or {}
    if declared and any(declared.get(card) != count for card, count in coverage.items()):
        raise ValueError("Assignment coverage declaration does not match the dealt cards.")
    return table


def load_reference_notes(pack, pack_path):
    """Optional per-reference display notes, bound to the exact pack hash.

    A notes file naming another pack is ignored so older packs keep working; a
    file naming this pack but failing hash or content checks is a packaging
    error and raises.
    """
    candidate, notes = _matching_sidecar(pack, pack_path, REFERENCE_NOTES_NAMES)
    if candidate is None:
        return None
    distribution = pack.get("distribution") or {}
    hashes = {hashlib.sha256(Path(pack_path).read_bytes()).hexdigest(), distribution.get("author_pack_sha256")}
    if notes.get("version") not in {1, 2} or notes.get("pack_sha256") not in hashes:
        raise ValueError("The reference notes are not bound to this exact material pack.")
    table = notes.get("notes") or {}
    if set(table) - {card["id"] for card in pack["cards"]}:
        raise ValueError("Reference notes contain cards outside the pack.")
    for card in pack["cards"]:
        card_notes = table.get(card["id"]) or {}
        if set(card_notes) != {ref["id"] for ref in card["pre"]["references"]}:
            raise ValueError(f"Reference notes do not match the references of {card['id']}.")
        for note in card_notes.values():
            if set(note) != {"did_zh", "did_en", "relation_zh", "relation_en"}:
                raise ValueError("Reference note lacks its bilingual content fields.")
    definition_plain = notes.get("definition_plain") or {}
    if set(definition_plain) - {card["id"] for card in pack["cards"]}:
        raise ValueError("Definition glosses contain cards outside the pack.")
    for gloss in definition_plain.values():
        if set(gloss) != {"zh", "en"}:
            raise ValueError("Definition gloss lacks its bilingual content fields.")
    return {"references": table, "definition_plain": definition_plain}


def load_significance(pack, pack_path):
    """Optional per-card significance sentences, bound to the exact pack hash.

    Same binding rules as the reference notes: a file naming another pack is
    ignored so older packs keep working; a file naming this pack but failing
    hash or content checks is a packaging error and raises.
    """
    candidate, payload = _matching_sidecar(pack, pack_path, SIGNIFICANCE_NAMES)
    if candidate is None:
        return None
    distribution = pack.get("distribution") or {}
    hashes = {hashlib.sha256(Path(pack_path).read_bytes()).hexdigest(), distribution.get("author_pack_sha256")}
    if payload.get("version") not in {1, 2} or payload.get("pack_sha256") not in hashes:
        raise ValueError("The significance notes are not bound to this exact material pack.")
    table = payload.get("significance") or {}
    if set(table) != {card["id"] for card in pack["cards"]}:
        raise ValueError("Significance notes must cover exactly the pack cards.")
    for sentence in table.values():
        if set(sentence) != {"zh", "en"} or not all(sentence.values()):
            raise ValueError("Significance note lacks its bilingual content fields.")
    return table


ALIGNED_EXPORT_VERSION = "neurodiscovery-expert-study-results-v2"
# Mirrors the client review order in static/discovery-study.js (single-round).
REVIEW_DISPLAY_ORDER = ["novelty", "grounding", "feedback", "design", "validation", "value"]
UNABLE_VALUES = {"unable", "insufficient", "outside", "not_assessable"}


def _fmt_duration(seconds):
    total = max(0, int(round(float(seconds or 0))))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def export_filename(result):
    """Same naming convention as the Expert Study (Extension) result export."""
    code = re.sub(r"[^\w.·-]+", "-", str(result["participant"]["id"]), flags=re.UNICODE).strip("-")[:60] or "participant"
    stamp = str(result["exported_at"]).replace(":", "-").replace(".", "-")
    return f"NeuroDiscovery-user-study-{code}-{stamp}.json"


def _note_present(value):
    if isinstance(value, dict):
        return any(str(v).strip() for v in value.values())
    return bool(str(value or "").strip())


def aligned_envelope(state, pack):
    """Expert Study (Extension)-style export envelope for the discovery review.

    Question rows follow the client's single-round display order; answers stay
    keyed by question id. Cards follow the session's own (shuffled) order, with
    card_id/question_id preserved for cross-expert alignment.
    """
    single_round = pack["public_meta"].get("review_flow") == "single_round"
    stage = state["stage"] if state["stage"] != "complete" else ("review" if single_round else "B")
    questions = [q for q in pack["questions"] if q["stage"] == stage]
    if state["stage"] == "complete" and not single_round:
        questions = list(pack["questions"])
    if single_round:
        questions = sorted(questions, key=lambda q: REVIEW_DISPLAY_ORDER.index(q["id"])
                           if q["id"] in REVIEW_DISPLAY_ORDER else len(REVIEW_DISPLAY_ORDER))
    cards_by_id = {card["id"]: card for card in pack["cards"]}
    results = []
    cannot = {}
    for position, card_id in enumerate(state["order"], 1):
        card = cards_by_id[card_id]
        answers = state["answers"].get(card_id, {})
        for question in questions:
            if question["id"] not in applicable_item_ids(pack, card_id):
                continue
            value = answers.get(question["id"])
            option = next((o for o in question["options"] if o["value"] == value), None)
            is_unable = value in UNABLE_VALUES
            if is_unable:
                cannot[value] = cannot.get(value, 0) + 1
            results.append({
                "question_number": len(results) + 1,
                "card_id": card_id,
                "card_position": position,
                "card_title": card["pre"]["title"],
                "question_id": question["id"],
                "capability": question.get("capability"),
                "title": question["title"],
                "answer": value,
                "answer_label": option["label"] if option else None,
                "cannot_judge": is_unable,
            })
    active_seconds = sum(float(v or 0) for v in state.get("active_seconds_client_reported", {}).values())
    notes = state.get("notes", {})
    issues = state.get("issues", {})
    return {
        "study": {"study_id": pack["pack_id"], "case_study": "multitopic_discovery_complete_output_review" if pack["public_meta"].get("material_layout") == "multitopic" else "case1_discovery_complete_output_review",
                  "protocol_version": pack["protocol_version"]},
        "participant": {"id": state["profile"].get("code"), "experience_years": state["profile"].get("experience"),
                        **({"consent": True} if state["profile"].get("consent") is True else {})},
        "timing": {"session_created_at": state.get("created"), "completed_at": state.get("completed_at"),
                   "exported_at": now(), "active_answering_time": _fmt_duration(active_seconds),
                   "active_answering_time_ms": int(round(active_seconds * 1000))},
        "summary": {
            "answered_questions": sum(1 for r in results if r["answer"] is not None),
            "total_questions": len(results),
            "cards_fully_answered": sum(1 for cid in state["order"]
                                        if all(q["id"] in state["answers"].get(cid, {}) for q in questions
                                               if q["id"] in applicable_item_ids(pack, cid))),
            "cannot_judge": sum(cannot.values()),
            "cannot_judge_reasons": cannot,
            "notes_present": sum(1 for value in notes.values() if _note_present(value)),
            "issue_tag_counts": {tag: sum(1 for tags in issues.values() if tag in tags)
                                 for tag in pack.get("issues", [])},
        },
        "question_results": results,
    }


class StudyError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class DiscoveryStudy:
    def __init__(self, pack_path=DEFAULT_PACK, data_dir=DEFAULT_DATA, record_kind="pilot"):
        self.pack = json.loads(Path(pack_path).read_text(encoding="utf-8"))
        cards = self.pack.get("cards", [])
        if self.pack.get("status") != "pilot_only" or not 1 <= len(cards) <= 40 or len({c["id"] for c in cards}) != len(cards):
            raise ValueError("This pilot requires 1–40 distinct reviewed material cards.")
        if self.pack["public_meta"]["card_count"] != len(cards):
            raise ValueError("Material count differs from the manifest.")
        if record_kind not in {"pilot", "test"}:
            raise ValueError("Invalid server-side record kind.")
        self.record_kind = record_kind
        self.assignments = load_assignments(self.pack, pack_path)
        self.reference_notes = load_reference_notes(self.pack, pack_path)
        self.significance = load_significance(self.pack, pack_path)
        self.pack_hash = digest(encoded(self.pack))
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "discovery_study.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS packs (
                    id TEXT PRIMARY KEY, hash TEXT NOT NULL, snapshot TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, secret_hash TEXT NOT NULL, pack_id TEXT NOT NULL,
                    state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL, kind TEXT NOT NULL, timestamp TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(session_id, request_id));
            """)
            old = db.execute("SELECT hash FROM packs WHERE id=?", (self.pack["pack_id"],)).fetchone()
            if old and old["hash"] != self.pack_hash:
                raise ValueError("The pack ID already exists with different content; use a new version.")
            db.execute("INSERT OR IGNORE INTO packs VALUES (?,?,?)", (self.pack["pack_id"], self.pack_hash, encoded(self.pack)))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def config(self):
        from core.web.evaluation_export import WORKFLOW_REVISION
        # Explicit allowlist: neither candidate outcome nor organiser data belongs here.
        result = {"pack_id": self.pack["pack_id"], "protocol_version": self.pack["protocol_version"],
                "pack_hash": self.pack_hash, "meta": copy.deepcopy(self.pack["public_meta"]),
                "questions": copy.deepcopy(self.pack["questions"]),
                "issues": self.pack["issues"], "scoring": copy.deepcopy(self.pack.get("scoring")),
                # The entry defaults to this machine's account name, shared with
                # the Expert Study (Extension) entry through the client.
                "default_code": getpass.getuser()}
        result["presentation"] = presentation(result)
        result["evaluation_workflow_revision"] = WORKFLOW_REVISION
        result["experimental_results_revision"] = "visible-results-v3"
        if self.reference_notes:
            result["reference_notes"] = copy.deepcopy(self.reference_notes["references"])
            if self.reference_notes["definition_plain"]:
                result["definition_plain"] = copy.deepcopy(self.reference_notes["definition_plain"])
        if self.significance:
            result["significance"] = copy.deepcopy(self.significance)
        if self.assignments:
            result["assignments"] = {
                "version": self.assignments["version"],
                "allocation_revision": self.assignments.get("allocation_revision", ""),
                "rule_zh": self.assignments["rule_zh"],
                "options": [{"id": PREVIEW_ALL, "card_count": len(self.pack["cards"])}] + [
                    {"id": expert, "card_count": len(cards)}
                    for expert, cards in self.assignments["experts"].items()],
            }
        return result

    @staticmethod
    def validate_profile(profile, legacy=False):
        fields = {"code", "experience"}
        if legacy:
            fields |= {"specialty", "old_study", "case_exposure", "mode", "consent"}
            valid_keys = fields
        else:
            # consent is no longer collected at the entry; old clients may still send it.
            valid_keys = fields | {"consent"}
        if not isinstance(profile, dict) or not fields <= set(profile) or not set(profile) <= valid_keys:
            raise StudyError("请完整填写评审信息。")
        if not re.fullmatch(r"[A-Za-z0-9_.·\-一-鿿 ]{2,40}", str(profile["code"])):
            raise StudyError("请使用 2–40 位的名字或代号（字母、数字或中文），不填写真实联系方式。")
        if legacy and profile["specialty"] not in SPECIALTIES:
            raise StudyError("请填写专业领域。")
        if profile["experience"] not in ["0-2", "3-5", "6-10", "10+"]:
            raise StudyError("请填写研究年限。")
        if legacy and (profile["old_study"] not in EXPOSURES or profile["case_exposure"] not in EXPOSURES):
            raise StudyError("请填写既往问卷和材料接触情况。")
        if legacy and (profile["mode"] not in ["pilot", "test"] or profile["consent"] is not True):
            raise StudyError("请确认已阅读数据保存说明并自愿参与评审。")

    def create(self, profile):
        legacy = not bool(self.pack.get("scoring"))
        single_round = self.pack["public_meta"].get("review_flow") == "single_round"
        stages = ["review"] if single_round else ["A", "B"]
        profile = dict(profile) if isinstance(profile, dict) else profile
        assignment = str(profile.pop("assignment_id", "") or "").strip().upper() if isinstance(profile, dict) else ""
        self.validate_profile(profile, legacy=legacy)
        cards = self.pack["cards"]
        if assignment and assignment != PREVIEW_ALL:
            if not self.assignments or assignment not in self.assignments["experts"]:
                raise StudyError("未知的分配编号，请按入口列表选择。")
            wanted = set(self.assignments["experts"][assignment])
            cards = [card for card in cards if card["id"] in wanted]
        session_id, token = secrets.token_urlsafe(18), secrets.token_urlsafe(32)
        order = [card["id"] for card in cards]
        secrets.SystemRandom().shuffle(order)
        state = {"id": session_id, "pack_id": self.pack["pack_id"], "pack_hash": self.pack_hash,
                 "protocol_version": self.pack["protocol_version"], "profile": profile,
                 "created": now(), "updated": now(), "stage": stages[0], "revision": 0,
                 "assignment_id": assignment or PREVIEW_ALL,
                 "order": order, "answers": {key: {} for key in order}, "notes": {key: dict.fromkeys(stages, "") for key in order},
                 "issues": {key: [] for key in order}, "active_seconds_client_reported": dict.fromkeys(stages, 0),
                 "stage_a_locked_at": None, "completed_at": None}
        if self.assignments and assignment and assignment != PREVIEW_ALL:
            state["allocation_revision"] = self.assignments.get("allocation_revision", "")
        if not legacy:
            # Test isolation is a server deployment setting, not an unasked
            # participant answer. Never infer previous exposure as "no".
            state["record_kind"] = self.record_kind
            state["profile_schema_version"] = "anonymous-experience-v4"
        if single_round:
            state.pop("stage_a_locked_at")
            state["review_flow"] = "single_round"
        event_profile = {**profile, "assignment_id": state["assignment_id"]}
        with self.connection() as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (session_id, digest(token), state["pack_id"], encoded(state)))
            db.execute("INSERT INTO events(session_id,request_id,kind,timestamp,payload_hash,payload) VALUES (?,?,?,?,?,?)",
                       (session_id, "created", "create", now(), digest(encoded(event_profile)), encoded(event_profile)))
        return {**self.public(state, self.pack), "session_token": token}

    @staticmethod
    def public(state, pack):
        cards_by_id = {card["id"]: card for card in pack["cards"]}
        cards = []
        for key in state["order"]:
            card = cards_by_id[key]
            material = {"id": key, "pre": copy.deepcopy(card["pre"])}
            if state["stage"] in ["review", "B", "complete"]:
                material["post"] = copy.deepcopy(card["post"])
            cards.append(material)
        result = {"session": copy.deepcopy(state), "meta": copy.deepcopy(pack["public_meta"]),
                  "questions": copy.deepcopy(pack["questions"]), "issue_options": pack["issues"],
                  "common_pre": copy.deepcopy(pack["common_pre"]), "cards": cards,
                  "scoring": copy.deepcopy(pack.get("scoring"))}
        if state["stage"] in ["review", "B", "complete"]:
            result["common_post"] = copy.deepcopy(pack["common_post"])
        if state["stage"] == "complete" and pack.get("scoring"):
            result["score_summary"] = score_session(state, pack)
        result["presentation"] = presentation(result)
        return result

    def load(self, db, session_id, token):
        row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row or not token or not secrets.compare_digest(row["secret_hash"], digest(token)):
            raise StudyError("无法访问此会话，请使用原浏览器或保存的继续码。", 403)
        state = json.loads(row["state"])
        pack = json.loads(db.execute("SELECT snapshot FROM packs WHERE id=?", (row["pack_id"],)).fetchone()["snapshot"])
        return state, pack

    def get(self, session_id, token):
        with self.connection() as db:
            state, pack = self.load(db, session_id, token)
            return self.public(state, pack)

    def discard(self, session_id, token):
        """Delete one unfinished session after an explicit participant choice.

        Completed reviews are immutable study records, mirroring the ranking
        study's delete_active_session rule.
        """
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            state, _pack = self.load(db, session_id, token)
            if state["stage"] == "complete":
                raise StudyError("已完成的评审记录不可删除。", 409)
            db.execute("DELETE FROM events WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return {"deleted": True, "id": session_id}

    @staticmethod
    def missing(state, pack, stage):
        required = [q["id"] for q in pack["questions"] if q["stage"] == stage]
        return [(card, key) for card in state["order"] for key in required
                if key in applicable_item_ids(pack, card) and key not in state["answers"][card]]

    def mutate(self, session_id, token, kind, payload):
        if kind not in ["save", "reveal", "submit"] or not isinstance(payload, dict):
            raise StudyError("不支持的请求。")
        request_id = payload.get("request_id", "")
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", request_id):
            raise StudyError("缺少请求标识。")
        payload_hash = digest(encoded({"kind": kind, "payload": payload}))
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            state, pack = self.load(db, session_id, token)
            old = db.execute("SELECT payload_hash FROM events WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
            if old:
                if old["payload_hash"] != payload_hash:
                    raise StudyError("重复请求标识的内容不一致。", 409)
                return self.public(state, pack)
            if type(payload.get("revision")) is not int or payload["revision"] != state["revision"]:
                raise StudyError("另一标签页已更新答案。请刷新后继续，未覆盖已保存答案。", 409)
            if state["stage"] == "complete":
                raise StudyError("已完成的评审不可改写；请导出或开始独立的新会话。", 409)
            if kind == "save":
                allowed_keys = {"request_id", "revision", "card_id", "answers", "note", "issues", "active_seconds", "display"}
                if set(payload) - allowed_keys:
                    raise StudyError("请求包含未定义字段。")
                card_id = payload.get("card_id")
                answers = payload.get("answers", {})
                if card_id not in state["order"] or not isinstance(answers, dict):
                    raise StudyError("材料或答案无效。")
                rubric = {q["id"]: q for q in pack["questions"] if q["stage"] == state["stage"]
                          and q["id"] in applicable_item_ids(pack, card_id)}
                for key, value in answers.items():
                    if key not in rubric or value not in [o["value"] for o in rubric[key]["options"]]:
                        raise StudyError("答案不属于当前阶段，或选项无效。")
                note = payload.get("note", "")
                if not isinstance(note, str) or len(note) > 4000:
                    raise StudyError("备注最多 4,000 字符。")
                issues = payload.get("issues", [])
                if not isinstance(issues, list) or len(issues) != len(set(str(x) for x in issues)) or any(x not in pack["issues"] for x in issues):
                    raise StudyError("问题标签无效。")
                seconds = payload.get("active_seconds", 0)
                if type(seconds) not in (int, float) or not 0 <= seconds <= 28800:
                    raise StudyError("活动时间值无效。")
                display = payload.get("display")
                if display is not None:
                    expected = presentation({})
                    if (not isinstance(display, dict)
                            or set(display) not in ({"language", "version", "sha256"}, {"language", "version", "sha256", "renderer_revision"})
                            or ("renderer_revision" in display and display["renderer_revision"] not in {"visible-results-v1", "visible-results-v2", "visible-results-v3"})
                            or display["language"] not in {"zh", "en"}
                            or display["version"] != expected["version"] or display["sha256"] != expected["sha256"]):
                        raise StudyError("显示版本已更新，请保存继续码后刷新页面。", 409)
                    history = state.setdefault("display_history", [])
                    if not history or history[-1]["display"] != display:
                        history.append({"display": display, "timestamp": now()})
                state["answers"][card_id].update(answers)
                state["notes"][card_id][state["stage"]] = note
                if state["stage"] == "B":
                    state["issues"][card_id] = issues
                state["active_seconds_client_reported"][state["stage"]] += seconds
            elif kind == "reveal":
                if pack["public_meta"].get("review_flow") == "single_round":
                    raise StudyError("本版为单轮评审，完整结果已提供，无需解锁。")
                if state["stage"] != "A" or self.missing(state, pack, "A"):
                    raise StudyError(f"请先完成全部 {len(state['order'])} 份材料的阶段 A，再解锁任何实验结果。")
                state["stage"], state["stage_a_locked_at"] = "B", now()
            elif kind == "submit":
                required_stage = "review" if pack["public_meta"].get("review_flow") == "single_round" else "B"
                if state["stage"] != required_stage or self.missing(state, pack, required_stage):
                    raise StudyError("请完成全部评审问题；确实无法判断时可选择对应选项。")
                for card_id in state["order"] if not pack.get("scoring") else []:
                    answer = state["answers"][card_id]
                    if answer["validity"] == "1" and any(answer[key] != "not_assessable" for key in ["outcome", "evidence", "coverage"]):
                        raise StudyError("若选择“没有形成有效分析结果”，请将该材料的结果、支持度与覆盖问题设为“不作科学支持判断”。")
                    if answer["validity"] == "3" and any(answer[key] == "not_assessable" for key in ["outcome", "evidence", "coverage"]):
                        raise StudyError("“有效分析”与“无有效分析”不一致；如证据不足，请使用“材料不足”或不确定选项。")
                state["stage"], state["completed_at"] = "complete", now()
            state["revision"] += 1
            state["updated"] = now()
            db.execute("UPDATE sessions SET state=? WHERE id=?", (encoded(state), session_id))
            db.execute("INSERT INTO events(session_id,request_id,kind,timestamp,payload_hash,payload) VALUES (?,?,?,?,?,?)",
                       (session_id, request_id, kind, now(), payload_hash, encoded(payload)))
            return self.public(state, pack)

    def export(self, session_id, token):
        with self.connection() as db:
            state, pack = self.load(db, session_id, token)
            events = [{"kind": row["kind"], "timestamp": row["timestamp"], "request_id": row["request_id"]}
                      for row in db.execute("SELECT kind,timestamp,request_id FROM events WHERE session_id=? ORDER BY sequence", (session_id,))]
            interpretation = pack["scoring"]["interpretation"] if pack.get("scoring") else "Human judgments, not verified discoveries or scientific ground truth. No total score is defined."
            base = self.public(state, pack)
            return {"schema_version": ALIGNED_EXPORT_VERSION, "export_schema": 2 if pack.get("scoring") else 1,
                    "exported_at": now(), "interpretation": interpretation,
                    **aligned_envelope(state, pack), **base, "events": events}

    def save_export(self, session_id, token):
        """Explicit export action; durable file even if a webview blocks downloads."""
        result = self.export(session_id, token)
        directory = self.data_dir / "exports"
        directory.mkdir(parents=True, exist_ok=True)
        filename = export_filename(result)
        target = directory / filename
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        return {"path": str(target.resolve()), "filename": filename, "stage": result["session"]["stage"], "bytes": target.stat().st_size}


def register_discovery_routes(app, service=None, ranking_service=None):
    # Material packs are local-only. A missing pack must not prevent the main
    # application or the original study from starting on another installation.
    holder = [service]
    app.state.discovery_study = service

    @app.post("/api/studies/evaluations/export")
    async def evaluations_export(payload: dict = Body(...)):
        from core.web.evaluation_export import build_evaluation_export
        try:
            if holder[0] is None:
                holder[0] = DiscoveryStudy()
                app.state.discovery_study = holder[0]
            result = build_evaluation_export(payload, holder[0], ranking_service)
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except StudyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        except (ValueError, KeyError, FileNotFoundError):
            return JSONResponse({"error": "Unable to export the linked sessions. Check the name/code and saved session references."}, status_code=400)

    def call(method, *args):
        try:
            if holder[0] is None:
                holder[0] = DiscoveryStudy()
                app.state.discovery_study = holder[0]
            return JSONResponse(getattr(holder[0], method)(*args), headers={"Cache-Control": "no-store"})
        except StudyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status, headers={"Cache-Control": "no-store"})
        except FileNotFoundError:
            return JSONResponse({"error": "本机尚未配置该评审材料包，请联系主办方。原有问卷不受影响。"}, status_code=503)

    @app.get("/discovery-study")
    async def discovery_page():
        return FileResponse(HERE / "static" / "discovery-study.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/studies/discovery/config")
    async def discovery_config():
        return call("config")

    @app.post("/api/studies/discovery/sessions")
    async def discovery_create(payload: dict = Body(...)):
        return call("create", payload)

    @app.get("/api/studies/discovery/sessions/{session_id}")
    async def discovery_get(session_id: str, x_discovery_session_token: str = Header(default="")):
        return call("get", session_id, x_discovery_session_token)

    @app.delete("/api/studies/discovery/sessions/{session_id}")
    async def discovery_discard(session_id: str, x_discovery_session_token: str = Header(default="")):
        return call("discard", session_id, x_discovery_session_token)

    @app.post("/api/studies/discovery/sessions/{session_id}/{action}")
    async def discovery_change(session_id: str, action: str, payload: dict = Body(...), x_discovery_session_token: str = Header(default="")):
        if action == "export-file":
            return call("save_export", session_id, x_discovery_session_token)
        return call("mutate", session_id, x_discovery_session_token, action, payload)

    @app.get("/api/studies/discovery/sessions/{session_id}/export")
    async def discovery_export(session_id: str, x_discovery_session_token: str = Header(default="")):
        response = call("export", session_id, x_discovery_session_token)
        if response.status_code == 200:
            payload = json.loads(bytes(response.body))
            response.headers["Content-Disposition"] = f'attachment; filename="{export_filename(payload)}"'
        return response


def create_preview_app(pack_path=DEFAULT_PACK, data_dir=DEFAULT_DATA, record_kind="pilot"):
    app = FastAPI(title="Discovery expert pilot · local preview", docs_url=None, redoc_url=None)
    register_discovery_routes(app, DiscoveryStudy(pack_path, data_dir, record_kind=record_kind))

    @app.get("/static/{name}")
    async def preview_asset(name: str):
        if name not in {"discovery-study.css", "discovery-study.js", "discovery-study-i18n.js",
                        "workspace-tokens.css", "study-workspace.css", "study-workspace.js", "evaluation-export.js"}:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse(HERE / "static" / name, headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "mode": "isolated-pilot-preview"}

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--test-only", action="store_true", help="Mark all new sessions as synthetic UI tests, isolated from expert ratings.")
    args = parser.parse_args()
    uvicorn.run(create_preview_app(args.pack, args.data, "test" if args.test_only else "pilot"), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
