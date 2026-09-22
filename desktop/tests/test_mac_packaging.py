import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "desktop/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build-mac-arm64")
stager = load("stage-mac-variants")


class MacPackagingTests(unittest.TestCase):
    def test_separate_arm64_configs(self):
        configs = [builder.builder_config(variant, f"/{variant}/runtime", f"/{variant}/artifacts")
                   for variant in stager.VARIANTS]
        self.assertEqual(len({config["appId"] for config in configs}), 3)
        self.assertEqual(len({config["directories"]["output"] for config in configs}), 3)
        for config in configs:
            self.assertIsNone(config["extends"])
            self.assertIsNone(config["mac"]["identity"])
            self.assertEqual(config["mac"]["target"], [
                {"target": "dmg", "arch": ["arm64"]}, {"target": "zip", "arch": ["arm64"]}])
        self.assertEqual(configs[1]["files"], ["evaluation-main.js", "evaluation-preload.js", "package.json"])
        self.assertEqual(configs[1]["extraResources"][0]["to"], "evaluation-runtime")
        self.assertEqual(configs[2]["extraMetadata"]["distribution"], "demo")

    def test_native_guard(self):
        with patch.object(builder.sys, "version_info", (3, 12)), patch.object(builder.sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                builder.native_guard()
        with patch.object(builder.sys, "version_info", (3, 12)), patch.object(builder.sys, "platform", "darwin"), \
                patch.object(builder.platform, "machine", return_value="x86_64"):
            with self.assertRaisesRegex(RuntimeError, "Rosetta"):
                builder.native_guard()

    def test_target_protection(self):
        for target in (ROOT, ROOT.parent, ROOT / "core/new", ROOT / "desktop/runtime-evaluation/new",
                       ROOT / "desktop/runtime-demo/new", ROOT / "desktop/scripts/new"):
            with self.assertRaises(ValueError):
                stager.fresh_target(ROOT, target)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                stager.fresh_target(ROOT, directory)

    def test_copy_excludes_private_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            for relative in ("core/public.py", "core/.env", "core/key.pem", "core/a.sqlite3",
                             "core/weights.pth", "core/.frozen/private.json", "core/tests/test.py",
                             "core/web/static/study.html", "core/web/discovery_study.py"):
                original = source / relative
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_text("fixture", encoding="utf-8")
            stager.copy_source_tree(source, target, "core", "demo")
            self.assertEqual([path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()],
                             ["core/public.py"])

    def test_safe_archive(self):
        for name, link in (("bin/python", None), ("../escape", None), ("link", "../../escape")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "python.tar"
                with tarfile.open(archive, "w") as stream:
                    member = tarfile.TarInfo(name)
                    if link:
                        member.type, member.linkname = tarfile.SYMTYPE, link
                        stream.addfile(member)
                    else:
                        member.size = 4
                        stream.addfile(member, io.BytesIO(b"test"))
                target = Path(directory) / "unpack"
                if name == "bin/python":
                    builder.safe_extract(archive, target)
                    self.assertEqual((target / name).read_bytes(), b"test")
                else:
                    with self.assertRaises(ValueError):
                        builder.safe_extract(archive, target)
                self.assertFalse((Path(directory) / "escape").exists())

    def test_actual_staged_backends(self):
        for variant in stager.VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                backend = Path(directory) / "backend"
                inventory = stager.stage(ROOT, backend, variant)
                if variant == "evaluation":
                    self.assertEqual(len(inventory), 29)
                elif variant == "demo":
                    self.assertFalse(any("study_materials/" in name or "/user_study/" in name for name in inventory))
                    self.assertIn("const HUMAN_EVALUATION_ENABLED = false;",
                                  (backend / "core/web/static/index.html").read_text(encoding="utf-8"))
                else:
                    pack = json.loads(next((backend / "core/web/study_materials").glob("*desktop_v1.json")).read_text(encoding="utf-8"))
                    self.assertNotIn("organizer", pack)
                    self.assertEqual(len(pack["cards"]), 10)
                before = set(backend.rglob("*"))
                result = subprocess.run([sys.executable, "-B", ROOT / "desktop/scripts/verify-mac-backend.py",
                                         "--backend", backend, "--variant", variant], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(before, set(backend.rglob("*")))


if __name__ == "__main__":
    unittest.main()
