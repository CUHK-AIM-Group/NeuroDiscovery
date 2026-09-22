"""Build the native Windows WebView2 evaluation shell from a pinned SDK."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

DESKTOP = Path(__file__).resolve().parents[1]
NAME = "NeuroDiscovery-Human-Evaluation-1.0.0-WebView-win-x64"
SDK_VERSION = "1.0.4191.47"
SDK_HASH = "f492bbf547d0da329553b6727435b677579b1e9f91cc9e4a1ad029366d5f23d0"
spec = importlib.util.spec_from_file_location("browser_packager", DESKTOP / "scripts/package-evaluation-browser.py")
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)


def build(runtime, sdk, output):
    runtime, sdk, output = Path(runtime).resolve(), Path(sdk).resolve(), Path(output).resolve()
    for source in (runtime, sdk):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate from source runtime and SDK")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    archive = sdk / f"microsoft.web.webview2.{SDK_VERSION}.nupkg"
    if browser.digest(archive) != SDK_HASH:
        raise ValueError("SDK archive does not match pinned SHA256")
    browser.verify_backend(runtime)
    package = output / NAME
    package.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        for entry, name in (("lib/net462/Microsoft.Web.WebView2.Core.dll", "Microsoft.Web.WebView2.Core.dll"),
                            ("lib/net462/Microsoft.Web.WebView2.WinForms.dll", "Microsoft.Web.WebView2.WinForms.dll"),
                            ("runtimes/win-x64/native/WebView2Loader.dll", "WebView2Loader.dll"),
                            ("LICENSE.txt", "WebView2-LICENSE.txt"), ("NOTICE.txt", "WebView2-NOTICE.txt")):
            (package / name).write_bytes(source.read(entry))
    compiler = Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
                    f"/out:{package / 'NeuroDiscovery Human Evaluation.exe'}",
                    "/reference:System.Windows.Forms.dll", "/reference:System.Drawing.dll",
                    "/reference:System.Web.Extensions.dll",
                    f"/reference:{package / 'Microsoft.Web.WebView2.Core.dll'}",
                    f"/reference:{package / 'Microsoft.Web.WebView2.WinForms.dll'}",
                    str(DESKTOP / "evaluation-webview.cs")], check=True)
    for name in ("backend", "python"):
        shutil.copytree(runtime / name, package / "runtime" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(runtime / "evaluation-manifest.json", package / "runtime/evaluation-manifest.json")
    browser.verify_backend(package / "runtime")
    for source, target in (("evaluation-browser.py", "evaluation-browser.py"),
                           ("evaluation-webview-server.py", "evaluation-webview-server.py"),
                           ("EVALUATION_WEBVIEW_README.md", "README.md")):
        shutil.copy2(DESKTOP / source, package / target)
    files = {path.relative_to(package).as_posix(): browser.digest(path)
             for path in sorted(package.rglob("*")) if path.is_file()}
    (package / "PACKAGE_MANIFEST.json").write_text(json.dumps({"mode": "human-evaluation-system-webview",
        "sdk_version": SDK_VERSION, "sdk_sha256": SDK_HASH, "files": files}, indent=2) + "\n", encoding="utf-8")
    target = output / f"{NAME}.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    receipt = {"archive": str(target), "bytes": target.stat().st_size,
               "sha256": browser.digest(target), "files": len(files), "zip_crc_verified": True,
               "host_source_sha256": browser.digest(DESKTOP / "evaluation-webview.cs")}
    (output / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=DESKTOP / "runtime-evaluation")
    parser.add_argument("--sdk", type=Path, default=DESKTOP / f"build/webview2-sdk-{SDK_VERSION}")
    parser.add_argument("--output", type=Path, default=DESKTOP / "dist-evaluation-webview-r3")
    args = parser.parse_args()
    print(json.dumps(build(args.runtime, args.sdk, args.output), indent=2))


if __name__ == "__main__":
    main()
