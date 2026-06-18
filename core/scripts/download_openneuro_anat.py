"""Download anatomical MRI files from a public OpenNeuro S3 dataset.

The script uses unsigned S3 access and is resumable: an object is skipped when
the local file exists with the expected size. By default it downloads T1w NIfTI
and JSON sidecars under ``sub-*/anat/`` plus common top-level metadata files.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


DEFAULT_BUCKET = "openneuro.org"


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int


def make_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def list_objects(bucket: str, prefix: str) -> list[S3Object]:
    s3 = make_client()
    out: list[S3Object] = []
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            out.append(S3Object(key=str(obj["Key"]), size=int(obj["Size"])))
        if not resp.get("IsTruncated"):
            break
        token = str(resp.get("NextContinuationToken"))
    return out


def wanted_anat_object(obj: S3Object, modalities: set[str]) -> bool:
    key = obj.key
    if "/anat/" not in key:
        return False
    if not (key.endswith(".nii.gz") or key.endswith(".json")):
        return False
    return any(f"_{mod}" in key for mod in modalities)


def local_path_for_key(output_root: Path, dataset: str, key: str) -> Path:
    prefix = f"{dataset}/"
    rel = key[len(prefix) :] if key.startswith(prefix) else key
    return output_root / rel


def download_one(bucket: str, output_root: Path, dataset: str, obj: S3Object) -> dict[str, object]:
    dst = local_path_for_key(output_root, dataset, obj.key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == obj.size:
        return {"key": obj.key, "path": str(dst), "size": obj.size, "status": "skip_exists"}
    s3 = make_client()
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    s3.download_file(bucket, obj.key, str(tmp))
    actual = tmp.stat().st_size
    if actual != obj.size:
        raise IOError(f"Downloaded size mismatch for {obj.key}: {actual} != {obj.size}")
    tmp.replace(dst)
    return {"key": obj.key, "path": str(dst), "size": obj.size, "status": "downloaded"}


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["key", "path", "size", "status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="OpenNeuro dataset id, e.g. ds000030")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", default=["T1w"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metadata", action="store_true", help="Also download top-level metadata and phenotype files")
    args = parser.parse_args()

    dataset = args.dataset.rstrip("/")
    output_root = Path(args.output_root)
    modalities = set(args.modalities)

    objects = list_objects(args.bucket, f"{dataset}/")
    selected = [obj for obj in objects if wanted_anat_object(obj, modalities)]
    if args.metadata:
        selected.extend(
            obj
            for obj in objects
            if obj.key in {
                f"{dataset}/dataset_description.json",
                f"{dataset}/participants.tsv",
                f"{dataset}/participants.json",
                f"{dataset}/README",
                f"{dataset}/CHANGES",
            }
            or obj.key.startswith(f"{dataset}/phenotype/")
        )
    selected = sorted({obj.key: obj for obj in selected}.values(), key=lambda x: x.key)
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit(f"No anatomical objects found for {dataset} modalities={sorted(modalities)}")

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = [pool.submit(download_one, args.bucket, output_root, dataset, obj) for obj in selected]
        for idx, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            print(f"{idx}/{len(futures)} {row['status']} {row['key']}", flush=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bucket": args.bucket,
        "dataset": dataset,
        "output_root": str(output_root),
        "modalities": sorted(modalities),
        "objects": len(rows),
        "bytes": int(sum(int(row["size"]) for row in rows)),
        "status_counts": {status: sum(1 for row in rows if row["status"] == status) for status in sorted({str(row["status"]) for row in rows})},
    }
    write_manifest(output_root / "metadata" / "openneuro_anat_download_manifest.csv", sorted(rows, key=lambda r: str(r["key"])))
    (output_root / "metadata" / "openneuro_anat_download_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
