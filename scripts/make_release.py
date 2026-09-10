"""Generate `replication/RELEASE.json` from a built bundle.

Archive digests, file sizes, code provenance, and pinned claims are derived
directly from the bundle. Descriptive metadata is retained from the committed
release pointer.

    uv run python scripts/make_release.py --version vN
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "replication" / "RELEASE.json"

# Derived from the tarball or from git on every run; never carried over.
DERIVED = {
    "release",
    "archive",
    "archive_sha256",
    "archive_size_bytes",
    "files",
    "code_commit",
    "built_from_branch",
    "expected_json_sha256",
    "pinned_claims",
    "pinned_values",
    "runs_included",
    # Frame provenance comes from the bundle manifest so every included run is
    # tied to its actual source frame.
    "frames",
    "frame",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def require_clean_tree() -> None:
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        preview = "\n".join(dirty.splitlines()[:10])
        raise SystemExit(
            "refusing to stamp a release from a dirty working tree; commit or "
            f"remove these changes first:\n{preview}"
        )


def require_bundle_commit(manifest: dict[str, object], current_commit: str) -> None:
    """Require the archive provenance to match the clean checkout being stamped."""
    bundle_commit = manifest.get("code_commit")
    if bundle_commit != current_commit:
        raise SystemExit(
            f"bundle was built from commit {bundle_commit}; current HEAD is "
            f"{current_commit}. Rebuild the bundle from this checkout."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="bundle version, e.g. v4")
    parser.add_argument("--dist", default="dist")
    args = parser.parse_args()
    require_clean_tree()

    name = f"packing-replication-{args.version}"
    dist = ROOT / args.dist
    archive = dist / f"{name}.tar.gz"
    extracted = dist / name
    if not archive.exists():
        raise SystemExit(f"{archive} not found; run `packing bundle` first")
    if not extracted.is_dir():
        raise SystemExit(f"{extracted} not found; the bundle leaves it in place")

    existing = json.loads(RELEASE.read_text(encoding="utf-8")) if RELEASE.exists() else {}
    expected = json.loads((extracted / "expected.json").read_text(encoding="utf-8"))
    manifest = json.loads((extracted / "MANIFEST.json").read_text(encoding="utf-8"))
    current_commit = git("rev-parse", "HEAD")
    require_bundle_commit(manifest, current_commit)
    frames = manifest.get("frames")
    if not frames:
        raise SystemExit(
            f"{extracted / 'MANIFEST.json'} records no frames; rebuild the "
            "bundle with a version of `packing bundle` that writes them"
        )

    files = {
        str(path.relative_to(extracted)): sha256(path)
        for path in sorted(extracted.rglob("*"))
        if path.is_file()
    }
    runs = sorted({key.split("/")[1] for key in files if key.startswith("runs/")})

    payload = {key: value for key, value in existing.items() if key not in DERIVED}
    payload.update(
        release=name,
        archive=archive.name,
        archive_sha256=sha256(archive),
        archive_size_bytes=archive.stat().st_size,
        files=files,
        code_commit=current_commit,
        built_from_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        expected_json_sha256=sha256(extracted / "expected.json"),
        pinned_claims=sorted(expected["claims"]),
        pinned_values=sum(len(v) for v in expected["claims"].values()),
        runs_included=runs,
        frames=frames,
    )

    RELEASE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{RELEASE.relative_to(ROOT)} regenerated")
    print(f"  archive  {payload['archive_sha256'][:16]}  {payload['archive_size_bytes'] / 1e6:.1f} MB")
    print(f"  commit   {payload['code_commit'][:12]}")
    print(f"  files    {len(files)}")
    print(f"  pinned   {payload['pinned_values']} values across {len(payload['pinned_claims'])} claims")
    for frame in frames:
        tag = "primary" if frame.get("is_primary") else ", ".join(frame.get("used_by") or ["-"])
        print(f"  frame    {frame['path']}  {frame['n_items']} items  ({tag})")


if __name__ == "__main__":
    main()
