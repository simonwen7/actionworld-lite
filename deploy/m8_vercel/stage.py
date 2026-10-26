"""Build the isolated, allowlisted M8 Vercel staging directory outside the repository.

Usage (from the repository root):
    .venv/bin/python -B deploy/m8_vercel/stage.py DESTINATION

DESTINATION must not exist and must be outside the repository. Only the files
in ALLOWLIST plus the Dockerfile are copied; no Git metadata, datasets, TEST
archives, validation reports, virtual environments or caches.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m8_web.model_adapter import ARTIFACTS  # noqa: E402

RUNTIME_ARTIFACTS = (
    "data/processed/m4_visual_v03/best_model.pt",
    "data/processed/m4_yaw_v04/best_model.pt",
    "data/processed/m6_residual_v01/model.json",
)
ALLOWLIST = (
    "app.py",
    "requirements.txt",
    "m8_web/__init__.py",
    "m8_web/api.py",
    "m8_web/engine.py",
    "m8_web/model_adapter.py",
    "m8_web/session.py",
    "public/index.html",
    "public/m8-app.js",
    "public/m8-style.css",
    "src/models/integrated_pusht_m4.py",
    "src/models/visual_dynamics.py",
    "src/models/visual_dynamics_v03.py",
    "src/models/yaw_specialist_v04.py",
    "src/research/m6_residual_v01.py",
    "src/simulation/pusht_env.py",
    "src/simulation/pusht_scene.xml",
) + RUNTIME_ARTIFACTS
GENERATED = {
    "Dockerfile": (ROOT / "deploy/m8_vercel/Dockerfile").read_bytes(),
    ".dockerignore": b".vercel\n.vercelignore\n.dockerignore\n.env*\n.gitignore\n",
    ".vercelignore": b".vercel\n.env*\n.gitignore\n",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    dest = Path(sys.argv[1]).expanduser().resolve()
    if dest == ROOT or ROOT in dest.parents:
        raise SystemExit("Staging directory must be outside the repository")
    if dest.exists():
        raise SystemExit(f"Staging directory already exists: {dest}")
    missing = set(ARTIFACTS) - set(ALLOWLIST)
    if missing:
        raise SystemExit(f"Frozen runtime inputs missing from allowlist: {sorted(missing)}")

    dest.mkdir(parents=True)
    manifest = []
    for relative in ALLOWLIST:
        data = (ROOT / relative).read_bytes()
        expected = ARTIFACTS.get(relative)
        if expected is not None and digest(data) != expected:
            raise SystemExit(f"Frozen SHA-256 mismatch: {relative}")
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({"path": relative, "bytes": len(data), "sha256": digest(data)})
    for relative, data in GENERATED.items():
        (dest / relative).write_bytes(data)
        manifest.append({"path": relative, "bytes": len(data), "sha256": digest(data)})

    copied = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    if copied != sorted(entry["path"] for entry in manifest):
        raise SystemExit("Staging directory contains unexpected files")
    weights = [p for p in copied if p.startswith("data/")]
    if sorted(weights) != sorted(RUNTIME_ARTIFACTS):
        raise SystemExit("Staging must contain exactly the three authorized runtime artifacts")
    if any(p.startswith("public/") and not p.endswith((".html", ".js", ".css")) for p in copied):
        raise SystemExit("Unexpected file under public/")

    for entry in manifest:
        print(f"{entry['bytes']:>9}  {entry['sha256']}  {entry['path']}")
    total = sum(entry["bytes"] for entry in manifest)
    print(f"{len(manifest)} files, {total} bytes -> {dest}")
    (dest.parent / "stage_manifest.json").write_text(
        json.dumps({"staging": str(dest), "files": manifest}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
