#!/usr/bin/env bash
set -euo pipefail

# Apply the simulator patches required by the pinned RoboTwin revision.
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sysconfig

assert version("sapien") == "3.0.0b1"
assert version("mplib") == "0.2.1"
site = Path(sysconfig.get_paths()["purelib"])
patches = {
    "sapien/wrapper/urdf_loader.py": [
        ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
        ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
    ],
    "mplib/planner.py": [
        ("if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
         "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"),
    ],
}
for relative, replacements in patches.items():
    path = site / relative
    text = path.read_text()
    for before, after in replacements:
        if after in text:
            continue  # Allow rerunning setup in an already patched environment.
        if text.count(before) != 1:
            raise RuntimeError(f"Unexpected source in {path}; check the package version")
        text = text.replace(before, after)
    path.write_text(text)
    print(f"Patched: {path}")
PY
