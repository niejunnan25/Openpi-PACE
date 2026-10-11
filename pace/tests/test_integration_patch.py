"""The reuse patch must reproduce the release without overwriting local work."""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea"
PATCH = ROOT / "pace/patches/openpi-robotwin.patch"
FILES = (
    ".gitignore",
    "pace/src/openpi_pace/__init__.py",
    "pace/src/openpi_pace/configs.py",
    "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
    "pyproject.toml",
    "scripts/compute_norm_stats.py",
    "src/openpi/serving/websocket_policy_server.py",
    "src/openpi/training/config.py",
)


def git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, check=check, capture_output=True)


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


@pytest.fixture
def upstream(tmp_path):
    git(tmp_path, "init", "-q")
    for name in (*FILES, "README.md", "uv.lock"):
        if name.startswith("pace/"):
            continue
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git(ROOT, "show", f"{BASE}:{name}").stdout)
    return tmp_path


def test_patch_reproduces_release_and_rejects_repeat(upstream):
    before = snapshot(upstream)
    git(upstream, "apply", "--check", str(PATCH))
    assert snapshot(upstream) == before
    git(upstream, "apply", str(PATCH))
    after = snapshot(upstream)
    assert {name for name in after if before.get(name) != after[name]} == set(FILES)
    for name in FILES:
        assert after[name] == (ROOT / name).read_bytes()
    assert git(upstream, "apply", "--check", str(PATCH), check=False).returncode != 0
    assert snapshot(upstream) == after
    git(upstream, "apply", "--reverse", "--check", str(PATCH))


def test_patch_preserves_custom_config_and_other_files(upstream):
    path = upstream / "src/openpi/training/config.py"
    custom = '    TrainConfig(name="my_existing_policy"),\n'
    path.write_text(path.read_text().replace("_CONFIGS = [\n", "_CONFIGS = [\n" + custom))
    (upstream / "my_notes.txt").write_text("Keep this local file.\n")
    git(upstream, "apply", "--check", str(PATCH))
    git(upstream, "apply", str(PATCH))
    assert custom in path.read_text()
    assert path.read_text().replace(custom, "") == (ROOT / "src/openpi/training/config.py").read_text()
    assert (upstream / "my_notes.txt").read_text() == "Keep this local file.\n"


def test_conflicting_patch_leaves_all_files_unchanged(upstream):
    path = upstream / "pyproject.toml"
    path.write_text(path.read_text().replace('build-backend = "hatchling.build"', 'build-backend = "custom.build"'))
    before = snapshot(upstream)
    assert git(upstream, "apply", "--check", str(PATCH), check=False).returncode != 0
    assert git(upstream, "apply", str(PATCH), check=False).returncode != 0
    assert snapshot(upstream) == before
