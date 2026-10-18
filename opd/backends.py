"""Assemble the veRL backends from one shared base tree plus small per-backend overlays.

vendor/verl-original is the common base. Each other backend is the base with the
files under vendor/overlays/NAME/ copied on top. Assembled trees are written to
vendor/.build/NAME (ignored by Git) and checked file by file against
provenance/frozen_vendor_sha256.json before use.
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .paths import ROOT

BASE = 'verl-original'
BACKENDS = ('verl-original', 'verl-code', 'verl-full', 'verl-qwen')
VENDOR = ROOT / 'vendor'
BUILD = VENDOR / '.build'
MANIFEST = ROOT / 'provenance/frozen_vendor_sha256.json'
SKIP = shutil.ignore_patterns('__pycache__', '*.pyc')


def tree_hashes(root, prefix):
    """SHA-256 of every file under root, keyed as prefix/relative/path."""
    return {f'{prefix}/{p.relative_to(root).as_posix()}': hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def assemble(name, destination):
    shutil.copytree(VENDOR / BASE, destination, ignore=SKIP)
    shutil.copytree(VENDOR / 'overlays' / name, destination, ignore=SKIP, dirs_exist_ok=True)


def verify(name, root):
    prefix = f'vendor/{name}'
    expected = {k: v for k, v in json.loads(MANIFEST.read_text()).items() if k.startswith(prefix + '/')}
    actual = tree_hashes(root, prefix)
    if actual != expected:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise RuntimeError(f'Backend {name} does not match {MANIFEST.name}: {changed[:5]}')


def backend_path(name):
    """Return a verified directory for the named backend, assembling it on first use."""
    if name not in BACKENDS:
        raise ValueError(f'Unknown backend: {name!r}')
    if name == BASE:
        return VENDOR / BASE
    target = BUILD / name
    if target.is_dir():
        try:
            verify(name, target)
            return target
        except RuntimeError:
            shutil.rmtree(target)
    BUILD.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD) as staging:
        tree = Path(staging) / name
        assemble(name, tree)
        verify(name, tree)
        tree.rename(target)
    return target


def manifest_hashes():
    """Hashes of every vendored tree as used at runtime; regenerates the frozen manifest."""
    hashes = {}
    for path in sorted(VENDOR.iterdir()):
        if path.is_dir() and path.name not in ('overlays', '.build'):
            hashes.update(tree_hashes(path, f'vendor/{path.name}'))
    with tempfile.TemporaryDirectory() as staging:
        for name in BACKENDS:
            if name != BASE:
                assemble(name, Path(staging) / name)
                hashes.update(tree_hashes(Path(staging) / name, f'vendor/{name}'))
    return dict(sorted(hashes.items()))
