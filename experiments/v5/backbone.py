"""Expose the frozen v4 trunk before its action heads, with either CPU backend."""
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import configure_engine, limit_math_threads, load_strategy

V4_SHA = 'a3d54e4672458e53f5dfd47444b59a25326f5d1592575f968a1a59f26c98b4b0'


def replace(source, before, after, count=1):
    if source.count(before) != count:
        raise ValueError('Unexpected source around ' + before[:80])
    return source.replace(before, after)


def exposed_source():
    source = (ROOT / 'opponents/yan-v4.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == V4_SHA
    source = source.decode()
    source = replace(source, 'def __call__(self, X, sid, sc=None):',
                     'def __call__(self, X, sid, sc=None, hidden=False):', 2)
    source = replace(source, '            hf = h.flatten(2)\n',
                     '            hf = h.flatten(2)\n            if hidden:\n                return hf.transpose(1, 2).contiguous().numpy()\n')
    source = replace(source, '            hf = h[self.board]\n',
                     '            hf = h[self.board]\n            if hidden:\n                return hf.permute(1, 0, 2).contiguous().numpy()\n')
    return source


def load():
    configure_engine()
    limit_math_threads()
    return load_strategy('v5_feature_trunk', ROOT / 'opponents/yan-v4.py', exposed_source().encode())
