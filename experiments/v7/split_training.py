"""Use our self-play worker proposal in ordinary positions and the Master policy under pressure."""
import argparse
import hashlib
from pathlib import Path

from own_search_prior import build as add_selfplay_prior

HERE = Path(__file__).resolve().parent
PARENT = '18a347615623b0501f75e9a565054b3b3aaa0c8fbbca1c30f7d952b1c7435f86'


def build(source):
    if hashlib.sha256(source.encode()).hexdigest() != PARENT:
        raise ValueError('Expected the frozen conditionally trained Master worker network')
    source = add_selfplay_prior(source, HERE / 'own-search-model.json')
    original = "        st['own_search_prior']=_v7_search_teacher(x,players,me,lw,st['prev'])"
    if source.count(original) != 1:
        raise ValueError('Unexpected root worker proposal')
    gated = """        if st['v5_history']['mm_active']:
            st['own_search_prior'] = None
        else:
            st['own_search_prior'] = _v7_search_teacher(x, players, me, lw, st['prev'])"""
    return source.replace(original, gated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    code = build((HERE / 'checkpoints/master-multi-film.py').read_text())
    compile(code, str(args.output), 'exec')
    with args.output.open('x') as f:
        f.write(code)
