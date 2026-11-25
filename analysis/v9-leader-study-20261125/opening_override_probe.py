"""Check the frozen v9 opening call order on recorded initial observations.

This runs only the two extracted, deterministic opening helpers. It does not run
the policy network, search, new games, or an estimated opponent. Zero base logits
make role placement arbitrary; the invariant being checked is wait versus hire.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE = 'e97a386b290a1727f26331470094ab57cd452948'
SOURCE_SHA = '83bc00ca52ba42ece876830649c9bc488105e763179b62018558f555bb94fa9c'
V9 = '0f9896ed3dc3cd7bfad29fe2dd38d6fb'
INDEX_SHA = 'a328b208a39a325907fd249aceb97895fefc89322eae090309bc991ee1471096'


class ZeroLogits:
    def __getitem__(self, key):
        return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = subprocess.check_output([
        'git', '-C', str(args.repository), 'show', SOURCE + ':yan-v8-own-prior.py'
    ])
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    tree = ast.parse(source)
    names = {'_r6_opening', '_v7_early_income'}
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(helpers) == 2
    mode = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '_R6_OPENING'
                        for target in node.targets))
    assert mode == 'react'
    act = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_act')
    calls = sorted((node.lineno, node.func.id) for node in ast.walk(act)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in names)
    assert [name for _, name in calls] == ['_r6_opening', '_v7_early_income']
    scope = {'_R6_OPENING': mode, 'BACT': [None, 'RECCOLLECTOR', 'RECPLANTER']}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), '<frozen opening helpers>', 'exec'), scope)
    index_bytes = (args.snapshot / 'index.json').read_bytes()
    assert hashlib.sha256(index_bytes).hexdigest() == INDEX_SHA
    records = []
    for meta in json.loads(index_bytes):
        slots = [slot for slot, strategy in meta['strategies'].items() if strategy['id'] == V9]
        if not slots:
            continue
        assert len(slots) == 1
        seat = meta['order'].index(slots[0])
        replay_bytes = (args.snapshot / meta['file']).read_bytes()
        replay = json.loads(gzip.decompress(replay_bytes))
        obs = dict(replay['steps'][0][0]['observation'])
        obs.update(replay['steps'][0][seat]['observation'])
        obs.pop('full_carbon', None)
        obs['player'] = seat
        assert obs['step'] == 0 and not obs['players'][seat][2]
        config = replay['configuration']
        reactive = scope['_r6_opening'](obs, config, ZeroLogits(), {})
        final = scope['_v7_early_income'](obs, config, ZeroLogits(), reactive)
        assert reactive == {}
        assert Counter(final.values()) == Counter({'RECCOLLECTOR': 1, 'RECPLANTER': 1})
        observed = replay['steps'][1][seat]['action'] or {}
        assert Counter(observed.values()) == Counter(final.values())
        # Synthetic next-step, mutually empty state: the existing helper hires
        # immediately, so this proposed restoration is not a 45-turn waiting rule.
        synthetic = dict(obs, step=1)
        next_action = scope['_r6_opening'](synthetic, config, ZeroLogits(), {})
        assert Counter(next_action.values()) == Counter(final.values())
        records.append(dict(match=meta['id'], game=meta['game'], seat=seat,
                            replay_sha256=hashlib.sha256(replay_bytes).hexdigest(),
                            reactive_action=reactive, overridden_role_counts=dict(Counter(final.values())),
                            observed_action=observed, synthetic_step1_hires=True))
    assert len(records) == 30
    result = dict(source_commit=SOURCE, source_sha256=SOURCE_SHA, index_sha256=INDEX_SHA,
                  opening_call_order=[{'line': line, 'helper': name} for line, name in calls],
                  games=len(records), wait_overridden_in_all=True, records=records,
                  limits=['Zero logits test only the wait/hire invariant, not actual network-selected base placement.',
                          'No new games or performance comparison were run.',
                          'Online v9-to-local artifact identity is reported by the user; public replay metadata does not include source SHA.',
                          'The old immediate-hire rule was intentional and had prior development evidence; this does not prove it is harmful.'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Checked', len(records), 'initial observations; existing wait is overridden in all.')


if __name__ == '__main__':
    main()
