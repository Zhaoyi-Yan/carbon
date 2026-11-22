"""Build a progressively screened CPU search from the frozen yan-v6."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA256 = '1995bbcc07592296e6fe6f816412acccb2d39ea2c79787cd937a210513f69afa'


def build(source):
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError('Expected frozen yan-v6.py')
    text = source.decode()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(old)
        text = text.replace(old, new)

    replace('_S_D2 = 7 ', '_S_D2 = 9 ')
    replace('for candidate in (2, 3, 5, 7)', 'for candidate in (2, 3, 5, 7, 9)')
    replace('_v5_value = _v5_territory_value', '''def _v5_value(board, side):
    key = ('territory_value', side)
    if key not in board:
        board[key] = _v5_territory_value(board, side)
    return board[key]''')
    replace('        out = []\n        for c in cs:', '        out = []\n        children = {}\n        for c in cs:')
    replace('                out.append([c, m, _turn(root, configuration, acts), players])', '''                key = (c, tuple(sorted(opp[m].items())))
                if key not in children:
                    children[key] = _turn(root, configuration, acts)
                out.append([c, m, children[key], players])''')
    replace('    def look(live, both):', '''    def look(live, both):
        # Two replies that generated the same engine state share one network evaluation.
        unique, indices, by_id = [], [], {}
        for board in live:
            key = id(board[2])
            index = by_id.get(key)
            if index is None:
                index = len(unique)
                by_id[key] = index
                unique.append(board)
            indices.append(index)
        values, policies = evaluate(unique, both)
        if len(unique) == len(live):
            return values, policies
        return [values[i] for i in indices], [policies[i] for i in indices]

    def evaluate(live, both):''')
    replace('        for (lwm, lbm, lwp, lbp), b in zip(pol, live):', '''        children = {}
        for (lwm, lbm, lwp, lbp), b in zip(pol, live):''')
    replace("            b[3] = o['players']; b[2] = _turn(o, configuration, [am, ao] if me == 0 else [ao, am])", '''            key = (id(o), tuple(sorted(am.items())), tuple(sorted(ao.items())))
            if key not in children:
                children[key] = _turn(o, configuration, [am, ao] if me == 0 else [ao, am])
            b[3] = o['players']
            b[2] = children[key]''')
    # Protect the proposed root, and keep the first alternative to preserve the network's prior.
    # The original 50% first-stage cut and all later rankings still use the full network.
    replace('    depth = 0; cut = True; rolls = M', '''    if D2 >= 3 and len(cands) >= 8:
        screen = [0.] * len(cands)
        for c, m, board, _ in boards:
            screen[c] += _v5_value(board, me)
        limit = max(5, (2 * len(cands) + 2) // 3)
        ranked = sorted(range(2, len(cands)), key=lambda c: -screen[c])
        alive = [0, 1] + ranked[:limit-2]
    depth = 0; cut = True; rolls = M''')
    compile(text, '<screened-search>', 'exec')
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    text = build((ROOT/'yan-v6.py').read_bytes())
    with args.output.open('x') as out:
        out.write(text)
    print(hashlib.sha256(text.encode()).hexdigest(), args.output)
