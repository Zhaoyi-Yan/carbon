"""Share opponent teacher corrections across neighboring counterfactual search boards."""
import argparse
import base64
import hashlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA256 = '1995bbcc07592296e6fe6f816412acccb2d39ea2c79787cd937a210513f69afa'


def _v6r_counterfactuals(net, X, S, side, part, ours=None):
    """Use the same-reply reference board to correct a student's local action logits."""
    n = len(part)
    both = ours is not None
    enemy = X[1::2] if both else X
    enemy_scalar = S[1::2] if S is not None and both else S
    refs, group, indices = {}, [], []
    for i, board in enumerate(part):
        reply = board[1]
        if reply not in refs:
            refs[reply] = len(indices)
            indices.append(i)
        group.append(refs[reply])
    off = n if both else 0
    teacher_rows = list(range(0,2*n,2)) + [2*i+1 for i in indices] if both else indices
    teacher_sides = ([ours]*n if both else []) + [side]*len(indices)
    tw, tb, tv = net(X[teacher_rows], _v5_ids(teacher_sides),
                     S[teacher_rows] if S is not None else None, play_sides=teacher_sides)
    sw, sb, sv = _V6R_TINY(np.ascontiguousarray(enemy), _v5_ids([side]*n), enemy_scalar)
    _S_STATS['student_rows'] = _S_STATS.get('student_rows',0) + n
    # Safe to reuse a directional correction only for matching worker roles and
    # immediate collision features. The full teacher resolves every reference board.
    refrows = np.asarray(indices)[group]
    planes = enemy.reshape(n,NPL,225)
    mask = (planes[:,2]==planes[refrows,2]) & (planes[:,4]==planes[refrows,4])
    mask &= (planes[:,25:35] == planes[refrows,25:35]).all(axis=1)
    ew = sw + (tw[off:][group] - sw[refrows])*mask[:,:,None]
    eb = sb + (tb[off:][group] - sb[refrows])
    ew[indices], eb[indices] = tw[off:], tb[off:]
    if not both:
        return ew, eb, None
    # Only interim, unpruned nodes need an opponent value. Critical pruning and
    # horizon/final values still come from the incumbent's full evaluator.
    ev = sv + tv[off:][group] - sv[refrows]
    w, b = np.empty((2*n,225,5),np.float32), np.empty((2*n,225,3),np.float32)
    w[::2], w[1::2] = tw[:n], ew
    b[::2], b[1::2] = tb[:n], eb
    return w,b,.5*(tv[:n]-ev)


def build(source, student):
    if hashlib.sha256(source).hexdigest()!=BASE_SHA256:
        raise ValueError('Expected frozen yan-v6.py')
    text=source.decode()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:
            raise ValueError(old)
        text=text.replace(old,new)
    replace('_S_D2 = 7 ', '_S_D2 = 9 ')
    replace('for candidate in (2, 3, 5, 7)', 'for candidate in (2, 3, 5, 7, 9)')
    replace('def _v5_ids(sides):',inspect.getsource(_v6r_counterfactuals)+'\n\ndef _v5_ids(sides):')
    cheap_src=base64.b64encode(student).decode()
    init = """if _SEARCH_OK:
    _V6R_BYTES = 'STUDENT_ARCHIVE'
    with np.load(io.BytesIO(base64.b64decode(_V6R_BYTES)),allow_pickle=False) as _tiny:
        _V6R_TINY = _build(_Policy(_V6R_BYTES),{k:_tiny[k] for k in _tiny if k.startswith(('v1.','v2.','v3.'))})
""".replace('STUDENT_ARCHIVE',cheap_src)
    replace('\n_net()\n','\n_net()\n'+init)
    pair="""                lw, lb, vv = net(X, _v5_ids([me,op]*n if both else [me]*n), S, play_sides=([me,op]*n if both else [me]*n))
                vv = .5*(vv[::2]-vv[1::2]) if both else vv"""
    replace(pair,"""                if both and n >= 4:
                    lw,lb,vv = _v6r_counterfactuals(net,X,S,op,part,me)
                else:
"""+'\n'.join('    '+line for line in pair.splitlines()))
    other='                lw2, lb2, _ = net(Xo, _v5_ids([op]*len(part)), play_sides=[op]*len(part))'
    replace(other,"""                if len(part) >= 4:
                    lw2,lb2,_ = _v6r_counterfactuals(net,Xo,None,op,part)
                else:
    """+other)
    compile(text,'<anchored-search>','exec')
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    text=build((ROOT/'yan-v6.py').read_bytes(),(ROOT/'experiments/v5/rollout-student.npz').read_bytes())
    with args.output.open('x') as f:
        f.write(text)
    print(hashlib.sha256(text.encode()).hexdigest(),args.output)
