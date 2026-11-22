"""Embed a refitted trunk or a compressed simulated-turn model in a standalone agent."""
import base64
import io
import re
import numpy as np
from backbone import replace


def archive(fields):
    buffer=io.BytesIO()
    np.savez_compressed(buffer,**fields)
    return base64.b64encode(buffer.getvalue()).decode()


def split(path):
    with np.load(path,allow_pickle=False) as data:
        v={k:data[k] for k in data if k in {stage+'.'+part for stage in ('v1','v2','v3') for part in ('weight','bias')}}
        p={k:data[k] for k in data if k not in v}
    return archive(p),archive(v)


def patch_networks(source,full,small):
    if not full and not small:return source
    source=replace(source,'a, b = net(X, _SID, S), w(X, _SID, S)',
                   "test_sid = min(_SID,len(pol.p['emb'])-1)\n            a, b = net(X, test_sid, S), w(X, test_sid, S)")
    source=replace(source,'self.emb[sid][None, :, None, None]',
                   '(self.emb[sid][None, :, None, None] if np.isscalar(sid) else self.emb[sid][:, :, None, None])')
    # Real opponent identity is unavailable. Its inferred posterior determines the simulation policy.
    source=replace(source,'    eye = _V5_EYE',"""    enemy_style = int(rival.argmax()) if rival.max()>.16 else 20
    ids = [None,None]
    ids[me],ids[op] = style,enemy_style
    state['simulation_ids'] = ids
    eye = _V5_EYE""")
    if full:
        pol,val=split(full)
        source=re.sub(r'^_WEIGHTS = "[^"]*"$',lambda m:'_WEIGHTS = "'+pol+'"',source,count=1,flags=re.M)
        source=re.sub(r'^_VALUE = "[^"]*"$',lambda m:'_VALUE = "'+val+'"',source,count=1,flags=re.M)
        source=replace(source,"_SID = _POLICY.sid('CL-CarbonEdge_Apollo')","_SID = 20")
        # The root's orientations are interleaved ours/rival; proper style is known before encoding.
        source=replace(source,'    opponent_x = _quant(encode(players,step,op,est,[1.]*N,previous))',
                       """    enemy_style = int(rival.argmax()) if rival.max()>.16 else 20
    state['root_ids'] = [style,enemy_style]
    opponent_x = _quant(encode(players,step,op,est,[1.]*N,previous))""")
        marker='np.concatenate(batch).astype(np.float32), _SID, hidden=True'
        if marker in source:
            source=replace(source,marker,"np.concatenate(batch).astype(np.float32), np.tile(_ST['v5_history']['root_ids'],len(batch)), hidden=True")
        else:
            source=replace(source,'np.stack((x,opponent_x)).astype(np.float32), _SID, hidden=True',
                           "np.stack((x,opponent_x)).astype(np.float32), np.array(_ST['v5_history']['root_ids']), hidden=True")
        source=replace(source,'lw, lb, vv = net(X, _SID, S, play_sides=([me,op]*n if both else [me]*n))',
                       "lw, lb, vv = net(X, _v5_ids([me,op]*n if both else [me]*n), S, play_sides=([me,op]*n if both else [me]*n))")
        source=replace(source,'lw2, lb2, _ = net(Xo, _SID, play_sides=[op]*len(part))',
                       'lw2, lb2, _ = net(Xo, _v5_ids([op]*len(part)), play_sides=[op]*len(part))')
    source=replace(source,'def _search(obs,',"""def _v5_ids(sides):
    lookup = _ST['v5_history']['simulation_ids']
    return np.array([lookup[s] for s in sides], dtype=np.int64)


def _search(obs,""")
    if not small:return source
    pol,val=split(small)
    extra="""_V5_STUDENT = None
def _v5_student():
    global _V5_STUDENT
    if _V5_STUDENT is None:
        with np.load(io.BytesIO(base64.b64decode(_V5_STUDENT_VALUE)), allow_pickle=False) as z:
            _V5_STUDENT = _build(_Policy(_V5_STUDENT_POLICY),dict(z))
    return _V5_STUDENT

"""
    source=replace(source,'def _v5_ids(sides):',"_V5_STUDENT_POLICY = '"+pol+"'\n_V5_STUDENT_VALUE = '"+val+"'\n"+extra+'def _v5_ids(sides):')
    line=next(l for l in source.splitlines() if 'lw, lb, vv = net(X,' in l)
    source=replace(source,line,"""                if both:
                    lw,lb,vv = _v5_student()(X,_v5_ids([me,op]*n),S)
                else:
    """+line)
    line=next(l for l in source.splitlines() if 'lw2, lb2, _ = net(Xo,' in l)
    source=replace(source,line,'                lw2,lb2,_ = _v5_student()(Xo,_v5_ids([op]*len(part)))')
    source=replace(source,'\n_net()\n',"\n_net()\nif _SEARCH_OK:\n    _v5_student()\n")
    source=replace(source,'_S_D2 = 7 ','_S_D2 = 10 ')
    source=replace(source,'for candidate in (2, 3, 5, 7)','for candidate in (3, 5, 7, 10)')
    return source
