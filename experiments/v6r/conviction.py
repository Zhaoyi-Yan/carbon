"""Keep well-supported early worker moves when a shallow rollout has no nearby tactical objection."""
import argparse
import inspect
from pathlib import Path


def _r6_early_locks(obs, action, logits):
    t=obs['step']
    if t<2 or t>35:return {}
    side=obs['player'];own=obs['players'][side];enemy=obs['players'][1-side]
    locked={}
    for uid,(pos,cargo,kind) in own[2].items():
        a=WIDX.get(action.get(uid),0)
        if a!=int(logits[pos].argmax()) or _probs(logits[pos],.30)[a]<.94:continue
        dst=_ROLLOUT_LANDING[pos][a]
        threatened=any(_V5_DIST[q][dst]<=1 and
                       ((role=='COLLECTOR' and (kind=='PLANTER' or load<=cargo)) or
                        (role=='PLANTER' and kind=='PLANTER'))
                       for q,load,role in enemy[2].values())
        if not threatened:locked[uid]=action.get(uid)
    return locked


def build(code):
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    anchor='def _search(obs,'
    change(anchor,inspect.getsource(_r6_early_locks)+'\n\n'+anchor)
    anchor="    fixed = {wid: act0.get(wid) for wid in home} if _S_HOMEFIX else None"
    change(anchor,anchor+"""
    protected=_r6_early_locks(obs,act0,lw0)
    if protected:
        fixed=dict(fixed or {},**protected)""")
    anchor='    if len(cands) < 2:\n        _S_STATS'
    change(anchor,"""    if protected:
        cands=[a for a in cands if all(a.get(uid)==move for uid,move in protected.items())]
"""+anchor)
    anchor="            act = _reply_choice(obs,configuration,est,act,lw,{c[1] for c in cand if c[0]==2.0},t0)"
    change(anchor,"""            risk_locked={c[1] for c in cand if c[0]==2.0}
            risk_locked.update(_r6_early_locks(obs,act,lw))
            act = _reply_choice(obs,configuration,est,act,lw,risk_locked,t0)""")
    compile(code,'<early-root-confidence>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
