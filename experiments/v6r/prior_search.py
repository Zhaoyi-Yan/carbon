"""Use both elite move priors as a small tie-breaker in noisy finite-horizon search."""
import argparse
import inspect
from pathlib import Path


def _r6_action_costs(players,me,lw,lb,candidates,step):
    hints=_ST['v5_history']['alternatives']
    aw,ab=hints[0] if hints else (lw,lb)
    initial=candidates[0];cost=np.zeros(len(candidates))
    own=players[me]
    for uid,pos,bottom in [(k,r[0],False) for k,r in own[2].items()]+[(k,p,True) for k,p in own[1].items()]:
        table=BIDX if bottom else WIDX
        plain=(lb if bottom else lw)[pos];alt=(ab if bottom else aw)[pos]
        # Either expert may endorse the deviation. Center each teacher's logits
        # at its own best action before combining.
        relative=np.maximum(plain-plain.max(),alt-alt.max())
        orig=table.get(initial.get(uid),0)
        choice=[table.get(c.get(uid),0) for c in candidates]
        cost+=np.clip(relative[orig]-relative[choice],0.,3.5)
    # Express the prior in the critic's cash/3000 scale. A nats-level
    # disagreement costs only a few carbon compared with a real economic swing.
    left=LAST_TURN-step+1;in_trouble=left<60 and own[0]+350<players[1-me][0]
    unit=.0019 if in_trouble else (.0042 if step<80 else .0032)
    return np.minimum(10.,cost)*unit


def build(code):
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    anchor='def _search(obs,'
    change(anchor,inspect.getsource(_r6_action_costs)+'\n\n'+anchor)
    anchor='    alive = list(range(len(cands)))'
    change(anchor,'    priors=_r6_action_costs(players,me,lw0,lb0,cands,step)\n'+anchor)
    anchor="mean = {c: sum(val[c, m] for m in range(M)) / M for c in alive}"
    change(anchor,"mean = {c: sum(val[c, m] for m in range(M)) / M - priors[c] for c in alive}")
    old="mean = {c: sum(val[c, m] for m in range(rolls)) / rolls for c in alive}"
    if code.count(old)!=2:raise ValueError('expected the two final comparisons')
    code=code.replace(old,"mean = {c: sum(val[c, m] for m in range(rolls)) / rolls - priors[c] for c in alive}")
    compile(code,'<regularized-search>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.source.read_text()))
