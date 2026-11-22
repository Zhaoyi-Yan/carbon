"""Build the dual-network challenger from pinned CPU competition files and fitted weights."""
import argparse
import re
from pathlib import Path

import conviction
from export import split
import first_tree
import forecast_build
import prior_search
import shadow
from sapling import export as add_sapling
import tempfile

HERE=Path(__file__).resolve().parent


def _r6_quiet_harvest(players,me,step,x,mine,old):
    """Blend the old collector policy on safe mining squares where its economy was stronger."""
    own,rival=players[me],players[1-me]
    enemies=[p for p,_,role in rival[2].values() if role=='COLLECTOR']
    resource=x[0].reshape(N)
    for pos,cargo,role in own[2].values():
        if role!='COLLECTOR' or cargo>95 or LAST_TURN+1-step<25:continue
        if any(_V5_DIST[pos][q]<=3 for q in enemies):continue
        richest=max(float(resource[q]) for q in _ROLLOUT_LANDING[pos])
        if richest<.20:continue
        weight=min(.48,.18+.30*richest)*(1.-.45*cargo/95.)
        mine[0][pos]*=1.-weight
        mine[0][pos]+=weight*old[0][pos]


def build(weights):
    import inspect
    # Keep the real referee's orientation for the archived rival's own policy.
    with tempfile.TemporaryDirectory() as tmp:
        sap=Path(tmp)/'sapling.py'
        add_sapling(HERE/'checkpoints/canopy.py',HERE/'sapling-model.json',sap)
        code=first_tree.build(sap.read_text())
    code=conviction.build(code)
    code=prior_search.build(code)
    code=shadow.build(code,(HERE.parents[1]/'yan-v5.py').read_text())
    code=forecast_build.build(code,weight=.35)
    for name,blob in zip(('_WEIGHTS','_VALUE'),split(weights)):
        code,n=re.subn('^'+name+r' = "[^"]*"$',lambda _:name+' = "'+blob+'"',code,flags=re.M)
        assert n==1
    anchor='def _shadow_update(obs,'
    code=code.replace(anchor,inspect.getsource(_r6_quiet_harvest)+'\n\n'+anchor)
    anchor="    state['alternatives'].insert(1,(hybrid,old[1]))"
    if code.count(anchor)!=1:raise ValueError(anchor)
    code=code.replace(anchor,anchor+'\n    _r6_quiet_harvest(players,side,step,x,mine,old)')
    # Keep the source lineage and update the old header before hashing or playing.
    code=code.replace('"""Carbon League strategy: Master3s2-family search with a network fine-tuned on top-rated play.',
                      '"""yan-v6 research challenger. Two elite styles, a compact opponent policy and replay-priced economic search.\n\n'
                      'Version-specific training and local evaluation: github.com/Zhaoyi-Yan/carbon, experiments/v6r.\n'
                      'The following attribution describes the original competition agent.')
    compile(code,'<dual-policy-release>','exec')
    if len(code.encode())>10_000_000:raise ValueError('single-file size limit exceeded')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('weights',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(build(a.weights))
