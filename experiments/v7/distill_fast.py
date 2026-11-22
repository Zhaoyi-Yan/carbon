"""Distill search replies from existing root logits and board geometry, without extra neural passes."""
import argparse
import inspect
import json
from pathlib import Path
import distill
import distill_id

FIT=Path(__file__).with_name('distill-fast-model.json')

def build(code):
    anchor='def _v5_root(obs, est, x, previous):'
    before='    _v5_rollout_set(me,style,rival)'
    if code.count(anchor)!=1 or code.count(before)!=1 or '_DS_COEF=' in code:
        raise ValueError('Expected compatible frozen v6 root prediction')
    m=json.loads(FIT.read_text())
    if m['extra_net']:raise ValueError('Expected board-only fit')
    feature=inspect.getsource(distill._ds_features)
    old='    a,b=(center(experts[k,positions]) for k in (0,1))'
    if feature.count(old)!=1:raise ValueError('Feature specification changed')
    feature=feature.replace(old,'    a=b=np.zeros_like(z0)')
    reply=inspect.getsource(distill_id._ds_class_reply).replace('_ds_experts(view)','None')
    const='_DS_LAND=np.array([[move(p,a) for a in range(5)] for p in range(225)])\n'
    for sym,key in (('_DS_COEF','positive_coefficients'),('_DS_SCALE','positive_scale'),
                    ('_DS_NEG_COEF','coefficients'),('_DS_NEG_SCALE','scale')):
        const+=sym+'=np.asarray('+repr(m[key])+',np.float32)\n'
    source='\n\n'.join((feature,inspect.getsource(distill._ds_logprob),inspect.getsource(distill_id._ds_identify),reply))
    hook='    _ds_class_reply(state,players,op,step,opponent_x,previous)\n'
    return code.replace(anchor,const+source+'\n\n'+anchor).replace(before,hook+before)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();code=build(a.parent.read_text());compile(code,str(a.output),'exec')
    with a.output.open('x') as f:f.write(code)
