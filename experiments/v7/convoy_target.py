"""Inspect inferred collision and receipt probability at the separate recorded turn 190."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import convoy as c
import convoy_fit as ft
from receipt_check import real_target
from receipt_data import layout,TARGET_LAYOUT
from convoy_check import load_policy,ROOT

def soft(x):
    p=np.exp(x-x.max());return p/p.sum()
def main(replay,parent):
    ft.geometry();c._CONVOY_W=np.asarray(json.loads(c.FIT.read_text())['weights'],np.float32)
    g=real_target(replay);assert layout(g)==TARGET_LAYOUT
    t190=190;me=0;m=load_policy(parent);c.WIDX=m.WIDX;c.WACT=m.WACT;c.LAST_TURN=m.LAST_TURN
    def root(o,cfg,est,base,work,lb,previous,start,locked):
        if o['step']==t190:info.update(base=dict(base),work=work,locked=set(locked))
        return base
    m._search=root;info={};steps=g['steps']
    for t,frame in enumerate(steps[:t190+1]):
        obs={k:v for k,v in frame[me]['observation'].items() if k!='full_carbon'}
        m._act(obs,g['configuration'],None)
        m._ST['tr'].set_actions(obs['players'],steps[t+1][me]['action'] or {})
    pl=obs['players'];depots={p for team in pl for p in team[1].values()}
    foresight=m._ST['v5_history']['opponent'][0]
    base=info['base'];work=info['work'];extra=c._convoy_plan(obs,base,work,foresight,info['locked'])
    uid='player-0-worker-23';pos,load,_=pl[0][2][uid];p=soft(work[pos])
    full,_,unpaid,_=c._convoy_paths(pos,p,c._convoy_context(pl,0,pos,load),depots)
    enemies=[];avoid=np.ones((3,225));rootrecords=[]
    for e,(q,v,k) in pl[1][2].items():
        if k!='COLLECTOR':continue
        probs=soft(foresight[q]);ctx=c._convoy_context(pl,1,q,v)
        _,all_,_,unbanked=c._convoy_paths(q,probs,ctx,depots)
        if v<=load:avoid*=1-all_
        if v<65:continue
        route=ft.observed(steps,190,1,e,q)
        enemies.append(dict(uid=e,pos=q,cargo=round(v,2),likely_move=m.WACT[probs.argmax()],
                            actual_route=[int(z[0]) for z in route],
                            likely_positions=[int(r.argmax()) for r in all_],
                            probability_paid_by=[round(float(1-r.sum()),3) for r in unbanked]))
    risk=np.einsum('atp,tp->at',unpaid,1.-avoid)
    for cmd in ('RIGHT','UP'):
        di=m.WIDX[cmd]
        rootrecords.append(dict(action=cmd,root_probability=round(float(p[di]),5),end=int(c._CONVOY_L[pos,di]),
                                unpaid_collision_probability=[round(float(v),4) for v in risk[di]],
                                probability_paid_by=[round(float(1-r.sum()),4) for r in unpaid[di]]))
    report=dict(layout=TARGET_LAYOUT,step=t190,game=1,source='Original match QQ-v6 vs Artemis, seat 0, historical own actions teacher-forced through turn 189.',
                other_movement='Loss-risk uses independent marginal rival occupancy combined cellwise; occupied depots and paths whose cargo already booked are excluded.',
                inspected_uid=uid,cargo=round(load,2),candidates=rootrecords,opposing_loaded=enemies,base=base,convoy=extra,
                replay_evidence='Subsequent receipt amounts and branch-cash attribution are in receipt-target.json and convoy-branches.json.')
    (ROOT/'analysis/v7/convoy-target.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('candidates','opposing_loaded','convoy')}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('replay',type=Path);p.add_argument('parent',type=Path)
    a=p.parse_args();main(a.replay,a.parent)
