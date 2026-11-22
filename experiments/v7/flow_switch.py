"""Respond to a persistent mining deficit only when recent bank and combat returns do not cover it."""
import argparse
import inspect
from pathlib import Path


def _r6_economic_mode(state, players, side, step):
    previous=state.get('flow_prior')
    state['flow_prior']=players
    if previous is None or not step:
        state['mine_gap']=state['loot_gap']=state['cash_gap']=0.
        return 0.
    mine=[0.,0.];loot=[0.,0.]
    for team in (0,1):
        now=players[team][2]
        enemy_old=previous[1-team][2]
        vanished=[(q,c) for uid,(q,c,k) in enemy_old.items() if uid not in players[1-team][2]]
        for uid,(p,c,kind) in now.items():
            before=previous[team][2].get(uid)
            if before is None or kind!='COLLECTOR':continue
            q,old,_=before
            gain=max(0.,c-old)
            # Collision gains can appear on a stationary or a moving worker. A
            # newly recruited rival cannot contribute positive cargo this turn.
            captured=min(gain,sum(v for z,v in vanished if _V5_DIST[z][p]<=1))
            loot[team]+=captured
            if p==q:mine[team]+=gain-captured
    other=1-side
    reserve=players[side][0]-players[other][0]
    old_cash=previous[side][0]-previous[other][0]
    for name,value in (('mine_gap',mine[other]-mine[side]),('loot_gap',loot[side]-loot[other]),('cash_gap',reserve-old_cash)):
        state[name]=.958*state[name]+value
    need=state['mine_gap']-.85*max(0.,state['loot_gap'])-.55*max(0.,state['cash_gap'])
    # A bounded transition between the two existing policies, with hysteresis.
    last=state.get('mine_mode',0.)
    available=np.clip((need-42.)/145.,0.,1.) if 20<step<LAST_TURN-22 else 0.
    mode=.83*last+.17*available
    state['mine_mode']=mode
    return mode


def build(code):
    anchor='def _r6_quiet_harvest(players,me,step,x,mine,old):'
    if code.count(anchor)!=1:raise ValueError('expected collector blending')
    code=code.replace(anchor,inspect.getsource(_r6_economic_mode)+'\n\n'+anchor)
    anchor="    resource=x[0].reshape(N)\n"
    if code.count(anchor)!=1:raise ValueError('expected collector resources')
    code=code.replace(anchor,anchor+"    mode=_r6_economic_mode(_ST['v5_history'],players,me,step)\n")
    anchor='        weight=min(.48,.18+.30*richest)*(1.-.45*cargo/95.)'
    if code.count(anchor)!=1:raise ValueError('expected policy weight')
    code=code.replace(anchor,anchor+'\n        weight+=mode*(.87-weight)*min(1.,.45+richest)')
    compile(code,'<economic-mode>','exec')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.output.open('x') as f:f.write(build(a.source.read_text()))
