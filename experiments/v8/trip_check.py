"""Regression checks for the deadline, the next delivery, and cash-only accounting."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark import configure_engine, limit_math_threads, load_strategy, new_game
from trip_value import build


def main():
    configure_engine()
    limit_math_threads()
    cfg=dict(new_game(791,300)[1].configuration)
    prior=json.loads((ROOT/'analysis/v8/trip-prior.json').read_text())['prior']
    source=build((ROOT/'yan-v7.py').read_text(),prior)
    assert len(source.encode())<10_000_000
    mod=load_strategy('trip_probes', ROOT/'yan-v7.py', source=source.encode())
    import numpy as np
    mine=[100,{'port':0},{'w':[1,60,'COLLECTOR']},{}]
    foe=[100,{'other':112},{},{}]
    stock=np.zeros(225)
    stock[1],stock[15]=30,100
    logits=np.zeros((225,5))
    def offer(remaining,action,me=0,locked=()):
        ob=dict(players=[mine,foe] if me==0 else [foe,mine], player=me, step=299-remaining)
        return mod._v8_trip_offer(ob,cfg,stock,action,logits,locked,[])
    for seat in (0,1):
        assert offer(4,{},seat)=={'w':'LEFT'}  # now allows a second trip worth 30
        assert offer(4,{'w':'LEFT'},seat) is None
        assert offer(1,{},seat)=={'w':'LEFT'}  # today's existing load must reach the port
        assert offer(4,{},seat,{'w'}) is None
    stock[1]=100
    assert offer(2,{'w':'LEFT'})=={}  # enough time to mine and bank
    assert offer(1,{'w':'LEFT'}) is None
    mem={}
    start=[100,{'b':0},{'w':(0,0,'COLLECTOR')},{}]
    field=[100,{'b':0},{'w':(1,48,'COLLECTOR')},{}]
    for t,(cur,old) in enumerate(((start,None),(field,start),(field,field),(start,field))):
        observed=mod._v8_observe_trips(mem,cur,old,t)
    assert observed==[(3,0,3,48.0)]
    assert mod._v8_observe_trips(mem,start,None,0)==[]
    print('Trip decision and observation checks passed.')


if __name__=='__main__':
    main()
