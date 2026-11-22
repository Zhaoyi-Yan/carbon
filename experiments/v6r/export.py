"""Embed the two separately trained elite styles in the CPU tournament agent."""
import argparse
import hashlib
from pathlib import Path
import re
import sys
from subprocess import check_output

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/v5'))
from model_export import split

PARENT='1995bbcc07592296e6fe6f816412acccb2d39ea2c79787cd937a210513f69afa'
REV='9907fc560c5cc0cf5164014178351dbfb5d64c29'

def original():
    raw=(ROOT/'yan-v6.py').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT:
        raw=check_output(['git','-C',str(ROOT),'show',REV+':yan-v6.py'])
    assert hashlib.sha256(raw).hexdigest()==PARENT
    return raw.decode()


def build(weights,mode,source=None):
    code=source if source is not None else original()
    pol,val=split(weights)
    for name,value in (('_WEIGHTS',pol),('_VALUE',val)):
        code,n=re.subn('^'+name+r' = "[^"]*"$',lambda _:name+' = "'+value+'"',code,flags=re.M)
        assert n==1
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError(old)
        code=code.replace(old,new)
    # Styles 21 and 22 are new. Rival IDs and the old action-head table keep their original indexing.
    if mode=='artemis':selection='    expert = 22\n'
    elif mode=='apollo':selection='    expert = 21\n'
    else:
        selection='''    # Artemis and Apollo specialize the shared trunk. In a sustained economic mismatch switch proposals.
    treasury = players[me][0]
    delta = econ[me] - econ[op]
    pressure = .8*max(0.,-delta[5]/4.) + .7*max(0.,-delta[7]) + .4*max(0.,delta[3]/3.)
    avg = .84*state.get('elite_pressure',pressure)+.16*pressure
    state['elite_pressure'] = avg
    expert = 21 if (avg > (1.20 if state.get('elite_id')==21 else 1.80) and step>32 and treasury<players[op][0]+350) else 22
    state['elite_id'] = expert
'''
    change("    state['root_ids'] = [style,enemy_style]",selection+"    state['root_ids'] = [expert,enemy_style]")
    change("    ids[me],ids[op] = style,enemy_style","    ids[me],ids[op] = expert,enemy_style")
    change("_V5_OWN_STRENGTH = 0.27","_V5_OWN_STRENGTH = 0.09")
    change("((own_side,identity,.68),(1-own_side,adversary,.72))","((own_side,identity,.15),(1-own_side,adversary,.72))")
    change("_V5_VALUE_MIX = 0.48","_V5_VALUE_MIX = 0.29")
    code=code.replace('_ROOT_TEMP = 0.40','_ROOT_TEMP = 0.30')
    compile(code,'<elite-candidate>','exec')
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('weights',type=Path);p.add_argument('output',type=Path);p.add_argument('--mode',choices=['artemis','apollo','duo'],default='artemis');p.add_argument('--speed-source',type=Path)
    a=p.parse_args();a.output.write_text(build(a.weights,a.mode,a.speed_source.read_text() if a.speed_source else None))
