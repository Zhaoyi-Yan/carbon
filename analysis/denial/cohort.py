"""Compare sustained pressure across distinct public top-player games without training on the two case studies."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path


def distance(a,b):
    dr,dc=abs(a//15-b//15),abs(a%15-b%15)
    return min(dr,15-dr)+min(dc,15-dc)


def measure(arg):
    name,path=arg
    r=json.loads(gzip.decompress(path.read_bytes()));frames=r['steps'];s=[f[0]['observation'] for f in frames]
    layout=hashlib.sha256(json.dumps([s[0]['full_carbon'],[sorted(p[1].values()) for p in s[0]['players']]],
                                    separators=(',',':')).encode()).hexdigest()
    rows=[];maxlock=0
    for side in (0,1):
        other=1-side;pressure=0;trailing=0;locked=0;longest=0
        for t,o in enumerate(s):
            pl=o['players'];opp=pl[other];us=pl[side]
            if us[0]<20 and not us[3] and not any(k=='COLLECTOR' for _,_,k in us[2].values()):
                locked+=1;longest=max(longest,locked)
            else:locked=0
            if 101<=t<=150:
                pressure+=any(k=='COLLECTOR' and v<30 and min(distance(p,b) for b in opp[1].values())<=2
                              for p,v,k in us[2].values())
                tree_income=sum(o['trees'][uid][1] for uid in opp[3])
                trailing+=int(opp[0]<100 and tree_income<8)
        outcome=s[-1]['players'][side][0]-s[-1]['players'][other][0]
        prev=s[100]['players']
        rows.append(dict(pressure_turns=pressure,constrained_turns=trailing,win=outcome>0,longest_lock=longest,
                         cash_margin_before=prev[side][0]-prev[other][0],
                         worker_margin_before=len(prev[side][2])-len(prev[other][2]),
                         tree_margin_before=len(prev[side][3])-len(prev[other][3])))
        maxlock=max(maxlock,longest)
    return dict(match=name,seed=int(r['configuration']['randomSeed']),layout=layout,seats=rows,longest_lock=maxlock)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();old=json.loads(a.manifest.read_text());seeds={g['seed'] for g in old};layouts={g['layout'] for g in old}
    here=Path(__file__).resolve().parents[2]/'experiments/v5'
    for file,key in [('design.json','qualification'),('design.json','development'),('confirmation.json','seeds'),('worker-confirmation.json','qualification')]:
        seeds.update(json.loads((here/file).read_text())[key])
    exempt={'0024bb667d204c1754568de71eb16170-1','9b1c90aa0357d7d62aa0095d6523219d-1'}
    paths=[(f.name.split('.')[0],f) for f in sorted((a.snapshot/'replays').glob('*.json.gz')) if f.name.split('.')[0] not in exempt]
    with ProcessPoolExecutor(2) as pool:items=list(pool.map(measure,paths,chunksize=4))
    games=[]
    for g in items:
        if g['seed'] in seeds or g['layout'] in layouts:continue
        seeds.add(g['seed']);layouts.add(g['layout']);games.append(g)
    report={'games':len(games),'hard_lock_games_at_least_10_steps':sum(g['longest_lock']>=10 for g in games),'groups':{}}
    for name,condition in [('all',lambda p:True),('pressure_only',lambda p:p['pressure_turns']>=35 and p['constrained_turns']<25),
                           ('pressure_and_constrained_income',lambda p:p['pressure_turns']>=35 and p['constrained_turns']>=25),
                           ('income_constraint_only',lambda p:p['pressure_turns']<35 and p['constrained_turns']>=25),
                           ('neither',lambda p:p['pressure_turns']<35 and p['constrained_turns']<25),
                           ('no_sustained_pressure',lambda p:p['pressure_turns']<35)]:
        seats=[p for g in games for p in g['seats'] if condition(p)]
        prebehind=[p for p in seats if p['cash_margin_before']<=0]
        report['groups'][name]=dict(seats=len(seats),wins=sum(p['win'] for p in seats),
            not_ahead_in_cash_at_100=len(prebehind),wins_not_ahead_at_100=sum(p['win'] for p in prebehind))
    report['matches']=[{k:v for k,v in g.items() if k!='seats'} for g in games]
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='matches'},flush=True)


if __name__=='__main__':
    main()
