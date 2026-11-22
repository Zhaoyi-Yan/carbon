"""Use referee bank events to distinguish direct gifts from cargo captured in battle."""
import argparse,json
from pathlib import Path
from collections import Counter
from branch_replay import step
from benchmark import configure_engine,limit_math_threads,new_game
from close_replay import referee_events


def run(files):
    cases=[]
    for path in files:
        for g in map(json.loads,path.open()):
            if g['status']!='completed':continue
            st,env=new_game(g['seed'],300);rows=[]
            for t,(moves,expected) in enumerate(g['actions_and_post_cash']):
                before=st[0].observation.players;workers=[{u:(p,c,k) for u,(p,c,k) in a[2].items()} for a in before]
                events=[]
                with referee_events(events):step(st,env,moves)
                assert expected==[p[0] for p in st[0].observation.players]
                for e in events:
                    if e[0]=='bank' and e[1]!=e[4]:
                        _,benefit,total,uid,payer,home=e;previous=workers[payer].get(uid)
                        direct=min(total,previous[1]) if previous else 0.
                        rows.append(dict(turn=t+1,beneficiary=benefit,payer=payer,amount=round(total,3),carried_before=round(direct,3),uid=uid))
            cases.append(dict(seed=g['seed'],names=[p['name'] for p in g['players']],gifts=rows))
    return cases


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();limit_math_threads();configure_engine();d=run(a.files);a.out.write_text(json.dumps(d,indent=2)+'\n')
    for name in {n for r in d for n in r['names']}:
        outcomes=[e['carried_before'] for r in d for e in r['gifts'] if r['names'][e['payer']]==name]
        print(name,len(outcomes),round(sum(outcomes)),sum(x>=30 for x in outcomes),round(max(outcomes or [0.])))
