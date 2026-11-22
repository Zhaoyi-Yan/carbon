"""Tally the cost of losing an initial planter before it establishes any source of income."""
import argparse,json
from pathlib import Path
from branch_replay import step
from benchmark import new_game,configure_engine,limit_math_threads


def measure(games):
    result=[]
    for game in games:
        if game['status']!='completed':continue
        st,env=new_game(game['seed'],300)
        marks=[dict(planter=None,first_tree=None,death=None) for _ in (0,1)]
        for t,(moves,expected) in enumerate(game['actions_and_post_cash'][:75],1):
            step(st,env,moves);ps=st[0].observation.players
            for s,p in enumerate(ps):
                z=marks[s]
                if not z['planter']:
                    z['planter']=next((u for u,(_,_,k) in p[2].items() if k=='PLANTER'),None)
                if p[3] and z['first_tree'] is None:z['first_tree']=t
                if z['planter'] and z['planter'] not in p[2] and z['death'] is None:z['death']=t
        for s,m in enumerate(marks):
            result.append(dict(seed=game['seed'],seat=s,name=game['players'][s]['name'],won=s==game['winner'],**m,
                          died_first=m['death'] is not None and (m['first_tree'] is None or m['death']<m['first_tree'])))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('games',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();limit_math_threads();configure_engine()
    rows=measure([r for f in a.games for r in map(json.loads,f.open())]);a.out.write_text(json.dumps(rows,indent=2)+'\n')
    for n in ({r['name'] for r in rows}):
        for dead in (True,False):
            group=[r for r in rows if r['name']==n and r['died_first']==dead]
            print(n,dead,len(group),sum(r['won'] for r in group))
