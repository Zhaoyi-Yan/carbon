"""Check paired map identities and evaluate each frozen file against the declared rule."""
import argparse
import json
from pathlib import Path
from statistics import mean,median


def summarize(frozen,results):
    seeds=frozen['seeds']
    candidate=frozen['sha256'];baseline=frozen['opponent_sha256']
    pairs={};engine=set();differences=[[],[]];seen_layouts={};invalid=[]
    searches=[[],[]];latency=[[],[]];slow=[0,0]
    for r in results:
        if r['episode_steps']!=300:raise ValueError('Nonstandard game length')
        ids=[p['sha256'] for p in r['players']]
        if sorted(ids)!=sorted((candidate,baseline)) or r['seed'] not in seeds:raise ValueError('Unregistered game')
        seat=ids.index(candidate);key=r['seed'],seat
        if key in pairs:raise ValueError('Repeated map and seat')
        pairs[key]=r
        fingerprint=r['layout_sha256'];engine.add(r['engine_sha256'])
        if seen_layouts.setdefault(fingerprint,r['seed'])!=r['seed']:raise ValueError('Repeated layout under another seed')
        for i,k in enumerate((seat,1-seat)):
            p=r['players'][k]
            if p.get('searches') is not None:searches[i].append(p['searches'])
            if p.get('max_act_seconds') is not None:latency[i].append(p['max_act_seconds'])
            slow[i]+=p.get('slow_steps',0)
        if r['status']!='completed':
            invalid.append([*key,r['status']]);continue
        d=r['scores'][seat]-r['scores'][1-seat]
        expected=seat if d>0 else 1-seat if d<0 else None
        if expected!=r['winner'] or d!=r['cash'][seat]-r['cash'][1-seat]:raise ValueError('Inconsistent referee result')
        differences[seat].append(d)
    games=[]
    for seed in seeds:
        a=pairs.get((seed,0));b=pairs.get((seed,1))
        if a and b and a['map_sha256']!=b['map_sha256']:raise ValueError('Swap changed initial map')
        scores=[]
        for s,r in enumerate((a,b)):
            scores.append(round(r['scores'][s]-r['scores'][1-s],2) if r and r['status']=='completed' else None)
        games.append([seed,*scores])
    if len(engine)>1:raise ValueError('Mixed referees')
    wins=[sum(d>0 for d in seat) for seat in differences];all_d=differences[0]+differences[1]
    done=len(pairs)==2*len(seeds)
    return dict(candidate=frozen['candidate'],sha256=candidate,baseline_sha256=baseline,complete=done,
                games=games,wins=sum(wins),losses=sum(d<0 for d in all_d),draws=sum(d==0 for d in all_d),wins_by_seat=wins,
                average=round(mean(all_d),2) if all_d else None,median=round(median(all_d),2) if all_d else None,
                both_wins=sum(all(d is not None and d>0 for d in row[1:]) for row in games),
                both_losses=sum(all(d is not None and d<0 for d in row[1:]) for row in games),
                searches_range=[[min(s),max(s)] if s else None for s in searches],
                slow_steps=slow,maximum_move_seconds=[round(max(v),3) if v else None for v in latency],
                invalid=invalid,passed=bool(done and not invalid and sum(wins)>=12 and min(wins)>=5 and mean(all_d)>0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('freeze',type=Path);p.add_argument('result',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();report=summarize(json.loads(a.freeze.read_text()),[json.loads(l) for l in a.result.read_text().splitlines()])
    line=json.dumps(report,indent=2)+'\n'
    if a.output:a.output.write_text(line)
    print(line)
