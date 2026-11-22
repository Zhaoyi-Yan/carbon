"""Select additional complete games by teacher coverage and loss of income before fitting."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('index','cohort','original','output'):p.add_argument(key,type=Path)
    p.add_argument('--limit',type=int,default=560)
    a=p.parse_args();base=json.loads(a.original.read_text())
    cohort={g['match']:g for g in json.loads(a.cohort.read_text())['matches']}
    rank={r['id']:r['rank'] for r in json.loads((Path(__file__).parent/'design.json').read_text())['ranking']}
    teachers={}
    for g in json.loads(a.index.read_text()):
        n=Path(g['file']).name.split('.')[0]
        if n in cohort:teachers[n]=[rank[x['id']] for x in g['strategies'].values() if x['id'] in rank]
    key=lambda n:hashlib.sha256(('carbon-league-income-'+cohort[n]['layout']).encode()).hexdigest()
    held={n for n in teachers if int(key(n)[:8],16)%6==0}
    va=sorted(held,key=key)[:80]
    eligible=set(teachers)-held
    special=sorted((n for n in eligible if cohort[n]['longest_lock']>=10),key=key)[:50]
    chosen=set(special);counts=collections.Counter(t for n in chosen for t in teachers[n])
    by_teacher={t:sorted((n for n in eligible if t in teachers[n]),key=key) for t in range(1,21)}
    positions=collections.Counter()
    while len(chosen)<min(a.limit,len(eligible)):
        available=[]
        for t,pool in by_teacher.items():
            while positions[t]<len(pool) and pool[positions[t]] in chosen:positions[t]+=1
            if positions[t]<len(pool):available.append(t)
        if not available:break
        t=min(available,key=lambda r:(counts[r]/(1.6 if r<=5 else 1.),r))
        n=by_teacher[t][positions[t]];chosen.add(n);counts.update(teachers[n])
    out=dict(training_matches=sorted(chosen)+base['training_matches'],validation_matches=va+base['validation_matches'],
             new_training=len(chosen),new_validation=len(va),training_teacher_seats=dict(counts),included_lockout_games=len(special))
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print('new games',len(chosen),len(va),'hard lockout games',len(special),flush=True)


if __name__=='__main__':
    main()
