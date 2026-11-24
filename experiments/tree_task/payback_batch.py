"""Fit on archived tree ownership, validate on unrelated games, and branch selected roots."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

import numpy as np

from run_task import (ROOT, PROXIES, REVISIONS, evaluate, configure_engine, read_replay,
                      build, write)
from benchmark import load_strategy
from payback import FEATURES, features, fit
from plan import planting_cost
from select_roots import options, order_key

MARGIN = 5.
CONFIDENCE = .65


def collect(path, match, focus, policy):
    e=read_replay(path)
    frames,cfg=e['steps'],e['configuration']
    examples=[]
    chances=options(e,focus,match)['left']
    by_turn=defaultdict(list)
    for c in chances:
        by_turn[c['turn']].append(c)
    initial=frames[0][0]['observation']['players']
    assert all(not p[3] for p in initial)
    layout=order_key(frames[0][0]['observation']['full_carbon'],[sorted(p[1].values()) for p in initial])
    for side in (0,1):
        tr=policy.BeliefTracker(policy._Bel,side)
        prev=None
        active={}
        for turn in range(len(frames)-1):
            before,after=(frames[t][0]['observation'] for t in (turn,turn+1))
            players=before['players']
            obs=dict(players=players,trees=before.get('trees',{}),step=turn,player=side,
                     carbon=frames[turn][side]['observation']['carbon'])
            estimate,known=tr.step(players,turn,obs['carbon'])
            estimate,known=policy._sense_tree_carbon(tr.bel,obs,cfg,prev,estimate,known)
            paid=planting_cost(players,side,cfg)
            if side==focus:
                for candidate in by_turn.get(turn,[]):
                    candidate['features']=features(players,estimate,side,candidate['cell'],turn,paid,cfg).tolist()
            old_own=players[side][3]
            old_enemy=players[1-side][3]
            now=after['players'][side][3]
            for uid in active.keys()-now.keys():
                active.pop(uid)
            for uid,(cell,_) in now.items():
                if uid not in old_own and uid not in old_enemy and 65<=turn<245:
                    row=dict(match=match,side=side,turn=turn,cell=cell,cost=paid,income=0.,
                             layout=layout, features=features(players,estimate,side,cell,turn,paid,cfg).tolist())
                    active[uid]=row
                    examples.append(row)
                if uid in active:
                    active[uid]['income']+=after['trees'][uid][1]
            tr.set_actions(players,frames[turn+1][side]['action'] or {})
            prev=players
    meta=dict(match=match,seat=focus,seed=cfg['randomSeed'],layout=layout,
              replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(meta=meta,examples=examples,chances=chances)


def extract(data,source):
    configure_engine()
    text=source.read_text().rsplit('\n_net()',1)[0]
    policy=load_strategy('payback_encoder',source,text.encode())
    rows=[]
    for match,side,opponent in json.loads((data/'games.json').read_text()):
        game=collect(data/(match+'-1.html.gz'),match,side,policy)
        game['meta']['opponent']=opponent
        rows.append(game)
    return rows


def auc(scores, labels):
    yes=[s for s,y in zip(scores,labels) if y]
    no=[s for s,y in zip(scores,labels) if not y]
    return round(mean(float(a>b)+.5*float(a==b) for a in yes for b in no),4)


def attach(pred, item):
    p,gross=pred.forecast(np.array(item['features']),item['cost'])
    return dict(**item,probability=p,projected_median_net=gross-item['cost'],
                selected=p>=CONFIDENCE and gross-item['cost']>=MARGIN)


def score(rows, prior):
    labels=[r['income']>=r['cost'] for r in rows]
    p=[r['probability'] for r in rows]
    eligible=[r for r in rows if r['selected']]
    return dict(trees=len(labels),repaid=sum(labels),
                brier=round(mean((v-y)**2 for v,y in zip(p,labels)),4),
                prior_brier=round(mean((prior-y)**2 for y in labels),4),
                auc=auc(p,labels),
                stock_auc=auc([r['features'][0] for r in rows],labels),
                eligible=len(eligible),eligible_repaid=sum(r['income']>=r['cost'] for r in eligible),
                by_seat=[sum(r['side']==s for r in rows) for s in (0,1)])


def prepare(source, holdout, out):
    out.mkdir(parents=True,exist_ok=False)
    for name,(tag,expected) in REVISIONS.items():
        raw=subprocess.check_output(['git','show',tag+':submission.py'],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==expected
        (out/(name+'.py')).write_bytes(raw)
    v8=out/'yan-v8.py'
    (out/'traced.py').write_text(build(v8.read_text(),expected_sha256=REVISIONS['yan-v8'][1]))
    train,hold=[extract(p,v8) for p in (source,holdout)]
    assert len({r['meta']['layout'] for r in train+hold})==len(train)+len(hold)
    training=[r for g in train for r in g['examples']]
    pred=fit(training)
    val=[attach(pred,r) for g in hold for r in g['examples']]
    prior=sum(r['income']>=r['cost'] for r in training)/len(training)
    cases=[]
    considered=passed=0
    for game in hold:
        m=game['meta']
        choices=[attach(pred,x) for x in game['chances']]
        considered+=len(choices)
        passed+=sum(x['selected'] for x in choices)
        if choices:
            # One decision per layout. Eligible tiles outrank diagnostic probes.
            best=max(choices,key=lambda x:(x['selected'],x['probability'],x['projected_median_net'],
                                           order_key(m['match'],x['turn'],x['unit'])))
            cases.append(dict(**m,**best,category='eligible' if best['selected'] else 'probe'))
    report=dict(model=pred.json(),features=FEATURES,thresholds=dict(min_probability=CONFIDENCE,min_net=MARGIN),
                frozen=REVISIONS,proxy_streams=PROXIES,train=[g['meta'] for g in train],
                holdout=[g['meta'] for g in hold],training_trees=len(training),prior=prior,
                offline=score(val,prior),considered=considered,passed=passed,
                cases=cases)
    write(out/'protocol.json',report)
    with gzip.open(out/'labels.json.gz','wt') as f:json.dump(dict(train=training,holdout=val),f)
    return cases


def aggregate(rows):
    summary={}
    for category in ('eligible','probe'):
        panel=[r for r in rows if r['case']['category']==category]
        if not panel:continue
        improvements=[]
        for row in panel:
            case=row['case']; own=case['seat']
            before,after=[row['arms'][a] for a in ('original','plant_once')]
            assert before['end']==after['end']==299
            assert '299' in before['marks'] and '299' in after['marks']
            born=after['target_id'] is not None and before['target_id'] is None
            improvements.append(dict(seed=case['seed'],turn=case['turn'],seat=own,proxy=row['proxy'],
                delta=round(after['margin']-before['margin'],2),
                target_net=round(after['marks']['299']['target_net']-before['marks']['299']['target_net'],2),
                new_tree=born,repaid=born and after['target_yield'][own]>=case['cost'],
                blocked=after['forced_blocked'],unchanged=before['trajectory']==after['trajectory']))
        group=improvements
        summary[category]=dict(maps=len(panel)//len(PROXIES),positive=sum(r['delta']>0 for r in group),
            negative=sum(r['delta']<0 for r in group),same=sum(r['unchanged'] for r in group),
            new_trees=sum(r['new_tree'] for r in group),repaid=sum(r['repaid'] for r in group),
            average_delta=round(mean(r['delta'] for r in group),2),
            average_target_net=round(mean(r['target_net'] for r in group),2),games=group)
    return summary


def main(source,holdout,out):
    cases=prepare(source,holdout,out)
    jobs=[(holdout,out,c,rival,salt,False,('original','plant_once')) for c in cases for rival,salt in PROXIES]
    with ProcessPoolExecutor(3) as pool,gzip.open(out/'branches.jsonl.gz','xt') as f:
        for row in pool.map(evaluate,jobs):
            f.write(json.dumps(row,default=lambda x:x.item())+'\n')
    # Evaluate the final saved form, including the JSON milestone keys.
    with gzip.open(out/'branches.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
    assert len(rows)==len(jobs)
    write(out/'results.json',aggregate(rows))


if __name__=='__main__':
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,*sys.argv])
    os.environ['CARBON_DEBUG']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
        os.environ[k]='1'
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','holdout','out'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    main(args.source.resolve(),args.holdout.resolve(),args.out.resolve())
