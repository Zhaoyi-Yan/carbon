"""Evaluate the frozen, reactive actors against decisions on disjoint map layouts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from surrogate import ROOT, STYLE, make, _engine

TARGET_SEED=205864169
TARGET_LAYOUT='6121ec0ff5ccac020570cc3ac18f9dbe4ce60ddab3f00075aa1dd3afa8f15d49'


def metrics(c):
    def ratio(up,down):
        return round(c[up]/c[down],4) if c[down] else None
    return dict(worker_accuracy=ratio('worker_correct','workers'),workers=c['workers'],
                collector_accuracy=ratio('collector_correct','collectors'),collectors=c['collectors'],
                planter_accuracy=ratio('planter_correct','planters'),planters=c['planters'],
                loaded_close_accuracy=ratio('close_correct','close'),loaded_close=c['close'],
                eligible_base_accuracy=ratio('base_correct','base_opps'),eligible_bases=c['base_opps'],
                hire_precision=ratio('hire_overlap','hires_predicted'),hire_recall=ratio('hire_overlap','hires_actual'),
                hires_actual=c['hires_actual'],hires_predicted=c['hires_predicted'],
                correct_hire_role_on_overlap=ratio('hire_role_correct','hire_overlap'),
                actual_deposited=round(c['amount_actual'],2),predicted_deposited=round(c['amount_pred'],2),
                deposit_turn_precision=ratio('deposit_overlap','deposit_turns_predicted'),
                deposit_turn_recall=ratio('deposit_overlap','deposit_turns_actual'),
                deposit_turns_actual=c['deposit_turns_actual'],deposit_turns_predicted=c['deposit_turns_predicted'])


def score(m,pl,side,actual,pred,cfg):
    mine=pl[side];homes=set(mine[1].values());r=Counter()
    pa=pp=0.
    for uid,(pos,cargo,role) in mine[2].items():
        truth=actual.get(uid);guess=pred.get(uid)
        good=int(guess==truth)
        group=role.lower()
        r['workers']+=1;r['worker_correct']+=good;r[group+'s']+=1;r[group+'_correct']+=good
        if role=='COLLECTOR' and cargo>=40 and min(m.DIST[pos][q] for q in homes)<=3:
            r['close']+=1;r['close_correct']+=good
        if m.move(pos,m.WIDX.get(truth,0)) in homes:pa+=cargo
        if m.move(pos,m.WIDX.get(guess,0)) in homes:pp+=cargo
    if len(mine[2])<cfg['workerLimit'] and mine[0]>=min(cfg['recCollectorCost'],cfg['recPlanterCost']):
        for uid in mine[1]:
            truth=actual.get(uid);guess=pred.get(uid)
            r['base_opps']+=1;r['base_correct']+=truth==guess
            r['hires_actual']+=bool(truth);r['hires_predicted']+=bool(guess);r['hire_overlap']+=bool(truth and guess)
            r['hire_role_correct']+=bool(truth and truth==guess)
    r['amount_actual']=pa;r['amount_pred']=pp;r['deposit_turns_actual']+=pa>0
    r['deposit_turns_predicted']+=pp>0;r['deposit_overlap']+=pa>0 and pp>0
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical',type=Path,required=True);p.add_argument('--recent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();m=_engine()
    selection=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())
    training={i.split('-')[0] for i in selection['training']}
    valid={i.split('-')[0] for i in selection['validation']}
    historical=json.load((args.historical/'index.json').open())
    latest=json.load((args.recent/'index.json').open())
    matches=[(r,args.historical/r['file'],'checkpoint_validation') for r in historical if r['id'] in valid]
    matches.extend((r,args.recent/r['file'],'post_training') for r in latest['rows'] if (args.recent/r['file']).is_file())
    seen=set();groups=defaultdict(Counter);records=[];ms=[]
    for row,path,cohort in matches:
        if row['id'] in training:raise ValueError('Training game in action audit')
        game=json.load(gzip.open(path))
        cfg=game['configuration']
        first=game['steps'][0][0]['observation']
        compact=[first['full_carbon'],[sorted(p[1].values()) for p in first['players']]]
        layout=hashlib.sha256(json.dumps(compact,separators=(',',':')).encode()).hexdigest()
        layout_sz=hashlib.sha256(json.dumps([cfg['size']]+compact,separators=(',',':')).encode()).hexdigest()
        if cfg['randomSeed']==TARGET_SEED or TARGET_LAYOUT in (layout,layout_sz):
            continue
        if layout in seen:continue
        seen.add(layout)
        for side,slot in enumerate(row['order']):
            teacher=next((k for k,(uid,_) in STYLE.items() if uid==row['strategies'][slot]['id']),None)
            if teacher is None:continue
            f=game['steps'];assert game['rewards']==[row['scores'][s] for s in row['order']]
            actors={k:make(k,31,cfg) for k in STYLE}
            sums=defaultdict(Counter)
            for t,fr in enumerate(f[:-1]):
                state=fr[0]['observation'];pl=state['players'];act=f[t+1][side]['action'] or {}
                obs=dict(players=pl,step=t,player=side,carbon=fr[side]['observation']['carbon'],
                         trees=state['trees'],remainingOverageTime=fr[side]['observation'].get('remainingOverageTime',5))
                for actor,proxy in actors.items():
                    time0=time.perf_counter();pred=proxy.predict(obs);ms.append((time.perf_counter()-time0)*1000)
                    # Teacher forcing: restore the real previous action before the NEXT step.
                    proxy._finish(obs,act)
                    s=score(m,pl,side,act,pred,cfg)
                    sums['all',actor].update(s)
                    if t>=190:sums['190+',actor].update(s)
            for (phase,actor),count in sums.items():groups[cohort,teacher,phase,actor].update(count)
            records.append(dict(match=row['id'],game=row['game'],cohort=cohort,teacher=teacher,
                                seat=side,seed=cfg['randomSeed'],layout=layout,
                                metrics_190={actor:metrics(sums['190+',actor]) for actor in STYLE}))
    summary=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for (cohort,teacher,phase,actor),counts in groups.items():summary[cohort][teacher][phase][actor]=metrics(counts)
    report=dict(target_excluded_seed=TARGET_SEED,target_excluded_layout=TARGET_LAYOUT,style_mapping=STYLE,
                checkpoint='experiments/v6r/checkpoints/grove.py',
                checkpoint_sha256=hashlib.sha256((ROOT/'experiments/v6r/checkpoints/grove.py').read_bytes()).hexdigest(),
                surrogate_sha256=hashlib.sha256((ROOT/'experiments/v7/surrogate.py').read_bytes()).hexdigest(),
                recent_snapshot=latest['revision'],source='https://github.com/JUZI1/carbon-league-replays',
                declared_net_training_games=len(training),declared_net_epoch_selection_games=len(valid),
                notes='The 12 old validation games were used to choose the neural checkpoint epoch. Post-training layouts are the independent test. Each action uses only the observer field of view. Delivery counts measure own carried carbon reaching an own depot on that one observed turn. Every next turn resets the action history to the expert action. These metrics do not certify multi-step or head-to-head equivalence.',
                phase_190_first_decision=190,median_predict_ms=round(median(ms),3),groups=summary,games=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')
    print('games',len(records),'counts',Counter((r['cohort'],r['teacher']) for r in records),
          'median ms',report['median_predict_ms'],flush=True)
    for cohort in summary:
        if 'artemis' in summary[cohort]:
            for actor,v in summary[cohort]['artemis']['190+'].items():print(cohort,actor,v,flush=True)

if __name__=='__main__':main()
