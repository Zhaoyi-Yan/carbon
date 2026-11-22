"""Summarize whole-map paired post-search changes and their realized fixed-command outcomes."""
import argparse,gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

FIELDS=('cash_margin','own_deposits','enemy_deposits','own_cargo_lost','enemy_cargo_lost','held_cargo_margin','unmatched_worker_turns')
COMPARISONS=(('legacy','searched'),('corrected','legacy'),('veto','legacy'),('searched','legacy'))
def mean(a):
    return dict(zip(FIELDS,np.asarray(a).round(4).tolist()))
def table(games,x,y):
    changed=[];allrows=[];allmaps={r['map']:[] for r in games}
    for r in games:
        for t in r['data']:
            same=t['changes'][x]==t['changes'][y]
            if same:d=np.zeros((3,7))
            else:
                d=np.stack([np.array(t['outcomes'][x][h])-t['outcomes'][y][h] for h in ('1','5','10')]);changed.append((r['map'],d))
            allrows.append(d);allmaps[r['map']].append(d)
    per=np.stack([np.mean(rows,0) if rows else np.zeros((3,7)) for rows in allmaps.values()])
    all_=np.stack(allrows);delta=np.stack([d for _,d in changed]) if changed else np.zeros((1,3,7))
    rng=np.random.RandomState(19531);boot=per[rng.randint(len(per),size=(10000,len(per)))].mean(1)
    result={}
    for j,h in enumerate((1,5,10)):
        ci=np.quantile(boot[:,j],[.025,.975],axis=0)
        result[h]=dict(mean_changed=mean(delta[:,j].mean(0)),mean_across_all_roots=mean(all_[:,j].mean(0)),
                       map_mean=mean(per[:,j].mean(0)),map_paired_ci95={k:v for k,v in zip(FIELDS,ci.T.round(4).tolist())},
                       maps_positive_cash=int((per[:,j,0]>.001).sum()),maps_negative_cash=int((per[:,j,0]<-.001).sum()))
    return dict(roots=len(allrows),changes=len(changed),maps_with_changes=len({g for g,d in changed}),maps=len(games),horizons=result)
def details(games,public):
    index={r['match']:r for r in json.loads((public/'index.json').read_text())};out=[]
    for row in games:
        with gzip.open(public/index[row['match']]['file']) as f:g=json.load(f)
        for t in row['data']:
            if not t.get('outcomes'):continue
            step=t['t'];s=row['side'];pl=g['steps'][step][0]['observation']['players'];enemy=g['steps'][step+1][1-s]['action'] or {}
            focal={a['pos'] for changes in t['changes'].values() for a in changes}
            def close(q):
                return any(min(abs(q%15-p%15),15-abs(q%15-p%15))+min(abs(q//15-p//15),15-abs(q//15-p//15))<=2 for p in focal)
            orders={uid:[p,round(v,3),k,enemy.get(uid,'NONE')] for uid,(p,v,k) in pl[1-s][2].items() if close(p)}
            out.append(dict(match=row['match'],layout=row['map'],seed=row['config_seed'],seat=s,fold=row['fold'],rival_orders_near_changed_worker=orders,**t))
    return out

def main(a):
    games=[json.loads(p.read_text()) for p in sorted(a.data.glob('*.json'))];assert len(games)==21
    changes=details(games,a.public)
    with gzip.open(a.changes,'wt') as f:json.dump(changes,f,separators=(',',':'))
    report=dict(parent_sha=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
                design='Seven development and fourteen confirmation layouts; all independent of the frozen receipts training corpus, both screenshot maps and v7 panels. From 30 onward select at most 35 contact observations per map by deterministic hash/time strata, any collector on either side carrying at least 45 within two of another. No outcome used to pick frames.',
                search='Reference Pilotage root search width 4, depth 2, one rival reply, 3 tail plies; every counted state completed a search. Use acting seat fog and teacher-forced own commands on each preceding frame; expensive root style estimator evaluated at selected states and their preceding states. Final tactical evaluator has identical opponent/candidate list and no wall-clock truncation between versions.',
                methods={'searched':'Joint action coming out of search before collision reassessment.','legacy':'Stock distance-only cargo valuation and original lower-quartile collision selection.',
                         'corrected':'Revalue all tactical options using estimated conditional arrival of current cargo by 5/10 plies, plus discounted residual. Excludes future mined stock and preserves cash, fleet, trees and elimination terms.',
                         'veto':'Allow the stock chooser to override the search only if that same alternative beats the searched joint by six units under the revised score.'},
                measures=FIELDS,accounting='Original observed rival command on the first transition, replayed commands fixed for the next nine plies. Initial ID counters and full stocks reconstructed from the original start. Later workers mapped by persistent identity, same-turn same-base same-role births. Own-team cargo transfers excluded from loss; deposits to an enemy depot, enemy looting or destruction count as lost. The alternate future is a controlled counterfactual, not observed live adaptation.',
                frozen_rebuild_max_cash_error=max(g['cash_rebuild_max_error'] for g in games),
                parity_checks=sum(t['exact_old_parity'] for r in games for t in r['data']),
                maps=[{k:v for k,v in r.items() if k!='data'} for r in games],
                roots=sum(len(g['data']) for g in games),change_log=a.changes.name,unique_changed_states=len(changes),
                estimate_ms_per_leaf_p50_p95=np.percentile([t['clock_ms'] for r in games for t in r['data']],[50,95]).round(4).tolist(),
                results={f:{x+'_vs_'+y:table([r for r in games if f=='all' or r['fold']==f],x,y) for x,y in COMPARISONS} for f in ('dev','confirm','all')})
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print('roots',report['roots'],'parity',report['parity_checks'])
    for f,vals in report['results'].items():
        for k,r in vals.items():
            print(f,k,r['changes'],r['maps_with_changes'],[r['horizons'][h]['mean_changed']['cash_margin'] for h in (1,5,10)],r['horizons'][10]['map_paired_ci95']['cash_margin'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('data',type=Path);p.add_argument('public',type=Path);p.add_argument('changes',type=Path);p.add_argument('report',type=Path)
    main(p.parse_args())
