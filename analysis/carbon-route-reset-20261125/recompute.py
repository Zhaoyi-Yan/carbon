"""Read-only M/G two-seed and M-wide evidence audit at e8a4151.
Usage: python recompute.py /path/to/carbon-checkout statistics.json
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np

SOURCE = 'e8a4151825b79d793b14cad85463765ba2aec9b3'
RNG_SEED = 20261124
REPLICATES = 100000
RIVALS = ('ZW0', 'ft3-generic-no-search', 'yan-v6-no-search')
CANDIDATES = ('ZW0', 'M50', 'G50', 'M50_s2', 'G50_s2', 'Mwide_s1')
RUNS = ('M-50k', 'G-50k', 'M-50k-s2', 'G-50k-s2', 'Mwide-s1', 'smoke-M-10k', 'smoke-G-10k')


def normalise(x):
    if isinstance(x,dict):return {str(k):normalise(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [normalise(v) for v in x]
    if isinstance(x,np.ndarray):return normalise(x.tolist())
    if isinstance(x,np.generic):return normalise(x.item())
    if isinstance(x,float) and not np.isfinite(x):raise ValueError('Nonfinite output')
    return x


def boot(x):
    x=np.asarray(x,float)
    idx=np.random.default_rng(RNG_SEED).integers(len(x),size=(REPLICATES,len(x)))
    return dict(mean=x.mean(),percentile95=np.quantile(x[idx].mean(1),[.025,.975]),maps=len(x))


def summary(rows):
    d=np.asarray([x['margin'] for x in rows])
    return dict(games=len(rows),wins=np.sum(d>0),draws=np.sum(d==0),losses=np.sum(d<0),
                mean_margin=d.mean(),mean_utility=np.mean([x['utility'] for x in rows]),
                eliminations=sum(x['elimination'] for x in rows),E=sum(x['turns'] for x in rows))


def check_game(g):
    delta=g['scores'][g['seat']]-g['scores'][1-g['seat']]
    u=np.sign(delta) if g['elimination'] else .5*np.sign(delta)+.5*np.tanh(delta/1500)
    assert abs(delta-g['margin'])<1e-9 and abs(u-g['utility'])<1e-12


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repository',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();repo=args.repository.resolve();folder=repo/'analysis/carbon-zero-terminal/b1/mg'
    paths=sorted(folder.glob('**/*.json'));sha={}
    for f in paths:
        rel=str(f.relative_to(repo));content=f.read_bytes()
        frozen=subprocess.run(['git','-C',str(repo),'show',f'{SOURCE}:{rel}'],check=True,stdout=subprocess.PIPE).stdout
        assert content==frozen,rel
        sha[rel]=hashlib.sha256(content).hexdigest()
    load=lambda x:json.loads((folder/x).read_text())
    result=dict(source_commit=SOURCE,input_sha256=sha,
                method=dict(numpy=np.__version__,python=platform.python_version(),rng_seed=RNG_SEED,bootstrap_replicates=REPLICATES,
                            unit='map; retain all rivals, seats, and fixed training seeds within each sampled map',
                            interval='exploratory unadjusted 95% percentile bootstrap',
                            limits=['All candidates share the same 12 development maps; there are not 24 independent maps after seed averaging.',
                                    'Intervals condition on these specific trained checkpoints, not the distribution of all training seeds.',
                                    'Failure of the prespecified budget gate stops this configuration; it does not establish equivalence or refute RL in general.',
                                    'One M-wide seed jointly reallocates maps and repetitions; it cannot rule out map coverage as a general bottleneck.']),
                evaluation={},paired={},training={},resources={},gates={},audit={})
    protocol=load('evaluation/protocol.json');assert protocol['shared_maps'] and set(protocol['panels'])==set(RIVALS)
    data={n:load(f'evaluation/{n}.json') for n in CANDIDATES};games={}
    all_eval=[]
    for n in CANDIDATES:
        assert set(data[n]['rivals'])==set(RIVALS)
        games[n]={};result['evaluation'][n]={'checkpoint_sha256':data[n]['checkpoint_sha256'],'rivals':{}}
        for rival in RIVALS:
            rows=data[n]['rivals'][rival]['games'];games[n][rival]={(g['seed'],g['seat']):g for g in rows}
            assert len(games[n][rival])==24
            expected={(m['seed'],m['seat']):m for m in protocol['panels'][rival]}
            assert set(games[n][rival])==set(expected)
            for g in rows:
                check_game(g)
                for k in ('learner_rng','opponent_rng','group'):assert g[k]==expected[(g['seed'],g['seat'])][k]
            result['evaluation'][n]['rivals'][rival]=dict(overall=summary(rows),seats={str(t):summary([g for g in rows if g['seat']==t]) for t in (0,1)})
            all_eval.extend(rows)
    keys=sorted(games['ZW0']['ZW0']);seeds=sorted({s for s,t in keys});assert len(seeds)==12
    for n in CANDIDATES:
        for rival in RIVALS:
            assert set(games[n][rival])==set(keys)
            for key in keys:
                for field in ('learner_rng','opponent_rng','layout_sha256'):
                    assert games[n][rival][key][field]==games['ZW0'][rival][key][field]
                assert games[n][rival][key]['layout_sha256']==games['ZW0']['ZW0'][key]['layout_sha256']
    comparisons=[(n,[n],['ZW0']) for n in CANDIDATES if n!='ZW0']
    comparisons += [('M_s1_minus_G_s1',['M50'],['G50']),('M_s2_minus_G_s2',['M50_s2'],['G50_s2']),
                    ('M_two_seed_mean_minus_ZW0',['M50','M50_s2'],['ZW0']),
                    ('G_two_seed_mean_minus_ZW0',['G50','G50_s2'],['ZW0']),
                    ('M_two_seed_mean_minus_G_two_seed_mean',['M50','M50_s2'],['G50','G50_s2']),
                    ('Mwide_s1_minus_M_s1',['Mwide_s1'],['M50'])]
    for label,left,right in comparisons:
        metrics={}
        for metric in ('utility','margin','win_score'):
            val=(lambda g:float(g['margin']>0)+.5*float(g['margin']==0)) if metric=='win_score' else lambda g:g[metric]
            arrays={r:np.asarray([[np.mean([val(games[n][r][(s,t)]) for n in left])-np.mean([val(games[n][r][(s,t)]) for n in right]) for t in (0,1)] for s in seeds]) for r in RIVALS}
            macro=np.stack([arrays[r] for r in RIVALS]).mean(0)
            metrics[metric]={r:boot(x.mean(1)) for r,x in arrays.items()}
            metrics[metric]['macro']=boot(macro.mean(1))
            if metric=='utility':
                metrics[metric]['macro_seats']={str(t):boot(macro[:,t]) for t in (0,1)}
        result['paired'][label]=metrics
    train=[];smoke=[];new=[]
    all_costs=[]
    for run in RUNS:
        ep=load(f'{run}/episodes.json');up=load(f'{run}/updates.json');pro=load(f'{run}/protocol.json')
        for g in ep:check_game(g)
        assert len(ep)==16*len(up) and sum(g['turns'] for g in ep)==up[-1]['E']==sum(u['batch_E'] for u in up)
        costs=dict(collection_seconds=sum(u['collection']['elapsed'] for u in up),
                   optimization_seconds=sum(u['optimization_seconds'] for u in up),
                   recorded_baseline_seconds=sum(u.get('baseline_seconds',0) for u in up),
                   baseline_time_recorded_in_all_batches=all('baseline_seconds' in u for u in up))
        result['training'][run]=dict(seed=pro['seed'],**summary(ep),batches=len(up),maps=len({g['seed'] for g in ep}),
                                    optimizer_steps=sum(u['optimization_updates'] for u in up),costs=costs)
        all_costs.append(costs)
        (smoke if run.startswith('smoke') else train).extend(ep)
        if run in ('M-50k-s2','G-50k-s2','Mwide-s1'):new.extend(ep)
    result['resources']={
        'five_main_runs':summary(train),'two_smoke_records':summary(smoke),
        'latest_three_new_runs':summary(new),'evaluation_all_six_candidates':summary(all_eval),
        'training_E_including_separately_recorded_smokes':sum(g['turns'] for g in train+smoke),
        'training_plus_evaluation_E_including_smokes':sum(g['turns'] for g in train+smoke+all_eval),
        'training_logged_collection_plus_optimization_seconds':sum(c['collection_seconds']+c['optimization_seconds'] for c in all_costs),
        'training_additionally_logged_baseline_seconds':sum(c['recorded_baseline_seconds'] for c in all_costs),
        'time_note':'Not end-to-end runtime. Older runs omit baseline/advantage construction; wall-clock overlap between runs is unknown.',
        'scope':'Only M/G and M-wide artifacts in this mg directory; excludes prior PPO/B1/critic work. Smoke prefixes repeat main prefixes and are not statistical replications.'}
    wide=result['paired']['Mwide_s1_minus_M_s1']['utility'];wide0=result['paired']['Mwide_s1']['utility']
    result['gates']={
        'M_two_seed_confirmation_eligible':result['paired']['M50']['utility']['macro']['mean']>0 and result['paired']['M50_s2']['utility']['macro']['mean']>0,
        'G_two_seed_confirmation_eligible':result['paired']['G50']['utility']['macro']['mean']>0 and result['paired']['G50_s2']['utility']['macro']['mean']>0,
        'wide_gt_same_seed_M':wide['macro']['mean']>0,'wide_gt_ZW0':wide0['macro']['mean']>0,
        'wide_no_opponent_upper95_below_zero_vs_M':all(wide[r]['percentile95'][1]>=0 for r in RIVALS)}
    result['audit']['all_evaluation_pairing_and_utility_checks_passed']=True
    result=normalise(result);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(f'Wrote {args.output}, {len(sha)} frozen hashed inputs.')
    for name,v in result['paired'].items():print(name,v['utility']['macro'])
    print('resources',result['resources']);print('gates',result['gates'])


if __name__=='__main__':main()
