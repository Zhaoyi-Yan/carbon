"""Read-only statistics for Carbon M/G at be792f64; no model training.
Usage: python stats_recompute.py /path/to/carbon-mg-checkout statistics.json
"""
import argparse
import hashlib
import json
import itertools
import platform
import subprocess
from pathlib import Path
import numpy as np

SOURCE_COMMIT = 'be792f64e3b25b392b1cd502e5b55cfbf11e0950'
BOOTSTRAP_SEED = 20261124
REPLICATES = 100000
RIVALS = ('ZW0', 'ft3-generic-no-search', 'yan-v6-no-search')
ARMS = ('ZW0', 'M50', 'G50')


def normalise(x):
    if isinstance(x, dict): return {str(k): normalise(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [normalise(v) for v in x]
    if isinstance(x, np.ndarray): return normalise(x.tolist())
    if isinstance(x, np.generic): return normalise(x.item())
    if isinstance(x, float) and not np.isfinite(x): raise ValueError('Nonfinite output scalar')
    return x


def bootstrap(values):
    a = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(a), size=(REPLICATES, len(a)))
    dist = a[draws].mean(1)
    leave_one = (a.sum()-a)/(len(a)-1)
    return dict(mean=a.mean(), percentile95=np.quantile(dist,[.025,.975]), maps=len(a),
                map_sd=a.std(ddof=1), positive_maps=np.sum(a>0), zero_maps=np.sum(a==0),
                leave_one_map_out_range=[leave_one.min(),leave_one.max()])


def stats(rows):
    margin=np.asarray([r['margin'] for r in rows],float)
    return dict(games=len(rows), wins=np.sum(margin>0), draws=np.sum(margin==0), losses=np.sum(margin<0),
                mean_margin=margin.mean() if len(rows) else None,
                median_margin=np.median(margin) if len(rows) else None,
                mean_utility=np.mean([r['utility'] for r in rows]) if len(rows) else None,
                full_games=sum(not r['elimination'] for r in rows),
                eliminations=sum(r['elimination'] for r in rows))


def validate_row(row):
    d=row['scores'][row['seat']]-row['scores'][1-row['seat']]
    assert np.isclose(d,row['margin'],rtol=0,atol=1e-9)
    u=np.sign(d) if row['elimination'] else .5*np.sign(d)+.5*np.tanh(d/1500)
    assert np.isclose(u,row['utility'],rtol=0,atol=1e-12)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repository',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();repo=a.repository.resolve()
    folder=repo/'analysis/carbon-zero-terminal/b1/mg'
    paths=sorted(folder.glob('**/*.json'))
    extra=[repo/'analysis/carbon-zero-terminal/b1/evaluation-shared/ZW0.json',
           repo/'analysis/carbon-zero-terminal/b1/evaluation/ZW0.json',
           repo/'analysis/carbon-zero-terminal/b1/critic/critic.json',
           repo/'analysis/carbon-zero-terminal/critic/critic.json',
           repo/'analysis/carbon-zero-terminal/critic-expanded/critic.json',
           repo/'analysis/carbon-zero-terminal/evaluation/ZW0.json',
           repo/'analysis/carbon-zero-terminal/confirmation/ZW0.json']
    paths+=extra
    hashes={}
    for path in paths:
        rel=str(path.relative_to(repo)); content=path.read_bytes()
        committed=subprocess.run(['git','-C',str(repo),'show',f'{SOURCE_COMMIT}:{rel}'],check=True,stdout=subprocess.PIPE).stdout
        assert content==committed, f'Input differs from frozen snapshot: {rel}'
        hashes[rel]=hashlib.sha256(content).hexdigest()
    load=lambda rel: json.loads((folder/rel).read_text())
    result={'source_commit':SOURCE_COMMIT,'input_sha256':hashes,
            'method':dict(python_version=platform.python_version(),numpy_version=np.__version__,
                          rng='numpy.default_rng PCG64 reset for each same-sized bootstrap',seed=BOOTSTRAP_SEED,
                          replicates=REPLICATES,interval='unadjusted exploratory 95% percentile bootstrap',
                          sampling_unit='map; preserve two seats and all three rivals in the same map cluster',
                          primary='equal-weight mean paired delta-U across the three rivals',
                          limitations=['One M and one G training seed at this snapshot; second-seed results absent.',
                                       'Bootstrap conditions on these fixed checkpoints, not training-seed randomness.',
                                       'New development maps differ from the previous plan to reuse the old shared panel.',
                                       'The continuation gate allocates research budget, not a statistical promotion.']),
            'evaluation':{},'paired':{},'training':{},'audit':{},'selection':{}}
    protocol=load('evaluation/protocol.json')
    assert protocol['shared_maps'] is True and set(protocol['panels'])==set(RIVALS)
    panels={n:load(f'evaluation/{n}.json') for n in ARMS}
    game_index={}
    for n,panel in panels.items():
        assert set(panel['rivals'])==set(RIVALS)
        result['evaluation'][n]={'checkpoint_sha256':panel['checkpoint_sha256'],'rivals':{}}
        game_index[n]={}
        for rival in RIVALS:
            rows=panel['rivals'][rival]['games']
            game_index[n][rival]={(g['seed'],g['seat']):g for g in rows}
            assert len(game_index[n][rival])==len(rows)
            proto={(g['seed'],g['seat']):g for g in protocol['panels'][rival]}
            assert set(proto)==set(game_index[n][rival])
            for row in rows:
                validate_row(row)
                for key in ('learner_rng','opponent_rng','group'):
                    assert row[key]==proto[(row['seed'],row['seat'])][key]
            summary=stats(rows)
            original=panel['rivals'][rival]['overall']
            for key in original:
                assert np.isclose(summary[key],original[key],rtol=0,atol=1e-9),(n,rival,key)
            result['evaluation'][n]['rivals'][rival]={
                'overall':summary,'seats':{str(seat):stats([g for g in rows if g['seat']==seat]) for seat in (0,1)}}
        result['evaluation'][n]['all_rivals_pooled']=stats([g for r in RIVALS for g in panel['rivals'][r]['games']])
    keys=sorted(game_index['ZW0']['ZW0']);seeds=sorted({k[0] for k in keys})
    assert len(seeds)==12 and len(keys)==24
    for n in ARMS:
        for rival in RIVALS:
            assert set(game_index[n][rival])==set(keys)
            for k in keys:
                row=game_index[n][rival][k]
                assert row['layout_sha256']==game_index['ZW0']['ZW0'][k]['layout_sha256']
                for field in ('learner_rng','opponent_rng','layout_sha256'):
                    assert row[field]==game_index['ZW0'][rival][k][field]
    for x,y in (('M50','ZW0'),('G50','ZW0'),('M50','G50')):
        pair={}
        for metric in ('utility','margin','win_score'):
            read=((lambda g:float(g['margin']>0)+.5*float(g['margin']==0)) if metric=='win_score' else lambda g:g[metric])
            deltas={r:np.asarray([[read(game_index[x][r][(s,t)])-read(game_index[y][r][(s,t)]) for t in (0,1)] for s in seeds]) for r in RIVALS}
            out={r:{'both_seats':bootstrap(v.mean(1)),
                    'seat0':bootstrap(v[:,0]),'seat1':bootstrap(v[:,1])} for r,v in deltas.items()}
            macro=np.stack([deltas[r] for r in RIVALS]).mean(0)
            out['macro']={'both_seats':bootstrap(macro.mean(1)),
                          'seat0':bootstrap(macro[:,0]),'seat1':bootstrap(macro[:,1])}
            if metric=='utility':out['macro']['map_values']=dict(zip(map(str,seeds),macro.mean(1)))
            pair[metric]=out
        result['paired'][x+'-'+y]=pair
    histories={};logs={}
    for run in ('M-50k','G-50k','smoke-M-10k','smoke-G-10k'):
        ep=load(f'{run}/episodes.json');up=load(f'{run}/updates.json');pro=load(f'{run}/protocol.json')
        for row in ep:
            validate_row(row)
            assert len(row['gaps'])==row['turns']
        assert len(ep)==16*len(up)
        assert sum(g['turns'] for g in ep)==up[-1]['E']==sum(u['batch_E'] for u in up)
        for i,u in enumerate(up):
            assert u['batch_E']==sum(g['turns'] for g in ep[i*16:(i+1)*16])
            assert u['E']==sum(v['batch_E'] for v in up[:i+1])
        summaries={}
        for key in ('policy','old_KL','clip_rate','gradient_norm','logprob_error'):
            vals=np.asarray([u[key] for u in up])
            summaries[key]=dict(min=vals.min(),max=vals.max(),batch_mean=vals.mean())
        result['training'][run]={
            'seed':pro['seed'],'requested_budget':pro['budget'],'actual_E':up[-1]['E'],
            'batches':len(up),'optimizer_steps':sum(u['optimization_updates'] for u in up),
            'maps':len({g['seed'] for g in ep}),'overall':stats(ep),
            'seats':{str(t):stats([g for g in ep if g['seat']==t]) for t in (0,1)},
            'full_only':stats([g for g in ep if not g['elimination']]),
            'eliminations':[{k:v for k,v in g.items() if k!='gaps'} for g in ep if g['elimination']],
            'log_ranges':summaries,
            'collection_seconds':sum(u['collection']['elapsed'] for u in up),
            'optimization_seconds':sum(u['optimization_seconds'] for u in up),
            'stored_advantage_fields':list(up[0]['advantages']),
            'advantage_note':'Stored mean_mc/mean_gae are per-game means averaged in a batch; signed means do not identify variance or RMS.',
            'signed_mean_mc_across_batches':np.mean([u['advantages']['mean_mc'] for u in up]),
            'signed_mean_gae_across_batches':np.mean([u['advantages']['mean_gae'] for u in up]),
        }
        histories[run]=ep;logs[run]=up
    ma,ga=histories['M-50k'],histories['G-50k']
    matching=[]
    for i in range(len(logs['M-50k'])):
        m,g=ma[16*i:16*(i+1)],ga[16*i:16*(i+1)]
        stream_match=all(all(x[k]==y[k] for k in ('seed','seat','group','learner_rng','opponent_rng','layout_sha256')) for x,y in zip(m,g))
        assert stream_match
        matching.append(dict(batch=i,same_layout_seat_rng=stream_match,
                             identical_complete_episode_records=sum(x==y for x,y in zip(m,g))))
    assert ma[:16]==ga[:16]
    result['audit']['training_pairing']=matching
    result['audit']['smoke_is_exact_episode_prefix']={arm:histories[f'smoke-{arm}-10k']==histories[f'{arm}-50k'][:48] for arm in ('M','G')}
    result['audit']['first_batch_evidence_scope']='Episode records and public cash-gap sequences identical; complete action-trace hash was not committed in this directory.'
    ev_layouts={g['layout_sha256'] for g in panels['ZW0']['rivals']['ZW0']['games']}
    other_sets={}
    for path in extra:
        d=json.loads(path.read_text());rel=str(path.relative_to(repo))
        if 'rivals' in d:
            other_sets[rel]={g['layout_sha256'] for r in d['rivals'].values() for g in r['games']}
        else:
            for split in ('training_games','heldout_games'):
                other_sets[rel+':'+split]={g['layout_sha256'] for g in d[split]}
    other_sets['current_M_G_training']={g['layout_sha256'] for g in ma+ga}
    result['audit']['evaluation_layout_overlap']={n:{'source_maps':len(s),'intersection':len(s&ev_layouts)} for n,s in other_sets.items()}
    result['audit']['evaluation_pairing_valid']=True
    for name in ('M50','G50'):
        u=result['paired'][name+'-ZW0']['utility']
        result['selection'][name]={
            'stored_rule':load(('M' if name=='M50' else 'G')+'-50k/protocol.json')['selection'],
            'macro_delta_U':u['macro']['both_seats']['mean'],
            'opponent_upper95':{r:u[r]['both_seats']['percentile95'][1] for r in RIVALS},
            'passes_budget_screen':u['macro']['both_seats']['mean']>0 and all(u[r]['both_seats']['percentile95'][1]>=0 for r in RIVALS),
        }
    values=np.asarray(list(result['paired']['M50-ZW0']['utility']['macro']['map_values'].values()))
    leave_one=(values.sum()-values)/(len(values)-1)
    leave_two=np.asarray([(values.sum()-values[i]-values[j])/(len(values)-2)
                          for i,j in itertools.combinations(range(len(values)),2)])
    result['audit']['M_macro_sensitivity']={
        'note':'Exploratory influence check only; not an added or modified continuation gate.',
        'positive_maps':np.sum(values>0),'negative_maps':np.sum(values<0),
        'leave_one_out_nonpositive':np.sum(leave_one<=0),
        'leave_one_out_range':[leave_one.min(),leave_one.max()],
        'leave_two_out_combinations':len(leave_two),
        'leave_two_out_nonpositive':np.sum(leave_two<=0),
        'leave_two_out_range':[leave_two.min(),leave_two.max()]}
    result['audit']['recorded_training_budget']={
        'main_two_arm_E':sum(result['training'][f'{arm}-50k']['actual_E'] for arm in ('M','G')),
        'separate_smoke_two_arm_E':sum(result['training'][f'smoke-{arm}-10k']['actual_E'] for arm in ('M','G')),
        'note':'Smoke and main logs represent matching prefixes, not independent seeds; runner restarts from ZW0 and has no resume argument. Do not add them as statistical replications.'}
    result=normalise(result)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(f'Wrote {a.output}; verified {len(hashes)} input files against {SOURCE_COMMIT}.')
    for name,pair in result['paired'].items():
        print(name)
        for r in (*RIVALS,'macro'):
            print(' ',r,pair['utility'][r]['both_seats'])
    print('screen', {n:v['passes_budget_screen'] for n,v in result['selection'].items()})


if __name__=='__main__':main()
