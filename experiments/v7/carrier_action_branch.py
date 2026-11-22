"""Single-first-action interventions in public positions, with both policies reactive for 20 steps."""
import argparse,gzip,json,os
from pathlib import Path
import master_e_branch as arena
import carrier_action as carrier

WINDOWS=((40,55,65),(89,95,103),(125,140,150),(164,172,183),(200,215,230))
SEEDS=(31051,49069)


class NoProposal(Exception):
    pass


def compare(m,clone,model,data,progress):
    times=sorted({x for group in WINDOWS for x in group})
    arena.START_END=tuple((t,t+20) for t in times)
    cfg=data['configuration']
    assert data['seats']==['claude-AlphaCarbon-Master7sE','yan-QQ-v6']
    data['actions']=data['actions'][:max(times)];data['expected']=data['expected'][:max(times)+1]
    m._ST={};m._FC_HISTORY={};clone.reset()
    snapshots=arena.history(data,{'harbormaster':m},clone)
    original=m.agent;search=m._search
    lock=set()
    def see_locked(obs,cfg,stock,act,lw,lb,prev,t0,home=()):
        if obs['step']==start:
            lock.update(home);lock.update(m._r6_early_locks(obs,act,lw))
        return search(obs,cfg,stock,act,lw,lb,prev,t0,home)
    m._search=see_locked
    attempts=[];pairs=[]
    def trial(start,seed,apply):
        root_info={}
        def actor(obs,cfg):
            if obs['step']!=start:return original(obs,cfg)
            previous=m._ST.get('prev');lock.clear()
            lock.update(uid for t,uid,tiles in m._ST.get('corridor_track',()) if t==start)
            base=original(obs,cfg)
            lock.update(uid for t,uid,tiles in m._ST.get('corridor_track',()) if t==start+1)
            alt,diag=carrier._ca_proposal(obs,base,lock,previous,model)
            if alt is None:raise NoProposal()
            uid=diag['uid'];assert uid not in lock and {w for w in alt.keys()|base.keys() if alt.get(w)!=base.get(w)}=={uid}
            root_info.update(diag,worker=uid,n_locked=len(lock),baseline=base,edited=alt)
            if apply:
                m._ST['tr'].set_actions(obs['players'],alt);m._ST['liquidity_action']=dict(alt)
                return alt
            return base
        m.agent=actor
        try:
            case=arena.branch(cfg,snapshots[start],clone,m,'harbormaster',start,start+20,seed)
        finally:
            m.agent=original
        windows={}
        for h in (10,20):
            t=min(start+h,case['through']);events=[e for e in case['deposits'] if e[0]<=t]
            windows[h]={'us':round(sum(x[4] for x in events if x[1]==x[2]==1),3),
                        'foe':round(sum(x[4] for x in events if x[1]==x[2]==0),3),
                        'uid':round(sum(x[4] for x in events if x[1]==x[2]==1 and x[3]==root_info['uid']),3),
                        'cash_margin':case['milestones'][t]['margin_QQ']}
        return root_info,dict(windows=windows,deposits=case['deposits'],initial=case['opening'],actions_sha=case['action_sha256'])
    for seed in SEEDS:
        for group in WINDOWS:
            for start in group:
                try:root,base=trial(start,seed,False)
                except NoProposal:
                    attempts.append([seed,start,False]);continue
                attempts.append([seed,start,True]);diag,alt=trial(start,seed,True)
                assert root==diag and base['initial']==alt['initial']
                deltas={h:{key:round(alt['windows'][h][key]-v,3) for key,v in base['windows'][h].items()} for h in (10,20)}
                for h in deltas:deltas[h]['net_own_minus_foe']=round(deltas[h]['us']-deltas[h]['foe'],3)
                row=dict(match=data['match_id'],map_seed=cfg['randomSeed'],seed=seed,start=start,edit=root,original=base,modified=alt,delta=deltas)
                pairs.append(row);progress.write(json.dumps(row)+'\n');progress.flush()
                print('pair',data['match_id'][:5],seed,start,root['old'],root['new'],deltas[10]['net_own_minus_foe'],deltas[20]['net_own_minus_foe'],flush=True)
                break
    m._search=search
    return {'match':data['match_id'],'map_seed':cfg['randomSeed'],'checked':attempts,'pairs':pairs}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--progress',required=True,type=Path)
    parser.add_argument('--out',default=arena.OUT/'carrier-action-branches.json',type=Path);a=parser.parse_args()
    arena.limit_math_threads();arena.configure_engine();os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    m=arena.load_ours('harbormaster',arena.HERE/'checkpoints/harbormaster.py');clone=arena.Clone()
    model=dict(carrier.np.load(arena.HERE/'carrier-action-model.npz'))
    src=arena.OUT/'branch-breakout-replays.json.gz';games=json.loads(gzip.decompress(src.read_bytes()));out=[]
    with a.progress.open('w') as f:
        for name,data in games.items():out.append(compare(m,clone,model,data,f))
    rows=[p for case in out for p in case['pairs']];summary={}
    for subset,pairs in [('all',rows)]+[(d['match'],d['pairs']) for d in out]:
        summary[subset]={}
        for h in (10,20):
            summary[subset][h]={key:{'mean':round(sum(p['delta'][h][key] for p in pairs)/max(1,len(pairs)),3),
                                       'positive':sum(p['delta'][h][key]>0 for p in pairs),
                                       'equal':sum(p['delta'][h][key]==0 for p in pairs),'negative':sum(p['delta'][h][key]<0 for p in pairs)}
                              for key in ('us','foe','uid','net_own_minus_foe','cash_margin')}
    report=dict(policy='Frozen Harbormaster, both arms. Change only the loaded collector first action; subsequent moves react with the same policy and per-turn seeds.',
                opponent='Reactive known-style Master7sE clone, seat 0; its subsequent actions also recomputed. Endpoints are official referee gross own-to-own deposits and cash differences.',
                eligibility='First model-supported proposal found in each predeclared time window, never selected on actual branch outcome.',
                seeds=SEEDS,window_candidates=WINDOWS,model_sha256=arena.shot(arena.HERE/'carrier-action-model.npz'),
                referee_input_sha256=arena.shot(src),maps=out,summary=summary)
    a.out.write_text(json.dumps(report,indent=1)+'\n');print('finished',len(rows),summary['all'],flush=True)

if __name__=='__main__':main()
