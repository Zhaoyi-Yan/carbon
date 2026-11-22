"""Check the turn-zero simulation choice against complete root-policy games on mirrored development maps."""
import hashlib
import json
from pathlib import Path
from statistics import mean
import opening_probe as op
from branch_replay import ROOT,OUT,HERE
from benchmark import limit_math_threads,configure_engine,load_strategy,new_game,observe
from opening_rollout import _v7_launch


def main():
    limit_math_threads();configure_engine();m=load_strategy('fitted_opening',HERE/'checkpoints/pilotage.py');m._torch.set_num_threads(1)
    _v7_launch.__globals__.update({k:getattr(m,k) for k in ('_turn','np','_time','_v5_pair','_V6R_TINY','_sample','_r6_opening','_r6_asset_score')})
    prev=op.variant;root=op.load_policy();controlled={};baseline={};saved=[]
    def select(mode,obs,cfg,a,logits,tree,net):
        if mode=='incumbent':return prev(mode,obs,cfg,a,logits,tree,net)
        if obs['step']>0:return prev('mixed_now',obs,cfg,a,logits,tree,net)
        return controlled[obs['player']] if mode=='launch' else baseline[obs['player']]
    op.variant=select
    seeds=json.loads((HERE/'panels.json').read_text())['development']
    for seed in seeds:
        st,env=new_game(seed,300);meta=[]
        for s in (0,1):
            o=observe(st,s,0);tr=m.BeliefTracker(m._Bel,s);stock,k=tr.step(o['players'],0,o['carbon'])
            x=m._quant(m.encode(o['players'],0,s,stock,k,None));m._ST={}
            _,lb=m._v5_root(o,stock,x,None);default=m._v7_early_income(o,env.configuration,lb,{})
            moves,meta_s=_v7_launch(o,env.configuration,stock,default,0.,True);controlled[s]=moves;baseline[s]=default
            meta.append(dict(seat=s,default=default,moves=moves,forecast=meta_s))
        baselines=[];candidates=[]
        for mode,store in [('mixed_now',baselines),('launch',candidates)]:
            for side in (0,1):
                g=op.create(seed,mode,side);op.advance(g,299,root);me=g['marks'][str(g['turn'])][side];them=g['marks'][str(g['turn'])][1-side]
                store.append(round(me['cash']-them['cash'],2))
        saved.append(dict(seed=seed,opening=meta,default_margins=baselines,launch_margins=candidates))
        print(seed,[(a['forecast']['choice'],a['forecast']['margin']) for a in meta],baselines,candidates,flush=True)
        (OUT/'opening-rollouts.json').write_text(json.dumps(dict(source='closed-loop v6 root continuation with fixed per-turn randomness; full simulated initial choice compared with immediate mixed base opening',maps=saved),indent=2)+'\n')
if __name__=='__main__':main()
