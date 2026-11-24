"""Calibrate ft3 Apollo, run synchronized PPO, and retain local checkpoints and evaluation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from arena import engine_fingerprint
from env import Game, carbon, reward, potential, utility
from evaluate import games, seeds
from policy import Actor, Encoder, FT3_SHA256, STYLE, load_ft3, slots, logits
from ppo import Collector, advantages, update


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)


def basic_checks(teacher, model, style, device):
    g=Game(726273)
    errors=[]
    for tick in range(5):
        obs=g.observations()
        for i,o in enumerate(obs):
            e=Encoder(teacher,i)
            x=e.encode(o)
            ref=teacher._POLICY.forward(x.astype(np.float32)*.01,style)
            with torch.inference_mode():
                w,b,_=model(torch.as_tensor(x[None],device=device))
            errors.append([float(np.max(np.abs(a-v[0].cpu().numpy()))) for a,v in zip(ref,(w,b),strict=True)])
        acts=tuple({next(iter(o['players'][i][1])):'RECCOLLECTOR'} if tick==0 else {}
                   for i,o in enumerate(obs))
        g.step(acts)
    assert max(max(e) for e in errors)<.003,errors
    g=Game(928374)
    start=g.cash(); cumulative=0.
    own=[]
    while not g.done:
        before=g.cash()
        eliminated,scores=g.step(({},{}))
        r=reward(before,g.cash(),g.done,scores,eliminated,0)
        assert r==-reward(before,g.cash(),g.done,scores,eliminated,1)
        cumulative+=r
    assert abs(cumulative+potential(start)-utility(scores[0]-scores[1],eliminated))<1e-12
    v=advantages(np.array([[.2],[.3]]),np.array([[.1],[.4]]),np.array([[False],[True]]),np.array([999.]))
    assert np.allclose(v[:,0],[.405,-.1])
    assert np.allclose(advantages(np.array([[.2]]),np.array([[.1]]),np.array([[False]]),np.array([.4])),[[.5]])
    return dict(parity_error=errors,max_absolute_error=max(max(e) for e in errors),
                shaped_return=cumulative,bootstrap=True)


def summary(panel):
    rows=panel['games']
    u=sum(r['utility'] for r in rows)/len(rows)
    return dict(wins=sum(r['margin']>0 for r in rows),draws=sum(r['margin']==0 for r in rows),
                losses=sum(r['margin']<0 for r in rows),mean_margin=sum(r['margin'] for r in rows)/len(rows),
                mean_utility=u,seconds=panel['seconds'],games=rows)


def main(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device=a.device
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=a.resume)
    teacher=load_ft3(a.ft3)
    style=teacher._POLICY.sid(STYLE)
    assert teacher._POLICY.snames[style]==STYLE
    initial=Actor(teacher._POLICY,style).to(device)
    base=copy.deepcopy(initial).eval()
    chosen_maps=seeds('temperature',4)
    eval_self=seeds('evaluation-self',4)
    eval_raw=seeds('evaluation-raw',2)
    tag='v11-yan-v8-elite-refit'
    frozen_v8=hashlib.sha256(subprocess.check_output(['git','show',tag+':submission.py'],cwd=ROOT)).hexdigest()
    protocol=dict(seed=a.seed,cpu_count=os.cpu_count(),threads=a.threads,device=device,engine=engine_fingerprint(carbon),
        source_sha256=FT3_SHA256,style=dict(name=STYLE,index=style),config=Game(a.seed).config,
        frozen_v8=[tag,frozen_v8],training=dict(beta=.5,K=1500,alpha=.5,gamma=1,gae=.95,
        envs=a.envs,rollout=a.rollout,decoder='sequential' if a.coordinated else 'independent',lr=a.lr,
        anchor_kl_weight=a.anchor_kl),panels=dict(temperature=chosen_maps,self=eval_self,raw=eval_raw))
    opt=torch.optim.AdamW(initial.parameters(),lr=a.lr)
    rng=np.random.default_rng(a.seed+133)
    if a.resume:
        saved=torch.load(a.resume,weights_only=False,map_location='cpu')
        initial.load_state_dict(saved['parameters'])
        opt.load_state_dict(saved['optimizer'])
        n=saved['E']
        calibrate=json.loads((out/'calibration.json').read_text())
        temp=calibrate['temperature']
        updates=json.loads((out/'updates.json').read_text())
        collector_seed=saved['next_seed']
        if 'optimizer_rng' in saved:rng.bit_generator.state=saved['optimizer_rng']
    else:
        write(out/'protocol.json',protocol)
        write(out/'checks.json',basic_checks(teacher,initial,style,device))
        trial=summary(games(teacher,base,base,chosen_maps,.4,1.,device,coordinated=a.coordinated))
        temp=.4 if trial['mean_utility']>=0 else 1.
        native=load_ft3(a.ft3,full=True)
        native._SID=style;native._ROOT_TEMP=.4
        native._search=lambda _o,_c,_e,action,*args:action
        raw=[games(teacher,base,None,[m],temp,.4,device,raw=(native,seat),coordinated=a.coordinated)
             for m in eval_raw for seat in (0,1)]
        calibrate=dict(temperature=temp,T04_vs_T1=trial,
                       decoder_vs_W_raw=summary(dict(games=[r for d in raw for r in d['games']],
                                                    seconds=sum(d['seconds'] for d in raw))))
        write(out/'calibration.json',calibrate)
        updates=[]
        n=0
        collector_seed=a.seed
    collector=Collector(teacher,a.envs,collector_seed,device,a.coordinated)
    if a.resume and 'collector' in saved:collector.recover(saved['collector'])
    checkpoints=sorted({10000,100000,200000,a.steps})
    while n<a.steps:
        pending=min(goal for goal in checkpoints if goal>n)
        turns=min(a.rollout,max(1,(pending-n+a.envs-1)//a.envs))
        trajectory=collector.collect(initial,base,temp,turns)
        n+=len(trajectory[0])
        now=time.perf_counter()
        row=update(initial,opt,trajectory,device,temp,rng,warmup=len(updates)<2,
                   anchor=base,anchor_weight=a.anchor_kl)
        row.update(E=n,L=n,optimization_seconds=time.perf_counter()-now,
                   cumulative=dict(collector.timing),forwards=collector.calls,finished=len(collector.episodes))
        updates.append(row)
        write(out/'updates.json',updates)
        if n>=pending:
            weights=dict(parameters=initial.state_dict(),optimizer=opt.state_dict(),E=n,next_seed=collector.next_seed,
                         optimizer_rng=rng.bit_generator.state,collector=collector.snapshot())
            torch.save(weights,out/f'checkpoint-{pending}.pt')
            head=copy.deepcopy(initial)
            head.load_state_dict(torch.load(out/f'checkpoint-{pending}.pt',weights_only=True,map_location='cpu')['parameters'])
            check=summary(games(teacher,head,base,eval_self,temp,temp,device,coordinated=a.coordinated))
            write(out/f'self-{pending}.json',check)
            write(out/'episodes.json',collector.episodes)
            print('checkpoint',pending,'E',n,'games',check['wins'],check['draws'],check['losses'],
                  'E/s',round(n/collector.timing['rollout'],2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ft3',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=24000000)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--rollout',type=int,default=64)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='mps')
    p.add_argument('--resume',type=Path)
    p.add_argument('--coordinated',action='store_true')
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--anchor-kl', type=float, default=0.,
                   help='Optional team KL penalty against frozen ZW0')
    main(p.parse_args())
