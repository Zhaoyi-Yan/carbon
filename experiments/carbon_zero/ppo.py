"""PPO over exact joint team actions generated under one frozen behavior policy."""
from dataclasses import dataclass, field
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from env import Game, reward
from policy import Encoder, commands, coordinated_sample, logits, sample, slots


def joint_logprob(scores, chosen):
    return scores.log_softmax(-1).gather(-1,chosen[...,None]).squeeze(-1).sum(-1)


def advantages(rewards, values, terminal, next_value, lam=.95):
    out=np.zeros_like(values)
    acc=np.zeros_like(next_value)
    for t in range(len(values)-1,-1,-1):
        active=~terminal[t]
        error=rewards[t]+active*next_value-values[t]
        acc=error+lam*active*acc
        out[t]=acc
        next_value=values[t]
    return out


@dataclass
class Position:
    game: Game
    enc: tuple
    seed: int
    seat: int
    actions: list = field(default_factory=list)


class Collector:
    def __init__(self, teacher, count, seed, device, coordinated=False):
        self.teacher=teacher
        self.next_seed=seed
        self.game_number=0
        self.device=device
        self.coordinated=coordinated
        self.rng=np.random.default_rng(seed)
        self.states=[self.new() for _ in range(count)]
        self.episodes=[]
        self.timing=dict(encode=0.,forward=0.,referee=0.,rollout=0.)
        self.calls=0

    def new(self):
        seed=self.next_seed
        self.next_seed+=1
        self.game_number+=1
        return Position(Game(seed),tuple(Encoder(self.teacher,s) for s in (0,1)),seed,(self.game_number-1)%2)

    def snapshot(self):
        return dict(next_seed=self.next_seed, game_number=self.game_number, rng=self.rng.bit_generator.state,
                    positions=[dict(seed=p.seed,seat=p.seat,actions=p.actions) for p in self.states],
                    episodes=self.episodes, timing=self.timing, calls=self.calls)

    def recover(self, checkpoint):
        self.next_seed=checkpoint['next_seed']
        self.game_number=checkpoint['game_number']
        self.rng.bit_generator.state=checkpoint['rng']
        self.episodes=checkpoint['episodes']
        self.timing=checkpoint['timing']
        self.calls=checkpoint['calls']
        self.states=[]
        for spec in checkpoint['positions']:
            g=Game(spec['seed'])
            e=tuple(Encoder(self.teacher,i) for i in (0,1))
            for moves in spec['actions']:
                for i,(encoder,obs) in enumerate(zip(e,g.observations(),strict=True)):
                    encoder.encode(obs)
                    encoder.commit(obs,moves[i])
                g.step(moves)
                assert not g.done
            self.states.append(Position(g,e,spec['seed'],spec['seat'],list(spec['actions'])))

    def make_inputs(self, seats):
        features=[]; positions=[]; legal=[]; names=[]; obs_rows=[]
        for p,side in seats:
            obs=p.game.observations()[side]
            w,pos,mask=slots(obs,p.game.config)
            features.append(p.enc[side].encode(obs))
            positions.append(pos);legal.append(mask);names.append(w);obs_rows.append(obs)
        return [np.stack(x) for x in (features,positions,legal)],names,obs_rows

    def tensors(self, arrays):
        return [torch.as_tensor(a,device=self.device) for a in arrays]

    @torch.inference_mode()
    def collect(self, learner, opponent, temperature, turns):
        start=time.perf_counter()
        parts=[]
        for _ in range(turns):
            own=[(p,p.seat) for p in self.states]
            enemy=[(p,1-p.seat) for p in self.states]
            now=time.perf_counter()
            a,ids,obs=self.make_inputs(own)
            b,enemy_ids,enemy_obs=self.make_inputs(enemy)
            self.timing['encode']+=time.perf_counter()-now
            now=time.perf_counter()
            z,v=logits(learner,*self.tensors(a),temperature)
            q,_=logits(opponent,*self.tensors(b),temperature)
            if self.coordinated:
                aa,mask=coordinated_sample(z,self.rng,a[1],a[2])
                bb,_=coordinated_sample(q,self.rng,b[1],b[2])
                a[2]=mask
                z=z.masked_fill(~torch.as_tensor(mask,device=self.device),-1.e9)
            else:
                aa,bb=sample(z,self.rng),sample(q,self.rng)
            old=joint_logprob(z,torch.as_tensor(aa,device=self.device)).cpu().numpy()
            values=v.cpu().numpy()
            self.calls+=2
            self.timing['forward']+=time.perf_counter()-now
            now=time.perf_counter()
            rewards=[];ends=[]
            for i,p in enumerate(self.states):
                act=commands(ids[i],aa[i]); reply=commands(enemy_ids[i],bb[i])
                p.enc[p.seat].commit(obs[i],act)
                p.enc[1-p.seat].commit(enemy_obs[i],reply)
                before=p.game.cash()
                joint=(act,reply) if p.seat==0 else (reply,act)
                p.actions.append(joint)
                eliminated,scores=p.game.step(joint)
                terminal=p.game.done
                rewards.append(reward(before,p.game.cash(),terminal,scores,eliminated,p.seat))
                ends.append(terminal)
                if terminal:
                    self.episodes.append(dict(seed=p.seed,seat=p.seat,scores=scores,
                                              kind='elimination' if eliminated else 'full',turns=p.game.turn))
                    self.states[i]=self.new()
            self.timing['referee']+=time.perf_counter()-now
            parts.append((*a,aa,old,values,rewards,ends))

        # Encoding advances BeliefTracker. Bootstrap on a copy and leave the
        # retained tracker at the same point for the next sampling window.
        saved=[p.enc[p.seat].track for p in self.states]
        for p,track in zip(self.states,saved):
            p.enc[p.seat].track=copy.deepcopy(track)
        tails,*_=self.make_inputs([(p,p.seat) for p in self.states])
        for p,track in zip(self.states,saved):
            p.enc[p.seat].track=track
        _,tail=logits(learner,*self.tensors(tails),temperature)
        self.calls+=1
        features,cells,legal,acts,lp,v,r,end=[np.stack(chunk) for chunk in zip(*parts,strict=True)]
        adv=advantages(r,v,end,tail.cpu().numpy())
        target=adv+v
        self.timing['rollout']+=time.perf_counter()-start
        fields=(features,cells,legal,acts,lp,adv,target)
        return [p.reshape((-1,*p.shape[2:])) for p in fields]


def update(model, optimizer, data, device, temperature, rng, warmup=False, anchor=None, anchor_weight=0.):
    features,cells,legal,acts,old,adv,target=data
    adv=(adv-adv.mean())/max(adv.std(),1.e-6)
    stats=[]
    count=len(features)
    test=np.arange(min(64,count))
    anchor_logp = None
    if anchor_weight and not warmup:
        if anchor is None:
            raise ValueError('The PPO anchor must be a frozen policy')
        with torch.no_grad():
            chunks = []
            for start in range(0, count, 256):
                rows = slice(start, start + 256)
                inputs = [torch.as_tensor(v[rows], device=device) for v in (features, cells, legal)]
                scores, _ = logits(anchor, *inputs, temperature)
                chunks.append(scores.log_softmax(-1))
            anchor_logp = torch.cat(chunks)
    anchor_values = []
    def batch(ids):
        a,b,c,d,e,f,g=(torch.as_tensor(field[ids],device=device) for field in (features,cells,legal,acts,old,adv,target))
        z,v=logits(model,a,b,c,temperature)
        return z,v,d,e,f,g,c
    with torch.no_grad():
        z,_,a,lp,*_=batch(test)
        error=float((joint_logprob(z,a)-lp).abs().max().cpu())
    assert error<2e-4,error
    if warmup:
        for name,param in model.named_parameters():
            param.requires_grad_(name.startswith('value.'))
    for _ in range(2):
        order=rng.permutation(count)
        for start in range(0,count,256):
            ids = order[start:start+256]
            z,v,a,b,c,d,mask=batch(ids)
            lp=joint_logprob(z,a)
            ratio=(lp-b).exp()
            joint_kl=(ratio-1-(lp-b)).mean()
            if not warmup and float(joint_kl.detach().cpu())>.02:
                break
            logp=z.log_softmax(-1)
            entropy=-(logp.exp()*logp).sum(-1)
            valid=mask.sum(-1)>1
            ent=(entropy*valid).sum(-1)/valid.sum(-1).clamp_min(1)
            clip=(ratio-1).abs()>.2
            value_loss=F.mse_loss(v,d.float())
            actor_loss=-torch.minimum(ratio*c,ratio.clamp(.8,1.2)*c).mean()
            loss=.5*value_loss
            if not warmup:
                loss=loss+actor_loss-.01*ent.mean()
            if anchor_logp is not None:
                frozen_logp = anchor_logp[torch.as_tensor(ids, device=device)]
                drift = (frozen_logp.exp() * (frozen_logp - logp)).sum((-1,-2)).mean()
                loss = loss + anchor_weight * drift
                anchor_values.append(drift.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            optimizer.step()
            info=torch.stack((joint_kl,clip.float().mean(),ent.mean(),value_loss,norm))
            stats.append(info.detach().cpu().numpy())
        if not warmup and float(joint_kl.detach().cpu())>.02:
            break
    if warmup:
        for param in model.parameters():
            param.requires_grad_(True)
    metrics = dict(zip(('joint_kl','clipped','entropy','value_loss','gradient_norm'),
                        map(float,np.asarray(stats).mean(0)),strict=True),logprob_error=error)
    if anchor_values:
        metrics['anchor_kl'] = float(torch.stack(anchor_values).mean().cpu())
    return metrics
