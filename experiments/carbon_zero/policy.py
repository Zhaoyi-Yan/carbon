"""Warm-start the Claude ft3 actor and share one auditable joint decoder."""
import hashlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from benchmark import load_strategy

FT3_SHA256 = '920f656a31277751651883099d3bc2891a68412a69533cce4af3e3c801ba9446'
STYLE = 'CL-CarbonEdge_Apollo'
MOVES = (None,'UP','RIGHT','DOWN','LEFT')
HIRES = (None,'RECCOLLECTOR','RECPLANTER')


def load_ft3(path, full=False):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FT3_SHA256:
        raise ValueError('ft3 does not match its frozen digest')
    code=raw.decode()
    if not full:
        code=code.rsplit('\n_net()',1)[0]
    return load_strategy('zero_ft3_'+str(full),path,code.encode())


class Encoder:
    def __init__(self, teacher, seat):
        self.teacher=teacher
        self.track=teacher.BeliefTracker(teacher._Bel,seat)
        self.previous=None

    def encode(self, obs):
        players,turn=obs['players'],obs['step']
        stock,known=self.track.step(players,turn,obs['carbon'])
        x=self.teacher.encode(players,turn,obs['player'],stock,known,self.previous)
        return np.clip(np.rint(x*100.),0,255).astype(np.uint8)

    def commit(self, obs, action):
        self.track.set_actions(obs['players'],action)
        self.previous=obs['players']


PUBLIC_DIM = 6


def public_state_features(gap, turn):
    """Explicit legal public-state baseline features from cash gap and turn."""
    gap_arr = np.asarray(gap, dtype=np.float64)
    turn_arr = np.asarray(turn, dtype=np.float64)
    fraction = turn_arr / 299.0
    cash = np.tanh(gap_arr / 1500.0)
    ones = np.ones_like(gap_arr, dtype=np.float64)
    return np.stack(
        (ones, cash, fraction, cash * fraction, cash * (fraction ** 2), np.sign(gap_arr) * fraction),
        axis=-1,
    ).astype(np.float32)


def public_features_from_obs(obs):
    seat = obs['player']
    gap = obs['players'][seat][0] - obs['players'][1 - seat][0]
    return public_state_features(gap, obs['step'])


class Actor(nn.Module):
    def __init__(self, policy, style, *, residual_critic=False, detach_value_trunk=False):
        super().__init__()
        arrays=policy.p
        width=arrays['stem.w'].shape[0]
        self.residual_critic = residual_critic
        self.detach_value_trunk = detach_value_trunk
        self.stem=nn.Conv2d(54,width,3)
        self.style=nn.Parameter(torch.from_numpy(arrays['emb'][style].copy()))
        self.blocks=nn.ModuleList([nn.ModuleList([nn.Conv2d(width,width,3),nn.Linear(width,width),
                                                nn.Conv2d(width,width,3)]) for _ in range(policy.nb)])
        self.workers=nn.Linear(width,5)
        self.bases=nn.Linear(width,3)
        if residual_critic:
            self.value_baseline = nn.Linear(PUBLIC_DIM, 1, bias=False)
            nn.init.zeros_(self.value_baseline.weight)
            self.value = nn.Sequential(nn.Linear(2 * width + PUBLIC_DIM, 64), nn.Tanh(), nn.Linear(64, 1))
        else:
            self.value_baseline = None
            self.value=nn.Sequential(nn.Linear(2*width,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

        def copy(layer, key):
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(arrays[key+'.w']).reshape_as(layer.weight))
                layer.bias.copy_(torch.from_numpy(arrays[key+'.b']))
        for layer,key in ((self.stem,'stem'),(self.workers,'hw'),(self.bases,'hb')):
            copy(layer,key)
        for i,block in enumerate(self.blocks):
            for layer,name in zip(block,('c1','g','c2'),strict=True):
                copy(layer,f'b{i}.{name}')

    def spatial(self, features):
        x=features.float().mul(.01)
        def pad(v):
            # Circular padding using concatenation works on CPU and MPS.
            h=torch.cat((v[:,:,:,-1:],v,v[:,:,:,:1]),3)
            return torch.cat((h[:,:,-1:,:],h,h[:,:,:1,:]),2).contiguous()
        x=(self.stem(pad(x))+self.style[None,:,None,None]).relu()
        for first,gate,second in self.blocks:
            h=first(pad(x)).relu()
            x=(x+second(pad(h+gate(h.mean((2,3)))[:,:,None,None]))).relu()
        return x.flatten(2).transpose(1,2)

    @staticmethod
    def value_input(cells):
        return torch.cat((cells.mean(1),cells.amax(1)),1)

    def predict_value(self, value_features, pub=None):
        if self.residual_critic:
            if pub is None:
                raise ValueError('Residual critic requires public state features')
            pub = pub.float()
            base = self.value_baseline(pub)[:, 0]
            res = self.value(torch.cat((value_features, pub), dim=1))[:, 0]
            return base + res
        return self.value(value_features)[:, 0]

    def forward(self, features, pub=None):
        cells = self.spatial(features)
        value_cells = cells.detach() if self.detach_value_trunk else cells
        value = self.predict_value(self.value_input(value_cells), pub)
        return self.workers(cells),self.bases(cells),value


def slots(obs, cfg):
    cash,bases,workers,_=obs['players'][obs['player']]
    names=[*sorted(bases),*sorted(workers)]
    assert len(bases)==2 and len(workers)<=10
    cells=np.zeros(12,np.int64)
    cells[:len(names)]=[bases[u] if u in bases else workers[u][0] for u in names]
    # Conservative per-base mask. Simultaneous attempts that exhaust the last slot
    # or remaining cash are still legal commands and are settled by the referee.
    legal=np.zeros((12,5),bool)
    legal[:,0]=True
    legal[2:len(names)]=True
    if len(workers)<cfg['workerLimit']:
        for k,role in enumerate(('Collector','Planter'),1):
            count=sum(w[2]==role.upper() for team in obs['players'] for w in team[2].values())
            cost=cfg[f'rec{role}Cost']+count*cfg[f'rec{role}IncreaseCost']
            legal[:2,k]=cash>=cost
    return names,cells,legal


def action_logits(w, b, cells, legal, temperature):
    rows=torch.arange(len(w),device=w.device)[:,None]
    gathered=torch.cat((F.pad(b[rows,cells[:,:2]],(0,2)),w[rows,cells[:,2:]]),dim=1)
    return (gathered/temperature).masked_fill(~legal,-1.e9)


def logits(model, x, cells, legal, temperature, pub=None):
    w,b,v=model(x, pub=pub) if getattr(model, 'residual_critic', False) else model(x)
    return action_logits(w,b,cells,legal,temperature),v


def sample(logits, rng):
    scores=logits.cpu().numpy()
    return np.argmax(scores+rng.gumbel(size=scores.shape),axis=-1)


def commands(names, indices):
    action={}
    for j,uid in enumerate(names):
        cmd=(HIRES if j<2 else MOVES)[indices[j]]
        if cmd is not None:
            action[uid]=cmd
    return action

DESTINATIONS = np.asarray([[
    ((p//15+dy)%15)*15+(p%15+dx)%15 for dy,dx in ((0,0),(-1,0),(0,1),(1,0),(0,-1))
] for p in range(225)])


def coordinated_sample(scores,rng,cells,legal,*,noise=None):
    """Fixed-order occupancy masks are conditional on actions already chosen."""
    z=scores.cpu().numpy()
    effective=legal.copy()
    destinations=DESTINATIONS[cells]
    gumbel=rng.gumbel(size=z.shape) if noise is None else noise
    chosen=np.zeros(z.shape[:2],np.int64)
    occupied=np.zeros((len(z),225),bool)
    rows=np.arange(len(z))
    for j in range(z.shape[1]):
        live=legal[:,j].sum(1)>1
        if j>=2:
            safe=~occupied[rows[:,None],destinations[:,j]]
            mask=safe&legal[:,j]
            # If there is no collision-free instruction, the referee settles
            # the selected legal command under the game rules.
            effective[:,j]=np.where(mask.any(1)[:,None],mask,legal[:,j])
        pick=np.argmax(np.where(effective[:,j],z[:,j]+gumbel[:,j],-np.inf),axis=1)
        chosen[:,j]=pick
        tile=cells[:,j] if j<2 else destinations[rows,j,pick]
        arriving=pick>0 if j<2 else live
        occupied[rows[arriving],tile[arriving]]=True
    return chosen,effective


def actor_from_checkpoint(teacher, style, path, device='cpu'):
    """Rebuild an Actor from a terminal-RL checkpoint, including critic flags."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    actor = Actor(teacher._POLICY, style,
                  residual_critic=bool(saved.get('residual_critic', False)),
                  detach_value_trunk=bool(saved.get('detach_value_trunk', False)))
    actor.load_state_dict(saved['parameters'])
    return actor.to(device).eval(), saved
