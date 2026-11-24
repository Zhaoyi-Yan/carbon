"""Export a zero actor alone or as root logits for the original ft3 search."""
import base64
import io
import re

import numpy as np


ACTOR_HOOK = '''
_ZERO_ACTOR = _Policy(_ZERO_WEIGHTS)
'''

PURE = '''
_ZRNG = np.random.default_rng(37)
_ZTURN = None


def agent(observation, configuration):
    global LAST_TURN, _ZRNG, _ZTURN
    step, side = observation['step'], observation['player']
    players = observation['players']
    LAST_TURN = configuration['episodeSteps'] - 2
    if step == 0 or _ZTURN != step - 1:
        _ST['tr'] = BeliefTracker(_Bel, side)
        _ST['prev'] = None
        _ZRNG = np.random.default_rng(37 + side)
    tr = _ST['tr']
    stock, known = tr.step(players, step, observation['carbon'])
    x = np.clip(np.rint(encode(players,step,side,stock,known,_ST['prev'])*100.),0,255)*.01
    w, b = _ZERO_ACTOR.forward(x,_SID)
    cash,bases,workers,_ = players[side]
    names=[*sorted(bases),*sorted(workers)]
    scores=np.full((12,5),-1.e9,dtype=np.float32)
    scores[:,0]=0.
    for j,u in enumerate(names):
        if j>=2:
            scores[j]=w[workers[u][0]]/_ROOT_TEMP
            continue
        scores[j,0]=b[bases[u],0]/_ROOT_TEMP
        if len(workers)>=configuration['workerLimit']:
            continue
        for k,role in enumerate(('Collector','Planter'),1):
            count=sum(r[2]==role.upper() for team in players for r in team[2].values())
            if cash>=configuration[f'rec{role}Cost']+count*configuration[f'rec{role}IncreaseCost']:
                scores[j,k]=b[bases[u],k]/_ROOT_TEMP
    noisy=scores+_ZRNG.gumbel(size=scores.shape)
    picked=np.argmax(noisy,axis=-1)
    if _ZERO_COORDINATED:
        arrived=set()
        for j,u in enumerate(names):
            if j<2:
                if picked[j]:arrived.add(bases[u])
                continue
            pos=workers[u][0]
            options=[a for a in range(5) if move(pos,a) not in arrived]
            if options:
                picked[j]=max(options,key=lambda a:noisy[j,a])
            arrived.add(move(pos,picked[j]))
    action={}
    for j,u in enumerate(names):
        a=int(picked[j])
        if a:
            action[u]=(BACT if j<2 else WACT)[a]
    tr.set_actions(players,action)
    _ST['prev']=players
    _ZTURN=step
    return action
'''


def weights(original, model, style):
    data={k:v.copy() for k,v in original.p.items()}
    def take(layer,key):
        data[key+'.w']=layer.weight.detach().cpu().numpy().reshape(data[key+'.w'].shape)
        data[key+'.b']=layer.bias.detach().cpu().numpy()
    for layer,key in ((model.stem,'stem'),(model.workers,'hw'),(model.bases,'hb')):
        take(layer,key)
    for i,block in enumerate(model.blocks):
        for layer,name in zip(block,('c1','g','c2'),strict=True):
            take(layer,f'b{i}.{name}')
    data['emb'][style]=model.style.detach().cpu().numpy()
    data['meta']=np.array([original.nb])
    data['snames']=np.array(original.snames)
    buf=io.BytesIO()
    np.savez_compressed(buf,**data)
    return base64.b64encode(buf.getvalue()).decode()


def build(source, original, model, style, temperature, mode, coordinated=False):
    parameters=weights(original,model,style)
    source=re.sub(r"^_SID = .*$",f'_SID = {style}',source,count=1,flags=re.M)
    source=re.sub(r"^_ROOT_TEMP = .*$",f'_ROOT_TEMP = {temperature!r}',source,count=1,flags=re.M)
    glue=f"_ZERO_WEIGHTS = {parameters!r}\n"+ACTOR_HOOK
    if mode=='pure':
        # The standalone policy only needs the existing encoder, referee-facing
        # constants and policy class. Reuse the original weight literal slot.
        head=source[:source.index('def _act(')]
        head=re.sub(r'^_WEIGHTS = .+$',f'_WEIGHTS = {parameters!r}',head,count=1,flags=re.M)
        head=re.sub(r'^_VALUE = .+$','',head,count=1,flags=re.M)
        return head+f'\n_ZERO_ACTOR = _POLICY\n_ZERO_COORDINATED = {coordinated!r}\n'+PURE
    source=source.replace('    lw, lb = _POLICY.forward(x, _SID)','    lw, lb = _ZERO_ACTOR.forward(x, _SID)',1)
    return source+glue
