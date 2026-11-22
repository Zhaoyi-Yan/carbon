"""Fit existing v5 action heads from winning teachers, with disjoint game/map partitions.

python experiments/v6/learn_policy.py --data ../v5-data --online ../v6-data \
    --output ../v6-data/learned.py --report experiments/v6/learned-validation.json
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / 'experiments' / 'v5'
sys.path.insert(0, str(ROOT))
from benchmark import configure_engine, limit_math_threads, load_strategy
from learn_export import HEAD_KEYS, export


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def archives(source):
    blobs = {}
    for name in ('_WEIGHTS', '_VALUE'):
        payload = re.search(r'^' + name + r' = "([^"]+)"$', source, re.M)[1]
        with np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False) as data:
            blobs[name] = dict(data)
    return blobs


class Features:
    def __init__(self, policy, device):
        self.w = {k: torch.tensor(v, device=device) for k, v in policy.items() if v.dtype.kind == 'f'}
        self.nb = int(policy['meta'][0])
        self.device = device

    @staticmethod
    def pad(x):
        x = torch.cat((x[:, :, :, -1:], x, x[:, :, :, :1]), dim=3)
        return torch.cat((x[:, :, -1:], x, x[:, :, :1]), dim=2).contiguous()

    @torch.inference_mode()
    def __call__(self, boards, styles):
        w = self.w
        x = torch.as_tensor(boards, device=self.device).float() * .01
        ids = torch.as_tensor(styles, dtype=torch.long, device=self.device)
        def conv(y, name):
            return F.conv2d(self.pad(y), w[name + '.w'], w[name + '.b'])
        h = (conv(x, 'stem') + w['emb'][ids, :, None, None]).relu()
        for j in range(self.nb):
            name = f'b{j}.'
            q = conv(h, name + 'c1').relu()
            pooled = F.linear(q.mean((2, 3)), w[name + 'g.w'], w[name + 'g.b'])
            h = (h + conv(q + pooled[:, :, None, None], name + 'c2')).relu()
        return h.flatten(2).transpose(1, 2).cpu().numpy()


def collect(args, policy, baseline):
    rank = {r['id']: i for i, r in enumerate(read_json(V5/'design.json')['ranking'])}
    layouts = {g['match']: g['layout'] for g in read_json(V5/'replay-manifest.json')}
    layouts.update({g['match']: g['layout'] for g in read_json(ROOT/'analysis/denial/history-cohort.json')['matches']})
    original_split = read_json(V5/'validation.json')
    old_validation = set(original_split['validation_matches'])
    old_layouts = {layouts[g] for g in original_split['training_matches'] + list(old_validation)}
    partition = read_json(V5/'league-split.json')
    train = set(partition['training_matches'])
    valid = set(partition['validation_matches'])
    # All original v5 training/selection maps are excluded from new dev and test scores.
    held = {layouts[g] for g in valid if g in layouts}
    metadata = {Path(g['file']).name.split('.')[0]:g for g in read_json(ROOT/'replays/top-five/index.json')}
    for subdir in ('older', 'current', 'extra', 'history'):
        source = args.data/subdir/'index.json'
        if source.exists():
            for item in read_json(source):
                metadata[Path(item['file']).name.split('.')[0]] = item
    feat = Features(policy, args.device)
    records = defaultdict(lambda: defaultdict(list))
    manifest = defaultdict(list)
    errors = []
    def capture(group, name, layout, x, styles, actions, credit, row_weights, per_game):
        rng = np.random.default_rng(int(digest((str(args.seed)+name).encode())[:12], 16))
        ix = np.sort(rng.choice(len(x), min(per_game, len(x)), replace=False))
        x, styles, actions, row_weights, credit = x[ix], styles[ix], actions[ix], row_weights[ix], credit[ix]
        if group == 'train':
            styles = np.where(rng.random(len(ix)) < .22, 20, styles)
        for start in range(0, len(ix), args.batch):
            end = start+args.batch
            hidden = feat(x[start:end], styles[start:end])
            for role in (0, 1):
                row, cell = np.where(actions[start:end,role] >= 0)
                segment = records[group][role]
                if len(row):
                    segment.append((hidden[row,cell], actions[start:end,role][row,cell],
                                    row_weights[start:end][row]*credit[start:end,role][row,cell]))
        manifest[group].append(dict(match=name, layout=layout, frames=len(ix)))
    # Use only the winning opponent's observed moves for the new online slice.
    new = []
    for g in read_json(args.online/'index.json'):
        own = next((k for k,v in g['strategies'].items() if v['name']=='yan-v5'), None)
        p = args.online/g['file']
        if own and p.is_file() and g['winner'] not in (own,None) and all(s == 'DONE' for s in g['statuses'].values()):
            r = json.loads(gzip.decompress(p.read_bytes()))
            first = r['steps'][0][0]['observation']
            key = digest(json.dumps([first['full_carbon'],[sorted(o[1].values()) for o in first['players']]],
                                    separators=(',',':')).encode())
            new.append((g,p,key))
    new.sort(key=lambda r: r[0]['finished'], reverse=True)
    online_holdout = {key for _,_,key in new[:max(1,len(new)//3)]}
    online_keys = set()
    for game, p, key in new:
        if key in online_keys or key in old_layouts:
            continue
        online_keys.add(key)
        group = 'online_test' if key in online_holdout else 'train'
        if group == 'train' and key in held:
            continue
        r = json.loads(gzip.decompress(p.read_bytes()))
        frames = r['steps']
        slot = game['winner']; side = game['order'].index(slot)
        teacher = rank.get(game['strategies'][slot]['id'],20)
        tracker = baseline.BeliefTracker(baseline._Bel, side)
        boards=[]; labels=[]; previous=None
        for t,(current,nxt) in enumerate(zip(frames,frames[1:])):
            players=current[0]['observation']['players']; act=nxt[side]['action'] or {}
            est, known = tracker.step(players,t,current[side]['observation']['carbon'])
            if t%3==int(game['id'][:4],16)%3:
                x=baseline.encode(players,t,side,est,known,previous)
                boards.append(np.rint(x*100).clip(0,255).astype(np.uint8))
                lab=np.full((2,225),-1,np.int8)
                for uid,(pos,_,_) in players[side][2].items():
                    lab[0,pos]=baseline.WIDX.get(act.get(uid),0)
                if players[side][0]>=30 and len(players[side][2])<10:
                    for bid,pos in players[side][1].items():
                        lab[1,pos]=baseline.BIDX.get(act.get(bid),0)
                labels.append(lab)
            tracker.set_actions(players,act); previous=players
        y=np.stack(labels)
        capture(group,p.name.split('.')[0],key,np.stack(boards),np.full(len(y),teacher),y,
                np.ones_like(y,dtype=np.float32),np.full(len(y),2.,np.float32),75)
    for p in sorted((args.data/'league-boards').glob('*.npz')):
        name=p.stem
        try:
            with zipfile.ZipFile(p) as archive:
                if archive.testzip() is not None:
                    raise ValueError('CRC verification failed')
            with np.load(p,allow_pickle=False) as z:
                x,t,a,w=(z[k] for k in ('X','T','A','W'))
            if x.shape[1:] != (54,15,15) or a.shape != w.shape or len(x)!=len(a) or t.shape!=(len(x),):
                raise ValueError('invalid array shapes')
        except (OSError,ValueError,KeyError,zipfile.BadZipFile,EOFError) as exc:
            errors.append([name,str(exc)])
            continue
        key=layouts[name]
        if key in online_keys: continue
        if name in train and key not in held:
            group='train'
        elif name in valid and key not in old_layouts:
            group='dev' if int(key[:8],16)%2 else 'test'
        else:
            continue
        g=metadata[name]; winner_style=rank.get(g['strategies'][g['winner']]['id'],-1) if g['winner'] else -1
        # Preserve some losing-teacher coverage, while upweighting the strongest winners.
        rw = np.where(t==winner_style,1.45,.70)*np.where(t<5,1.25,1.)
        capture(group,name,key,x,t,a,np.clip(w,.5,2.),rw,args.frames_per_game)
    groups={k:{role:[torch.from_numpy(np.concatenate(z)) for z in zip(*rows)]
               for role,rows in roles.items()} for k,roles in records.items()}
    parts=[{m['layout'] for m in manifest[k]} for k in groups]
    assert all(not a.intersection(b) for i,a in enumerate(parts) for b in parts[i+1:])
    return groups,manifest,errors


def fit_head(train, prior, bias):
    x,y,weights=train
    x,y,weights=x.float(),y.long(),weights.float()
    mean=x.mean(0); sd=x.std(0).clamp_min(.1); z=((x-mean)/sd)
    old=F.linear(x,prior,bias).detach()
    # Mean/scale are folded back into the ordinary 1x1 convolution on export.
    delta=torch.zeros_like(prior,requires_grad=True)
    db=torch.zeros_like(bias,requires_grad=True)
    opt=torch.optim.LBFGS([delta,db],lr=1.,max_iter=45,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        correction=F.linear(z,delta,db)
        ce=F.cross_entropy(old+correction,y,reduction='none')
        loss=(ce*weights).sum()/weights.sum()+.12*correction.square().mean()+.03*delta.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (delta.detach()/sd,db.detach()-(delta.detach()/sd)@mean)


def score(rows,heads):
    result={}
    for group,roles in rows.items():
        result[group]={}
        for role,(x,y,_) in roles.items():
            w,b=heads[role]
            logits=F.linear(x.float(),w,b)
            result[group][str(role)]={'decisions':len(y),'nll':float(F.cross_entropy(logits,y.long())),
                                    'accuracy':float((logits.argmax(1)==y).float().mean())}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','online','output','report'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--frames-per-game',type=int,default=18)
    parser.add_argument('--seed',type=int,default=32917)
    parser.add_argument('--batch',type=int,default=24)
    args=parser.parse_args();args.device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(args.seed);torch.set_num_threads(1);configure_engine();limit_math_threads()
    path=ROOT/'yan-v5.py'; source=path.read_text(); arrays=archives(source); policy=arrays['_WEIGHTS']
    with np.load(V5/'full-refit.npz',allow_pickle=False) as base:
        assert all(np.array_equal(base[k],v) for bundle in arrays.values() for k,v in bundle.items())
    baseline=load_strategy('v6_encoding_reference',path)
    rows,manifest,errors=collect(args,policy,baseline)
    print('features', {k:len(g) for k,g in manifest.items()},'invalid',len(errors),flush=True)
    original=[];updates=[]
    for i,key in enumerate(('hw','hb')):
        w=torch.from_numpy(policy[key+'.w'].reshape(len(policy[key+'.b']),-1));b=torch.from_numpy(policy[key+'.b'])
        original.append((w,b));updates.append(fit_head(rows['train'][i],w,b))
    def mix(factor):
        return [(w+factor*dw,b+factor*db) for (w,b),(dw,db) in zip(original,updates)]
    # The two test partitions are read once after the mixture has been chosen.
    candidates={}
    for scale in (.35,.70,1.):
        dev=score({'dev':rows['dev']},mix(scale))['dev']
        candidates[scale]=dev['0']['nll']+.35*dev['1']['nll']
    chosen=min(candidates,key=candidates.get); final=mix(chosen)
    metrics={'frozen':score(rows,original),'fitted':score(rows,final)}
    for (w,b),key in zip(final,('hw','hb')):
        policy[key+'.w']=w.numpy().reshape(policy[key+'.w'].shape);policy[key+'.b']=b.numpy()
    heads_path=args.report.parent/'learned-heads.npz'
    np.savez_compressed(heads_path, base_sha256=digest(source.encode()), **{key:policy[key] for key in HEAD_KEYS})
    export(path, heads_path, args.output)
    candidate=args.output.read_text()
    report=dict(seed=args.seed,device=args.device,frames_per_game=args.frames_per_game,selected_scale=chosen,
                dev_objective=candidates,metrics=metrics,training_and_holdout_maps=manifest,
                skipped_invalid=errors,source_sha256=digest(source.encode()),candidate_sha256=digest(candidate.encode()))
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('mixture',chosen,'untouched test',metrics['frozen']['test'],metrics['fitted']['test'],flush=True)
    print('online holdout',metrics['frozen'].get('online_test'),metrics['fitted'].get('online_test'),flush=True)


if __name__=='__main__':
    main()
