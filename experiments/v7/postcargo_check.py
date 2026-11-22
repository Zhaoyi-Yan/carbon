"""Paired counterfactual validation of the post-search collision chooser on unseen public boards."""
import argparse,copy,gzip,hashlib,inspect,json,sys,time,types
from collections import defaultdict
from pathlib import Path
import numpy as np
from postcargo_model import _post_cargo_adjust,_post_score
from receipt_data import layout,ACT,LAND
from starvation_replay import landings
from distill_data import ROOT
from benchmark import validate_actions

def fixed(path):
    src=path.read_text().replace('_torch.set_num_threads(2)','_torch.set_num_threads(1)')
    start=src.index('    if _S_COST[0] > 0.0:',src.index('def _search('));end=src.index('    stages =',start)
    src=src[:start]+'    most, D2, M = 4, min(D2,2), 1\n'+src[end:]
    src=src.replace('for tick in range(1,15):','for tick in range(1,4):')
    m=types.ModuleType('post_research');exec(compile(src,str(path),'exec'),m.__dict__);m._torch.set_num_threads(1)
    m._time=types.SimpleNamespace(perf_counter=lambda:0.);m._S_MMAX=1
    assert m._SEARCH_OK and m._RX
    original=m._reply_choice
    start=src.index('def _reply_choice(');end=src.index('    best,best_score=act',start)
    hook=src[start:end]+"    return _POST_CAPTURE(obs,config,est,act,candidates,weights,actions)\n"
    exec(inspect.getsource(_post_cargo_adjust)+'\n'+hook,m.__dict__)
    return m,original

def select(frames,side,seed,limit):
    choices=defaultdict(list)
    for t in range(30,len(frames)-11):
        pl=frames[t][0]['observation']['players']
        stake=0.
        for p,v,k in pl[side][2].values():
            for q,w,r in pl[1-side][2].values():
                dy,dx=abs(p//15-q//15),abs(p%15-q%15)
                if min(dy,15-dy)+min(dx,15-dx)>2:continue
                if k=='COLLECTOR' and r=='COLLECTOR':stake=max(stake,v,w)
        if stake>=45:
            # Across all phases; selection only knows pre-action positions and cargo.
            key=hashlib.sha256(f'{seed}-{t}'.encode()).hexdigest();choices[t//24].append((key,t))
    strata=[sorted(v) for v in choices.values()]
    proposed=[]
    for j in range(limit):
        available=[r[j] for r in strata if len(r)>j]
        for _,t in sorted(available):
            if all(abs(t-n)>2 for n in proposed):proposed.append(t)
            if len(proposed)>=limit:return sorted(proposed)
    return sorted(proposed)

def changes(before,after,pl):
    return [dict(uid=u,from_action=before.get(u,'NONE'),to_action=after.get(u,'NONE'),kind=k,cargo=round(v,3),pos=p)
            for u,(p,v,k) in pl[2].items() if before.get(u)!=after.get(u)]

def replay_state(m,g):
    frames=g['steps'];obs=frames[0][0]['observation']
    board=dict(players=obs['players'],trees=obs['trees'],carbon=obs['full_carbon'],step=0,player=0)
    cfg=g['configuration'];straight=[board];maxcash=0.
    for t,frame in enumerate(frames[1:]):
        b=m._turn(board,cfg,[z['action'] or {} for z in frame]);expected=frame[0]['observation']['players']
        for pl,exp in zip(b['players'],expected):
            if exp[2] or exp[0]:
                maxcash=max(maxcash,abs(exp[0]-pl[0]))
                assert set(pl[2])==set(exp[2]) and set(pl[3])==set(exp[3])
                assert all(w[0]==exp[2][uid][0] and w[2]==exp[2][uid][2] and abs(w[1]-exp[2][uid][1])<1e-5 for uid,w in pl[2].items())
        board=b;straight.append(b)
    assert maxcash<.011
    return straight,maxcash

def tail(m,start,t,side,cfg,frames,own):
    board=start;received=np.zeros(2);lost=np.zeros(2);out={};fallback=0
    mapping=[{u:u for u in p[2]} for p in board['players']]
    for delta in range(1,11):
        commands=[dict(z['action'] or {}) for z in frames[t+delta]]
        if delta==1:commands[side]=own
        teams=board['players']
        # Any unit born/lost on the counterfactual branch has no command in the
        # fixed historical policy; the interpreter lets it stay. Extra obsolete IDs are ignored.
        original=frames[t+delta-1][0]['observation']['players']
        if delta>1:
            fallback+=sum(len(set(pl[2])-set(mapping[s].values())) for s,pl in enumerate(teams))
            for s,pl in enumerate(teams):
                commands[s]={u if u in pl[1] else mapping[s][u]:a for u,a in commands[s].items() if u in pl[1] or u in mapping[s]}
        homes={p:j for j,pl in enumerate(teams) for p in pl[1].values()}
        groups=landings(teams,commands,cfg)
        for p,ws in groups.items():
            bank=homes.get(p)
            collectors=[w for w in ws if w[3]=='COLLECTOR']
            lightest=min((w[2] for w in collectors),default=None)
            first=[w for w in collectors if w[2]==lightest]
            winner=ws[0] if len(ws)==1 else (first[0] if len(first)==1 else None)
            for ss,uid,load,k in ws:
                if bank is not None:received[bank]+=load
                if load and ((bank is not None and bank!=ss) or
                            (bank is None and len(ws)>1 and (winner is None or winner[0]!=ss))):
                    lost[ss]+=load
        board=m._turn(board,cfg,commands)
        actual_after=frames[t+delta][0]['observation']['players']
        # A matching numeric ID alone cannot identify an alternate birth.
        # Only map new workers born on the same turn, same depot and same role.
        for s,new in enumerate(board['players']):
            truth=actual_after[s]
            matched={u:v for u,v in mapping[s].items() if u in truth[2] and v in new[2]}
            births={(w[0],w[2]):u for u,w in new[2].items() if u not in teams[s][2]}
            for u,w in truth[2].items():
                if u not in original[s][2] and (w[0],w[2]) in births:matched[u]=births[w[0],w[2]]
            mapping[s]=matched
        if delta in (1,5,10):
            pl=board['players'];held=[sum(v for p,v,k in s[2].values()) for s in pl]
            orient=lambda v:float(v[side]-v[1-side])
            out[str(delta)]=[round(orient([s[0] for s in pl]),3),round(received[side],3),round(received[1-side],3),
                             round(lost[side],3),round(lost[1-side],3),round(orient(held),3),fallback]
    return out

def run(m,orig,g,side,turns,verify):
    frames=g['steps'];cfg=g['configuration'];straight,cash_error=replay_state(m,g)
    m._ST.clear();m._FC_HISTORY.clear();real=m._search;research=m._reply_choice
    m._ST['tr']=m.BeliefTracker(m._Bel,side);m._ST['rng']=np.random.RandomState(m._SEED);m._ST['prev']=None
    rows=[];tested=0
    for t in range(max(turns)+1):
        frame=frames[t];obs={k:v for k,v in frame[0]['observation'].items() if k!='full_carbon'}
        obs.update(frame[side]['observation']);obs.pop('full_carbon',None);obs['player']=side;pl=obs['players']
        if t in turns:
            snapshot={};clock=[];important=[]
            def capture(o,config,est,root_action,candidates,weights,replies):
                old=[];fresh=[];raw=[];newraw=[]
                root=dict(players=pl,step=t,player=side,trees=o['trees'],carbon=est)
                for move,penalty in candidates:
                    vals=[];revised=[]
                    for enemy in replies:
                        actions=[move,enemy] if side==0 else [enemy,move]
                        b=m._turn(root,config,actions);v=m._reply_assets(b,side,config);vals.append(v)
                        begin=time.perf_counter();adjust=m._post_cargo_adjust(b,side);clock.append((time.perf_counter()-begin)*1000)
                        revised.append(v+adjust)
                    old.append(_post_score(vals,weights,penalty));fresh.append(_post_score(revised,weights,penalty))
                    raw.append(vals);newraw.append(revised)
                a=int(np.argmax(old));b=int(np.argmax(fresh))
                keys={'searched':0,'legacy':a,'corrected':b,'veto':a if fresh[a]>fresh[0]+6. else 0}
                snapshot.update(moves={k:candidates[j][0] for k,j in keys.items()},forecast=dict(old_gap=old[a]-old[0],
                                adjusted_legacy_gap=fresh[a]-fresh[0],adjusted_winner_gap=fresh[b]-fresh[0]),
                                count=len(candidates),replies=len(weights))
                return candidates[a][0]
            m._POST_CAPTURE=capture
            def recheck(*args):
                result=research(*args)
                if hasattr(m,'_POST_PRODUCTION') and snapshot.get('moves'):
                    start=time.perf_counter();actual_old=orig(*args);m._POST_TIMES[0].append((time.perf_counter()-start)*1e3)
                    start=time.perf_counter();actual_new=m._POST_PRODUCTION(*args);m._POST_TIMES[1].append((time.perf_counter()-start)*1e3)
                    assert actual_old==result and actual_new==snapshot['moves']['corrected']
                if verify and tested<2:
                    assert orig(*args)==result
                    snapshot['parity']=True
                return result
            m._reply_choice=recheck
            def searched(o,c,e,base,lw,lb,prev,t0,locked):
                snapshot['called']=True
                z=real(o,c,e,base,lw,lb,prev,t0,locked);snapshot['search_action']=z;return z
            m._search=searched
            before=m._S_STATS['searched']
            m._act(obs,cfg,0.)
            assert snapshot.get('called')
            actual=m._S_STATS['searched']>before
            if snapshot.get('moves') and actual:
                moves=snapshot['moves'];assert moves['searched']==snapshot['search_action']
                keys={k:tuple(sorted(a.items())) for k,a in moves.items()}
                audit=dict(exact_old_parity=bool(snapshot.get('parity')),candidates=snapshot['count'],replies=snapshot['replies'],t=t,clock_ms=float(np.median(clock)),
                           forecast={k:round(v,4) for k,v in snapshot['forecast'].items()},changes={k:changes(moves['searched'],a,pl[side]) for k,a in moves.items()})
                if len(set(keys.values()))>1:
                    audit['searched_action']=moves['searched'];outcomes={}
                    for k,a in moves.items():
                        validate_actions(a,pl[side])
                        if keys[k] not in outcomes:outcomes[keys[k]]=tail(m,straight[t],t,side,cfg,frames,a)
                    audit['outcomes']={k:outcomes[v] for k,v in keys.items()}
                rows.append(audit);tested+=1
        elif t+1 in turns:
            m._search=lambda o,c,e,a,*args:a
            m._act(obs,cfg,None)
        else:
            m._fc_track(obs);tr=m._ST['tr'];est,known=tr.step(pl,t,obs['carbon']);m._sense_tree_carbon(tr.bel,obs,cfg,m._ST['prev'],est,known)
        m._ST['tr'].set_actions(pl,frames[t+1][side]['action'] or {});m._ST['prev']=pl
    m._search=real;m._reply_choice=research
    return rows,cash_error

def main(a):
    root=a.public;manifest=json.loads((root/'index.json').read_text());excluded=set(json.loads((ROOT/'analysis/v7/distill-validation.json').read_text())['excluded_layouts'])
    historical={r['layout'] for r in json.loads((ROOT/'../v7-data/receipts-cache/index.json').read_text())['games']}
    maps=[]
    for r in manifest:
        with gzip.open(root/r['file']) as f:g=json.load(f)
        sig=layout(g);assert sig not in excluded and sig not in historical
        maps.append((hashlib.sha256(('post-carry-1'+sig).encode()).hexdigest(),r,sig))
    maps.sort();m,orig=fixed(a.parent);a.out.mkdir(exist_ok=True,parents=True)
    for j,(_,r,sig) in enumerate(maps):
        dest=a.out/(r['match']+'.json')
        if dest.exists():continue
        with gzip.open(root/r['file']) as f:g=json.load(f)
        side=r['seat'];turns=select(g['steps'],side,sig,a.limit)
        rows,error=run(m,orig,g,side,turns,j==0)
        report=dict(match=r['match'],side=side,map=sig,config_seed=g['configuration']['randomSeed'],fold='dev' if j<7 else 'confirm',
                    requested_turns=len(turns),cash_rebuild_max_error=error,data=rows)
        dest.write_text(json.dumps(report,separators=(',',':')))
        print(j+1,len(maps),len(turns),len(rows),sum(bool(s['changes']['legacy']) for s in rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('public',type=Path);p.add_argument('out',type=Path);p.add_argument('--limit',type=int,default=20)
    main(p.parse_args())
