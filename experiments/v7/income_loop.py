"""One low-cargo miner may finance a missing hire from a nearby safe patch."""
import inspect
import numpy as np


def _v7_il_recent(obs,mem):
    t=obs['step'];players=obs['players']
    old=mem.get('previous')
    if not t or mem.get('last')!=t-1:
        mem.clear();old=None
    seen=mem.setdefault('seen',np.full(225,-100))
    for q,_,role in players[obs['player']][2].values():
        if role=='COLLECTOR':seen[q]=t
        else:seen[_V7_IL_L[q,1:]]=t
    trail=mem.setdefault('flows',[])
    found=[[0.,0.],[0.,0.]]
    if old is not None:
        for side,p in enumerate(players):
            homes=p[1].values()
            for uid,(q,load,role) in p[2].items():
                if role!='COLLECTOR' or uid not in old[side][2]:continue
                before,c,_=old[side][2][uid]
                if q==before and q not in homes:found[side][1]+=min(32.,max(0.,load-c))
                if before!=q and q in homes and load==0:found[side][0]+=c
    trail.append(found)
    del trail[:-8]
    mem['previous']=players;mem['last']=t
    return np.sum(trail,axis=0)


def _v7_il_step(pos,target,base,blocked,enemy,tax,home=False):
    """Choose a safe first step; at a friendly depot a carrier can still bank on collision."""
    dist=_V7_IL_D[target];best=None
    for a,q in enumerate(_V7_IL_L[pos]):
        if q in blocked or (q!=target and dist[q]>=dist[pos]) or (q==pos and pos!=target):continue
        danger=min((_V7_IL_D[e][q] for e in enemy),default=15)
        if danger<=1 and not (home and q==target):continue
        score=-float(dist[q])-.25*tax[q]-.22*(danger==2)+.08*(a==base)
        if best is None or score>best[0]:best=(score,a)
    return None if best is None else best[1]


def _v7_income_loop(obs,cfg,stock,base,mem):
    side=obs['player'];t=obs['step'];own,opp=obs['players'][side],obs['players'][1-side]
    recent=_v7_il_recent(obs,mem)
    current=mem.get('mission')
    price=cfg.get('recCollectorCost',30)
    rate=sum(obs.get('trees',{}).get(k,(None,0))[1] for k in own[3])
    reserve=own[0]+4.*rate
    def stop(reason):
        if 'mission' in mem:
            mem['event']=('stop',mem['mission']['uid'],reason);mem.pop('mission')
            mem['retry']=t+3
        return base
    mem['event']=None
    if t<5 or t>155 or reserve>=price or len(own[2])>=8:
        return stop('financed_or_outside_window')
    homes=list(own[1].values());eh=set(opp[1].values())
    if not homes:return stop('no_depot')
    hw=_V7_IL_D[homes].min(0)
    if current is None:
        if t<mem.get('retry',0):
            return base
        # A short carried load will pay for the hire without diverting another worker.
        if any(k=='COLLECTOR' and c>=price-own[0] and c>=22. and hw[q]<=4 for q,c,k in own[2].values()):
            return base
        fl=_v7_receipts(obs['players'],t,cfg.get('episodeSteps',300)-2)
        if reserve+fl[side,0]>=price:return base
        # Observe income through surviving workers only. Captures, unmatched worker IDs
        # and cargo on invisible soil are not used to infer enemy commands.
        past=recent[:,0]+.6*recent[:,1]
        if past[1-side]-past[side]<10. and fl[1-side,0]-fl[side,0]<12.:
            return base
    elif current['uid'] not in own[2] or t-current['start']>12:
        return stop('lost_or_expired')
    trees={p for pl in obs['players'] for p,a in pl[3].values()}
    tax=np.zeros(225)
    for q,age in opp[3].values():
        if age>=49:continue
        y,x=divmod(q,15)
        for dy in (-1,0,1):
            for dx in (-1,0,1):tax[((y+dy)%15)*15+(x+dx)%15]+=cfg.get('collectorAbsorptionRate',.1)
    rivals=[(p,c) for p,c,k in opp[2].values() if k=='COLLECTOR']
    orig={u:WIDX.get(base.get(u),0) for u in own[2]}
    ends={u:int(_V7_IL_L[v[0],orig[u]]) for u,v in own[2].items()}
    hires={p for b,p in own[1].items() if b in base}
    stock=np.asarray(stock)
    if current is None:
        best=None
        for u,(pos,load,k) in own[2].items():
            if k!='COLLECTOR' or load>14. or hw[pos]>6:continue
            blocked={p for key,p in ends.items() if key!=u}|hires|eh
            threats=[q for q,c in rivals if c<=max(load,price+8.)]
            enemies=_V7_IL_D[threats].min(0) if threats else np.full(225,15)
            # Discount cells without a recent own observation; the stock estimate is fog-limited.
            harvest_rate=max(0.,cfg.get('initialCollectRate',.30)-cfg.get('collectDecreaseRate',0.)*len(own[2]))
            possible=np.flatnonzero((hw>=1)&(hw<=4)&(_V7_IL_D[pos]<=3)&(stock>=32.)&(enemies>=3)&(tax<.20))
            for target in possible:
                if target in trees or target in eh or target in blocked:continue
                d=int(_V7_IL_D[pos,target]);back=int(hw[target])
                reliable=1. if t-mem['seen'][target]<=8 else .67
                incoming=(load+reliable*stock[target]*(1-(1-harvest_rate)**3))*(1.-min(.5,tax[target]))**(back+2)
                if incoming<max(20.,price-own[0]+4.):continue
                a=_v7_il_step(pos,target,orig[u],blocked,threats,tax)
                if a is None:continue
                merit=min(incoming,price*2.)/(d+3.+back)-.32*(enemies[target]==3)+.20*(a==orig[u])
                if best is None or merit>best[0]:best=(merit,u,int(target))
        if best is None:return base
        _,u,spot=best
        current=dict(uid=u,spot=spot,start=t,stays=0,banking=False)
        mem['mission']=current;mem['event']=('assign',u,spot)
    u=current['uid'];pos,load,_=own[2][u];spot=current['spot']
    blocked={p for key,p in ends.items() if key!=u}|hires|eh
    threats=[q for q,c in rivals if c<=load+3.]
    if current['banking'] and pos in homes:return stop('delivered')
    if current['stays']>=3 or (current['stays']>=2 and load+own[0]>=price+2.):
        current['banking']=True
    if not current['banking'] and (spot in trees or (pos==spot and stock[spot]<12.) or any(_V7_IL_D[q,spot]<=1 for q,c in rivals if c<=load+12.)):
        if load>=18.:current['banking']=True
        else:return stop('site_unsafe_or_depleted')
    if current['banking']:
        routes=[]
        for target in homes:
            a=_v7_il_step(pos,target,orig[u],blocked,threats,tax,True)
            if a is not None:routes.append((int(_V7_IL_D[pos,target])+.25*tax[_V7_IL_L[pos,a]]-.05*(a==orig[u]),a))
        action=min(routes)[1] if routes else None
    else:
        action=_v7_il_step(pos,spot,orig[u],blocked,threats,tax)
        if action==0:current['stays']+=1
    if action is None:return stop('blocked')
    if action==orig[u]:return base
    altered=dict(base)
    if action:altered[u]=WACT[action]
    else:altered.pop(u,None)
    return altered


def build(code):
    if __package__:from . import receipts
    else:import receipts
    if 'def _v7_receipts(' not in code:code=receipts.build(code)
    root='def _act(obs, configuration=None, t0=None):'
    point="    st['tr'].set_actions(players, act)"
    if code.count(root)!=1 or code.count(point)!=1 or 'def _v7_income_loop(' in code:raise ValueError('Expected compatible root')
    geometry="_V7_IL_D=np.asarray(_V7_RECEIPT_DIST)\n_V7_IL_L=np.asarray([[move(p,a) for a in range(5)] for p in range(225)])\n"
    functions='\n\n'.join(inspect.getsource(fn) for fn in (_v7_il_recent,_v7_il_step,_v7_income_loop))
    code=code.replace(root,geometry+functions+'\n\n'+root)
    return code.replace(point,"    if configuration is not None:\n        act=_v7_income_loop(obs,configuration,est,act,st.setdefault('income_loop',{}))\n"+point)
