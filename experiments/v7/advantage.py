"""Score differences between root joint actions using public movement, receipts and risk features."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np

FIT=Path(__file__).with_name('advantage-model.json')
FEATURES=('own_deposit','opponent_deposit','safe_harvest','lost_cargo_risk','lost_worker_risk',
          'enemy_cargo_prize','enemy_planter_prize','equal_cargo_trade','homeward_loaded','early_homeward',
          'homeward_contested','loaded_near_depot','block_enemy_depot','planter_claim','plant_opportunity','protect_tree',
          'collector_hire','planter_hire','low_cash_hires','first_planter_hire','congestion','joint_cargo_collision',
          'crowded_depot','early_mining','late_homeward','threatened_planter','cash_poor_deposit','enemy_tree_tax')

def _adv_joint(obs,stock,rival_heads,actions):
    me=obs['player'];players=obs['players'];us,foe=players[me],players[1-me]
    t=obs['step'];left=LAST_TURN+1-t;cash=us[0];poor=float(cash<110);early=float(t<80)
    friendly=set(us[1].values());foreign=set(foe[1].values())
    if not friendly or not foreign:return np.zeros((len(actions),28),np.float32)
    home=FEAT_DIST[list(friendly)].min(0);e_home=FEAT_DIST[list(foreign)].min(0)
    mytrees={q for q,age in us[3].values()};etrees={q for q,age in foe[3].values()}
    foes=[]
    for q,v,k in foe[2].values():
        z=rival_heads[q].astype(float)/.85;pr=np.exp(z-z.max());pr/=pr.sum()
        foes.append((q,v,k,dict(zip(_ADV_L[q],pr))))
    n=len(actions);out=np.zeros((n,28),np.float32);seen=[{} for _ in actions];planned=[set() for _ in actions]
    for uid,p in us[1].items():
        for j,act in enumerate(actions):
            hire=act.get(uid)
            if hire:
                planned[j].add(p);out[j,16]+=30*(hire=='RECCOLLECTOR');out[j,17]+=30*(hire=='RECPLANTER')
                out[j,18]+=30*poor;out[j,19]+=30*(hire=='RECPLANTER' and not any(k=='PLANTER' for _,_,k in us[2].values()))
    for uid,(p,v,k) in us[2].items():
        opts=[WIDX.get(a.get(uid),0) for a in actions];cache={}
        for a in set(opts):
            q=int(_ADV_L[p,a]);bank=q in friendly;foreign_bank=q in foreign
            collect=k=='COLLECTOR';lose=ties=prize=planter=0.;near=0.
            for ep,ev,ek,prob in foes:
                if FEAT_DIST[p,ep]>4:continue
                mass=prob.get(q,0.)
                if ek=='COLLECTOR' and (not collect or ev<=v):
                    lose=1.-(1.-lose)*(1.-mass);near=max(near,float(FEAT_DIST[q,ep]<=2))
                if collect and ek=='COLLECTOR' and ev>v and not (bank or foreign_bank):prize+=mass*(min(400.,ev)+30.)
                if collect and ek=='PLANTER':planter+=mass
                if collect and ek=='COLLECTOR' and ev==v:ties+=mass
            protect=(q in mytrees);enemy_tree=(q in etrees);wait=a==0
            resource=float(stock[q]);tax=float(q in etrees)
            f=np.zeros(28,np.float32);gain=home[p]-home[q]
            if collect:
                f[0]=v*bank;f[1]=v*foreign_bank;f[2]=.3*resource*wait*(1.-lose)*(not(bank or foreign_bank))*(not enemy_tree)
                f[3]=lose*min(450.,v)*(not(bank or foreign_bank));f[5]=prize;f[6]=30.*planter
                f[7]=30.*ties;f[8]=min(v,400.)*gain/4;f[9]=f[8]*poor
                f[10]=f[8]*near;f[11]=min(v,400.)*(home[q]<=3)*gain/5
                f[12]=25.*(v<35 and e_home[q]<=2 and not foreign_bank)
                f[23]=early*f[2];f[24]=f[8]*max(0.,1.-left/55.);f[26]=f[0]*poor;f[27]=tax*min(v,400.)*.1
            else:
                f[13]=45.*(wait and enemy_tree);f[14]=min(100.,float(np.mean(np.asarray(stock)[_ADV_L[q,1:]])))*wait*(not protect and not enemy_tree and not bank and not foreign_bank)
                f[15]=30.*(protect and wait);f[25]=30.*near
            f[4]=30.*lose;cache[a]=(q,f)
        for j,a in enumerate(opts):
            q,f=cache[a];out[j]+=f
            if q in seen[j]:
                out[j,20]+=30.;out[j,21]+=min(450.,v+seen[j][q])*(q not in friendly)
            if q in planned[j]:out[j,22]+=30.
            seen[j][q]=v
    return out

def build(code):
    w=json.loads(FIT.read_text())
    anchor='def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):'
    score='    best = max(alive, key=lambda c: mean[c])\n    if best == 0'
    if code.count(anchor)!=1 or code.count(score)!=1 or '_ADV_COEF=' in code:raise ValueError('Expected the frozen final root selection')
    setup='_ADV_L=np.asarray([[move(p,a) for a in range(5)] for p in range(225)])\n_ADV_COEF=np.asarray('+repr(w['coef'])+',np.float32)\n'
    add="""    if len(alive)>=2 and _time.perf_counter()-t0<.86:
        finalists=list(dict.fromkeys([0]+list(alive)))
        features=_adv_joint(obs,est,_ST['v5_history']['opponent'][0],[cands[i] for i in finalists])
        corrections=np.clip((features-features[0])@_ADV_COEF,-55.,55.)*WEIGHT
        for i,correction in zip(finalists,corrections):mean[i]+=float(correction)
"""
    add=add.replace('WEIGHT',repr(w['weight']/w['score_scale']))
    return code.replace(anchor,setup+inspect.getsource(_adv_joint)+'\n\n'+anchor).replace(score,add+score)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();src=build(a.parent.read_text());compile(src,str(a.output),'exec')
    with a.output.open('x') as f:f.write(src)
