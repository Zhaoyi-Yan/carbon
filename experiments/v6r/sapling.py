"""Use estimated neighborhood carbon to refine the value of a recently planted tree."""
import argparse
import inspect
import json
from pathlib import Path


FEATURES=('bias','original_asset','still_life','uncontested_ground','shared_ground','blocked_ground',
          'best_soil','empty_ground_cells','contested_ground','guarded_ground','own_nearby_trees',
          'enemy_nearby_trees','tax_now','tax_approaching','tax_near_depot','enemy_planter_threat',
          'rate_now','rate_times_threat','age_fraction','soil_times_age')


def _sapling_context(players,old):
    cells={}
    by_side=[{},{}]
    for j,p in enumerate(players):
        for cell,age in p[3].values():
            cells[cell]=(j,age)
        for wid,(cell,load,role) in p[2].items():
            prior=old[j][2].get(wid) if old else None
            by_side[j][cell]=(role,load,prior[0] if prior else None)
    return cells,by_side


def _sapling_features(board,meta,side,uid,base,context,last):
    players=board['players']; pl=players[side]
    pos,age=pl[3][uid]; turns=min(50-age,last+1-board['step'])
    horizon=min(36,max(0,turns))
    factor=horizon/36
    cells,workers=context
    carbon=board['carbon']
    old_income=meta.get(uid,(None,0))[1]
    rival=workers[1-side]
    econ=[0.,0.,0.]
    indiv=[]
    for q in NB[pos]:
        if q in cells:
            continue
        hits=sum(x in cells for x in NB[q])
        if not hits:continue
        # Discount the present soil stock as several trees deplete it each turn.
        ratio=.968*(1-.0375*hits)
        value=float(carbon[q])*.0375*(1-ratio**horizon)/(1-ratio)
        resident=workers[0].get(q) or workers[1].get(q)
        idx=2 if resident is not None and resident[0]=='COLLECTOR' and resident[2]==q else int(hits>1)
        econ[idx]+=value
        indiv.append(value)
    nearest_thief=min((DIST[pos][q] for q,w in rival.items() if w[0]=='PLANTER'),default=12)
    nearest_guard=min((DIST[pos][q] for q,w in workers[side].items() if w[0]=='COLLECTOR' and w[1]<25),default=12)
    threat=max(0.,1-nearest_thief/5.)
    defense=max(0.,1-nearest_guard/5.)
    tax=near=0.
    for q,(role,load,prior) in rival.items():
        if role!='COLLECTOR':continue
        if q==pos or q in NB8[pos]:
            tax+=min(load,600.)*.10
        elif DIST[pos][q]<=4 and prior is not None and DIST[pos][prior]>DIST[pos][q]:
            near+=min(load,600.)*.10
    closest_base=min(DIST[pos][q] for q in players[1-side][1].values())
    owned=sum(v[0]==side for q,v in cells.items() if q!=pos and DIST[pos][q]<=2)
    foreign=sum(v[0]!=side for q,v in cells.items() if DIST[pos][q]<=2)
    soil=sum(econ)/100
    return [factor,base/100,factor*factor,*[v/100 for v in econ],max(indiv,default=0.)/100,
            factor*len(indiv)/4,soil*threat,soil*defense,factor*min(4,owned)/4,
            factor*min(4,foreign)/4,factor*tax/20,factor*near/20,factor*tax/20*max(0.,1-closest_base/6),
            factor*threat,factor*min(old_income,35)/14,factor*min(old_income,35)/14*threat,
            factor*age/6.,soil*age/6.]


def _sapling_total(board,meta,side,woods,context,last):
    total=0.
    for uid,row in woods.items():
        base=float(row@_R6_TREE)
        age=board['players'][side][3][uid][1]
        value=base
        if age<=6 and board['step']<last+1:
            group=0 if age==1 else (1 if age<=3 else 2)
            features=_sapling_features(board,meta,side,uid,base,context,last)
            value+=sum(a*b for a,b in zip(features,_SAPLING_COEFS[group]))
        total+=min(300.,max(-90.,value))
    return total


def export(source,model,output):
    text=source.read_text();data=json.loads(model.read_text())
    pre='_SAPLING_COEFS = '+repr(data['coefficients'])+'\n\n'
    for fn in (_sapling_context,_sapling_features,_sapling_total):
        pre+=inspect.getsource(fn)+'\n\n'
    def change(a,b):
        nonlocal text
        if text.count(a)!=1:raise ValueError('Unexpected asset search anchor '+a[:40])
        text=text.replace(a,b,1)
    change('def _r6_asset_score(board,side):',pre+'def _r6_asset_score(board,side):')
    change("    valuations=[]\n", "    valuations=[]\n    sap_context=_sapling_context(players,board.get('_sap_previous')) if any(age<=6 for p in players for _,age in p[3].values()) else None\n")
    change("tree=sum(float(np.clip(row@_R6_TREE,-90,300.)) for row in woods.values())",
           "tree=_sapling_total(board,meta,me,woods,sap_context,LAST_TURN)")
    change("return {'players': s2.players, 'step': s2.step, 'carbon': s2.carbon, '_st': s2}",
           "return {'players': s2.players, 'step': s2.step, 'carbon': s2.carbon, '_st': s2, '_sap_previous':o['players']}")
    compile(text,str(output),'exec')
    output.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','model','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();export(a.base,a.model,a.output)
