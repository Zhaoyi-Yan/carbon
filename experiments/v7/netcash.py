"""Predict incremental cash over 20/35 turns and cap its effect on the search asset term."""
import inspect
from pathlib import Path
import numpy as np

MODEL=Path(__file__).with_name('netcash-model.npz')


def _v7_nc_inputs(players,meta,step,last,flow,public,simple_asset=None):
    """Public quantities in natural units, with exact seat antisymmetry."""
    sides=[]
    left=last+1-step
    for i,p in enumerate(players):
        cash=p[0]; workers=p[2];trees=p[3]
        cc=[(q,v) for q,v,r in workers.values() if r=='COLLECTOR']
        plant=len(workers)-len(cc)
        loads=sum(v for q,v in cc)
        income=pay20=pay35=age=0.
        for uid,(_,a) in trees.items():
            pay=meta.get(uid,(None,0.))[1];life=max(0,50-a)
            income+=pay;pay20+=pay*min(20,life,left);pay35+=pay*min(35,life,left);age+=min(35,life)
        f5,f10=flow[i]
        onboard=loads-f10
        # A receipt spike can be the liquidation of carried inventory.
        # Replenishment depends on the surviving fleet and on other cash-producing assets.
        sides.append((f5,f10,f10-f5,onboard,loads,len(cc),plant,len(workers),len(trees),income,pay20,pay35,age,
                      min(cash,30.),min(cash,60.),min(cash,150.),min(cash,600.),
                      f10/(1.+len(cc)),f10/(1.+loads/80.),f10*min(1.,cash/90.),
                      f10*min(1.,len(cc)/6.),onboard*min(1.,len(cc)/6.),loads/(1.+len(cc)),
                      max(0.,8-len(workers))*min(1.,cash/100.),np.log1p(max(0.,cash))))
    delta=np.r_[np.asarray(sides[0])-sides[1],public]
    phase=step/max(1,last)
    x=np.concatenate((delta,delta*max(0.,1.-step/90.),delta*phase,delta*max(0.,1.-left/100.)))
    if simple_asset is None:return x
    return np.r_[x,simple_asset*np.asarray([1.,max(0.,1.-step/90.),phase,max(0.,1.-left/100.)])]


def _v7_nc_leaf(board,side,old=None):
    step=board['step'];left=LAST_TURN+1-step
    if left<35:return _r6_asset_score(board,side) if old is None else old
    if '_v7_nc_assets' not in board:
        board['_v7_nc_assets']=[None,None]
        raw=_r6_asset_score(board,side)
        if old is None:old=raw
    elif old is None:old=_r6_asset_score(board,side)
    if '_v7_nc_result' not in board:
        if '_v7_flow' not in board:board['_v7_flow']=_v7_receipts(board['players'],step,LAST_TURN)
        if '_v7_nc_public' not in board:board['_v7_nc_public']=_fc_predict(board,LAST_TURN,_FC_HISTORY)
        st=board.get('_st');meta=st.trees_meta if st is not None else board.get('trees',{})
        simple=board['_v7_nc_assets'][0]-board['_v7_nc_assets'][1]
        x=_v7_nc_inputs(board['players'],meta,step,LAST_TURN,board['_v7_flow'],board['_v7_nc_public'],simple)
        heads=x@_V7_NC_COEF
        cash=board['players'][0][0]-board['players'][1][0]
        board['_v7_nc_result']=(cash+.55*heads[0]+.45*heads[1])/3000.
    predicted=board['_v7_nc_result']*(1-2*side)
    return old+_V7_NC_WEIGHT*min(.10,max(-.10,predicted-old))


def build(code,weight=.45):
    """Blend into the asset term used at search leaves; tail rollouts keep their evaluator."""
    if __package__:
        from . import receipts
    else:
        import receipts
    if not 0<=weight<=1:raise ValueError('weight outside [0,1]')
    if 'def _v7_nc_leaf(' in code:raise ValueError('netcash already applied')
    if 'def _v7_receipts(' not in code:code=receipts.build(code)
    root='def _act(obs, configuration=None, t0=None):'
    ptr='value += 0.26 * (_r6_asset_score(future, me) - value)'
    old='    public=_fc_predict(board,last,history)*sign'
    if any(code.count(k)!=1 for k in (ptr,root,old)):raise ValueError('Expected compatible cash and search functions')
    with np.load(MODEL,allow_pickle=False) as f:coef=f['coef'].tolist()
    chunk='_V7_NC_COEF=np.asarray('+repr(coef)+',dtype=float)\n_V7_NC_WEIGHT='+repr(weight)+'\n'
    chunk+='\n\n'.join(inspect.getsource(f) for f in (_v7_nc_inputs,_v7_nc_leaf))+'\n\n'
    code=code.replace(old,"    if '_v7_nc_public' not in board:board['_v7_nc_public']=_fc_predict(board,last,history)\n    public=board['_v7_nc_public']*sign")
    # Use the original per-unit valuations while they are already available in the asset scorer.
    # The later sapling and receipt adjustments vary between candidates. This public subscore
    # deliberately reconstructs the same frozen simplified asset feature used in training.
    asset_start=code.index('def _r6_asset_score(')
    marker="        c=sum(p[2][uid][1]*min(1.,max(-.45,row@_R6_CARGO)) for uid,row in car.items())"
    at=code.index(marker,asset_start)+len(marker)
    added="""
        if '_v7_nc_assets' in board and board['_v7_nc_assets'][me] is None:
            nc_count=sum(k=='COLLECTOR' for _,_,k in p[2].values())
            nc_work=34.*min(7,nc_count)+46.*min(3,len(p[2])-nc_count)
            nc_tree=sum(min(300.,max(-90.,float(row@_R6_TREE))) for row in woods.values())
            board['_v7_nc_assets'][me]=c+nc_tree+min(1.,max(0,LAST_TURN+1-t)/24.)*nc_work"""
    code=code[:at]+added+code[at:]
    return code.replace(root,chunk+root).replace(ptr,'value += 0.26 * (_v7_nc_leaf(future,me) - value)')
