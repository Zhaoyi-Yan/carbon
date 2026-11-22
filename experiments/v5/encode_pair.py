"""Compute the two simulated perspectives together and reuse common board geometry."""
_V5_POINTS=np.arange(225)
_V5_REACH=np.array([[move(p,a) for a in range(5)] for p in _V5_POINTS])
_V5_TWO=np.array([np.flatnonzero(FEAT_DIST[p]<=2) for p in _V5_POINTS])


def _v5_pair(players,step,carbon,previous):
    views=np.zeros((2,NPL,N),dtype=np.float32)
    views[:,0]=np.asarray(carbon,dtype=np.float32)/100.
    views[:,1]=1.
    views[:,18]=step/300.
    left=LAST_TURN-step
    views[:,37]=2.4*max(0.,1.-left/16.)
    views[:,38]=2.4*max(0.,1.-left/6.)
    roles=[]
    for side,pl in enumerate(players):
        me,op=views[side],views[1-side]
        units=list(pl[2].values())
        cols=[(p,v) for p,v,k in units if k=='COLLECTOR']
        cp=np.array([p for p,v in cols],int)
        loads=np.array([v for p,v in cols],float)
        pp=np.array([p for p,v,k in units if k=='PLANTER'],int)
        tp=np.array([p for p,age in pl[3].values()],int)
        age=np.array([age for p,age in pl[3].values()],int)
        base=list(pl[1].values())
        me[2,cp]=op[5,cp]=1.
        me[3,cp]=op[6,cp]=np.log1p(loads)/7.
        me[4,pp]=op[7,pp]=1.
        me[8,tp]=op[10,tp]=1.
        me[9,tp]=op[11,tp]=age/50.
        me[12,base]=op[13,base]=1.
        if previous is not None:
            pc=[p for p,v,k in previous[side][2].values() if k=='COLLECTOR']
            pq=[p for p,v,k in previous[side][2].values() if k=='PLANTER']
            me[14,pc]=op[16,pc]=1.
            me[15,pq]=op[17,pq]=1.
        me[19]=op[20]=np.log1p(pl[0])/9.
        me[21]=op[22]=len(units)/10.
        me[23]=op[24]=len(tp)/20.
        home=np.asarray(_v5_home(tuple(base)))
        me[39]=op[40]=home/14.
        me[41]=home>left+1
        around=_V5_REACH[cp].ravel()
        reach=np.bincount(around,minlength=N)
        minimum=np.full(N,np.inf)
        np.minimum.at(minimum,around,np.repeat(loads,5))
        safe_min=np.where(reach>0,minimum,0.)
        me[44]=np.minimum(10,reach)/4.
        op[35]=reach>0
        me[45]=op[36]=np.log1p(safe_min)/7.
        wide=_V5_TWO[cp].ravel()
        wide_count=np.bincount(wide,minlength=N)
        wide_min=np.full(N,np.inf)
        np.minimum.at(wide_min,wide,np.repeat(loads,_V5_TWO.shape[1]))
        op[42]=np.minimum(wide_count,10)/4.
        op[43]=np.log1p(np.where(wide_count>0,wide_min,0.))/7.
        planters=np.bincount(_V5_REACH[pp].ravel(),minlength=N)
        me[46]=op[47]=np.minimum(planters,10)/4.
        me[52]=op[53]=np.log1p(loads.sum())/9.
        margin=pl[0]-players[1-side][0]
        me[50]=op[51]=np.clip(margin/2000.,0.,2.5)
        roles.append((cp,loads,pp,tp,age,minimum,planters))
    for side in (0,1):
        me,op=views[side],views[1-side]
        cp,loads,pp,tp,age,_,_=roles[side]
        danger,p_reach=roles[1-side][-2:]
        contact=tp[p_reach[tp]>0]
        life=1.-age[p_reach[tp]>0]/50.
        me[49,contact]=op[48,contact]=np.maximum(0.,life)
        d=danger[_V5_REACH[cp]]
        me[25:30,cp]=(d<loads[:,None]).T
        me[30:35,cp]=(d==loads[:,None]).T
        me[25:30,pp]=np.isfinite(danger[_V5_REACH[pp]]).T
        me[30:35,pp]=(p_reach[_V5_REACH[pp]]>0).T
    return _quant(views.reshape(2,NPL,SIZE,SIZE))
