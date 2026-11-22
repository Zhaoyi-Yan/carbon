"""Estimate whether funded collectors still have a usable return route."""
_V5_CROSS=[[(p//15+dr)%15*15+(p%15+dc)%15 for dr,dc in ((-1,0),(0,1),(1,0),(0,-1))] for p in range(225)]
_V5_PATHS = {}


def _v5_bank_paths(bases):
    if bases not in _V5_PATHS:
        dist=_v5_home(bases)
        paths=[]
        for p in range(225):
            first=[]
            for q in _V5_CROSS[p]:
                if dist[q]>=dist[p]:continue
                near=[r for r in _V5_CROSS[q] if dist[r]<dist[q]] if dist[q] else [q]
                first.append((q,near))
            paths.append(first)
        _V5_PATHS[bases]=paths
    return _V5_PATHS[bases]


def _v5_access(pl,side,left):
    us,opp=pl[side],pl[1-side]
    bases=tuple(us[1].values());home=_v5_home(bases);paths=_v5_bank_paths(bases)
    hunters=[(p,v) for p,v,k in opp[2].values() if k=='COLLECTOR']
    denied=0.;exits=0.;at_risk=0.
    for pos,cargo,kind in us[2].values():
        if kind!='COLLECTOR' or cargo<15 or not paths[pos] or home[pos]>left:continue
        local=[p for p,v in hunters if v<=cargo and _V5_DIST[pos][p]<=4]
        if not local:continue
        route=0.
        for q,further in paths[pos]:
            if q in bases:
                route=1.;break
            enemy=min(_V5_DIST[q][p] for p in local)
            first=.16 if enemy<=1 else (.65 if enemy==2 else 1.)
            escape=0.
            for r in further:
                if r in bases:escape=1.;break
                depth=min(_V5_DIST[r][p] for p in local)
                escape=max(escape,.30 if depth<=1 else (.65 if depth==2 else 1.))
            route=max(route,first*(.5+.5*escape))
        lost=1.-route
        denied+=lost*min(450.,cargo)/(1.+.075*home[pos])
        exits+=lost
        at_risk+=float(route<.5)
    return denied/100.,exits,at_risk


def _v5_corridor_features(pl,step,last,side):
    left=last+1-step
    own=np.asarray(_v5_access(pl,side,left));enemy=np.asarray(_v5_access(pl,1-side,left))
    diff=own-enemy
    cashrisk=lambda i:1./(1.+pl[i][0]/80.)
    return np.r_[diff,diff*max(0.,1.-left/55.),own*cashrisk(side)-enemy*cashrisk(1-side)]


def _v5_corridor_score(board,me):
    pl=board['players'];step=board['step'];left=LAST_TURN+1-step
    fit=_v5_corridor_features(pl,step,LAST_TURN,me)@_V5['corridors']
    trees=board['_st'].trees_meta if board.get('_st') is not None else board.get('trees',{})
    def stopped(side):
        side_pl=pl[side];workers=side_pl[2].values()
        return not side_pl[3] and side_pl[0]<20 and not any(k=='COLLECTOR' for p,v,k in workers)
    # A lone unfunded planter cannot secure a new income stream without an external transfer.
    stranded=float(stopped(1-me))-float(stopped(me))
    return _v5_cashflow_eval(board,me)+(np.clip(fit,-.7,.7)+.32*stranded)*min(1.,left/35.)/3.
