"""Share full-policy references between both sides of similar intermediate search boards."""
import inspect
import numpy as np


def _v7_actor_references(X,part):
    n=len(part)
    # Distances count changes to both teams' occupied squares. Group only within
    # the same sampled rival response; a second reference covers a distinct outcome.
    board=X[::2].reshape(n,NPL,225)[:,[2,4,5,7,8,10],:].reshape(n,-1)>.15
    groups={}
    for i,b in enumerate(part):groups.setdefault(b[1],[]).append(i)
    indices=[];ref_for=np.zeros(n,dtype=int)
    for members in groups.values():
        members=np.asarray(members);states=board[members]
        first=members[0]
        delta=(states!=board[first]).sum(1)
        refs=[first]
        if len(members)>=4 and max(delta)>=6:
            refs.append(members[int(delta.argmax())])
        distances=np.asarray([(states!=board[ref]).sum(1) for ref in refs])
        ref_for[members]=len(indices)+distances.argmin(0)
        indices.extend(refs)
    return np.asarray(indices),ref_for


def _v7_joint_interim(net,X,S,enemy,part,own,safeguard=True):
    n=len(part)
    refs,assignment=_v7_actor_references(X,part)
    if len(refs)*2>=n:return _v6r_counterfactuals(net,X,S,enemy,part,own)
    teacher_rows=np.column_stack((2*refs,2*refs+1)).ravel()
    actual=[own,enemy]*n;reference_sides=[own,enemy]*len(refs)
    tw,tb,tv=net(X[teacher_rows],_v5_ids(reference_sides),S[teacher_rows],play_sides=reference_sides)
    sw,sb,sv=_V6R_TINY(X,_v7_rollout_ids(actual),S)
    teacher_index=(2*assignment[:,None]+np.arange(2)).ravel()
    source=teacher_rows[teacher_index]
    cells=X.reshape(2*n,NPL,225)
    ok=(cells[:,2]==cells[source,2])&(cells[:,4]==cells[source,4])
    ok&=(cells[:,25:35]==cells[source,25:35]).all(1)
    w=sw+(tw[teacher_index]-sw[source])*ok[:,:,None]
    b=sb+(tb[teacher_index]-sb[source])
    v=sv+tv[teacher_index]-sv[source]
    w[teacher_rows],b[teacher_rows],v[teacher_rows]=tw,tb,tv
    if safeguard and n-len(refs)>=3:
        # Recheck the most uncertain own continuation, especially when a contested
        # collector changes location relative to its reference state.
        local=cells[::2]
        units=(local[:,2]>.5)|(local[:,4]>.5)
        top=np.partition(w[::2],-2,axis=2)
        gap=top[:,:,-1]-top[:,:,-2]
        disputed=units*((~ok[::2]).astype(float)+.35)*(.4+local[:,3]+.4*local[:,24])
        uncertainty=(disputed/(.15+np.maximum(0.,gap))).sum(1)
        uncertainty[refs]=-1.
        count=max(1,(n-len(refs))//3)
        rows=np.argsort(-uncertainty)[:count]*2
        ew,eb,ev=net(X[rows],_v5_ids([own]*count),S[rows],play_sides=[own]*count)
        w[rows],b[rows],v[rows]=ew,eb,ev
    return w,b,.5*(v[::2]-v[1::2])


def build(code):
    anchor='def _v6r_counterfactuals('
    old='                    lw,lb,vv = _v6r_counterfactuals(net,X,S,op,part,me)'
    if code.count(anchor)!=1 or code.count(old)!=1:raise ValueError('expected joint-action intermediate estimate')
    funcs='\n\n'.join(inspect.getsource(f) for f in (_v7_actor_references,_v7_joint_interim))
    return code.replace(anchor,funcs+'\n\n'+anchor).replace(old,'                    lw,lb,vv = _v7_joint_interim(net,X,S,op,part,me)')
