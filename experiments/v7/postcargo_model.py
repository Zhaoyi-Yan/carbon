"""Research-only cargo adjustment for the final collision critic, excluding income from future mining."""
import numpy as np


def _post_cargo_adjust(board,side):
    players=board['players'];step=board['step'];remaining=max(0,LAST_TURN+1-step)
    if remaining==0:return 0.
    x,load,dist,seat=_v7_receipt_rows(players,step,LAST_TURN)
    if not len(load) or not np.any(load>0):return 0.
    model=_V7_RECEIPT_MODEL
    h=np.tanh(x@model['w0']+model['b0']);logits=(h@model['w1']+model['b1'])[:,:11].copy()
    reach=_V7_RECEIPT_STEPS[None,:]>=np.maximum(1,dist[:,None])
    logits[:,:10]=np.where(reach,logits[:,:10],-100.)
    p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    p5=p[:,:min(remaining,5)].sum(1);p10=p[:,:min(remaining,10)].sum(1)
    tail=.5*np.clip((remaining-10.)/12.,0.,1.)
    worth=load*(.12*p5+.86*p10+tail*(1.-p10))*(dist<=remaining)
    legacy=load*(dist<=remaining)*(.84/(1.+.035*dist))
    gain=worth-legacy
    return float(gain[seat==side].sum()-gain[seat!=side].sum())


def _post_score(vals,weights,penalty):
    vals=np.asarray(vals);order=np.argsort(vals);left=.25;low=0.
    for i in order:
        part=min(left,weights[i]);left-=part;low+=part*vals[i]
        if left<=1e-9:break
    return float(.7*(vals@weights)+.3*low/.25-penalty)
