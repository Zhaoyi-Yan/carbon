"""Map-symmetric probability features and calibrated scalar score transforms."""
import numpy as np


def win_basis(turn):
    turns=np.asarray(turn,dtype="f")
    nodes=np.array([25.,105.,190.,270.],dtype="f")
    right=np.clip(np.searchsorted(nodes,turns),1,3)
    inter=np.clip((turns-nodes[right-1])/(nodes[right]-nodes[right-1]),0.,1.)
    basis=np.zeros((len(turns),4),"f");rows=np.arange(len(turns))
    basis[rows,right-1]=1.-inter;basis[rows,right]=inter
    return basis


def win_pieces(score,turn):
    u=score/(160.+1.6*(299-turn[:,None]))
    v=np.abs(u)
    shape=np.stack((np.minimum(v,1.),np.clip(v-1.,0,2.),np.clip(v-3.,0,5.)),axis=-1)*np.sign(u)[:,:,None]
    return (shape[:,:,:,None]*win_basis(turn)[:,None,None,:]).reshape(len(u),u.shape[1],12)


def win_sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-20,20)))
