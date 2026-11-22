"""Compare tactical replies by the probability of preserving or overturning the terminal lead."""
import inspect,json
from pathlib import Path
import numpy as np


def _v7_risk_score(vals,weights,lower,penalty,step):
    vals=np.asarray(vals);mean=float(weights@vals)
    scale=float(np.interp(step,_V7_RISK_TURNS,_V7_RISK_SCALE))
    # Express expected win probability in units of cash so action priors keep their units.
    # Center at the mean before clipping to avoid flattening every option in a decided game.
    effective=np.clip(mean,-4.*scale,4.*scale)
    z=np.clip((vals-mean+effective)/scale,-15.,15.)
    win=float(weights@(1./(1.+np.exp(-z))))
    equiv=mean-effective+scale*(np.log(win)-np.log1p(-win))
    # Rare credible replies deserve more protection when defending a lead; a trailing team
    # can accept a bad reply in exchange for a visible path to recovering the difference.
    conserve=.08+.24/(1.+np.exp(-np.clip(mean/scale,-12.,12.)))
    return float((1.-conserve)*equiv+conserve*lower-penalty)


def build(code):
    data=json.loads(Path(__file__).with_name('win-risk-model.json').read_text())
    anchor='def _reply_choice(obs,config,est,act,logits,locked,t0):'
    old='        score=.70*float(weights@vals)+.30*lower/.25-penalty'
    if code.count(anchor)!=1 or code.count(old)!=1:raise ValueError('unexpected reply utility')
    helper='_V7_RISK_TURNS='+repr(data['turns'])+'\n_V7_RISK_SCALE='+repr(data['scales'])+'\n'+inspect.getsource(_v7_risk_score)
    return code.replace(anchor,helper+'\n\n'+anchor).replace(old,'        score=_v7_risk_score(vals,weights,lower/.25,penalty,step)')
