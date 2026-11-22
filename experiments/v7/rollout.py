"""Use the refitted small policy for hypothetical continuations of each player."""
import base64
import inspect
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent


def _v7_rollout_ids(sides):
    ids=_v5_ids(sides)
    return np.where(ids>=23,21+(ids-23)//2,ids)


def build(code):
    blob=base64.b64encode((HERE/'rollout.npz').read_bytes()).decode()
    code,count=re.subn(r"(?m)^    _V6R_BYTES = '[^']+'$",lambda _:"    _V6R_BYTES = '"+blob+"'",code)
    if count!=1:raise ValueError('expected one embedded rollout policy')
    marker='def _r6_tail('
    line='w,b,_=_V6R_TINY(np.asarray(views),0)'
    if code.count(marker)!=1 or code.count(line)!=1:raise ValueError('unexpected tail search')
    return code.replace(marker,inspect.getsource(_v7_rollout_ids)+'\n\n'+marker).replace(line,
        'w,b,_=_V6R_TINY(np.asarray(views),_v7_rollout_ids([0,1,0,1]))')
