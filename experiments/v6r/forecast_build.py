"""Embed the short-horizon forecast in a compatible root-search leaf, preserving the native critic."""
import argparse
import base64
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def build(source,weight=.25,use20=1.,mode="supplement"):
    if not (0<=weight<=1 and 0<=use20<=1):raise ValueError("weights must be in [0,1]")
    if "def _fc_heads(" in source:raise ValueError("forecast already present")
    blob=base64.b64encode((HERE/"forecast-model.npz").read_bytes()).decode()
    calibration=json.loads((HERE/"forecast-blend.json").read_text())
    engine=(HERE/"forecast.py").read_text()
    engine+='\nwith np.load(io.BytesIO(base64.b64decode('+repr(blob)+')),allow_pickle=False) as _fc_file:\n'
    engine+='    _FC_WEIGHTS = tuple(_fc_file[k] for k in ("mean","inv","ridge","w1","b1","w2","b2","w3","b3"))\n'
    for key,v in (("BLEND",calibration["blend"]["coef"]),("TERR",calibration["legacy_territory"]["coef"]),("ALPHA",calibration["horizon_mix"])):
        engine+='_FC_'+key+' = np.asarray('+repr(v)+')\n'
    engine+='_FC_MIX = '+repr(weight)+'\n_FC_USE20 = '+repr(use20)+'\n'
    root='def _act(obs, configuration=None, t0=None):'
    native='value += _V5_VALUE_MIX * (_v5_value(future, me) - value)'
    if source.count(root)!=1 or source.count(native)!=1:raise ValueError("check root and leaf entry points on this strategy")
    source=source.replace(root,engine+'\n'+root+'\n    _fc_track(obs)')
    if mode=="supplement":
        replacement='value += _V5_VALUE_MIX * (_fc_leaf(future, me, LAST_TURN, _v5_value(future, me)) - value)'
    elif mode=="public20":
        replacement='value += _V5_VALUE_MIX * (_fc_public_short(future, me, LAST_TURN) - value)'
    else:
        raise ValueError("unknown forecast mode")
    source=source.replace(native,replacement)
    compile(source,"<forecast-leaf>","exec")
    return source


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--weight",type=float,default=.25,help="fraction inside the existing v5 economic component")
    p.add_argument("--short-share",type=float,default=1.,help="20-step share of forecast, remainder uses the 50-step head")
    p.add_argument("--mode",choices=("supplement","public20"),default="supplement",help="supplement native long-range value or replace its economic component with standalone 20-step public flow")
    a=p.parse_args()
    if a.mode=="public20" and (a.weight!=.25 or a.short_share!=1.):
        p.error("--weight and --short-share only apply to supplement mode")
    if a.source.resolve()==a.output.resolve():p.error("give a separate output filename")
    a.output.write_text(build(a.source.read_text(),a.weight,a.short_share,a.mode))
