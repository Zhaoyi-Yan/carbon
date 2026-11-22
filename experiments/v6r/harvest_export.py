"""Embed the optional collector proposal in an existing single-file strategy.

    python experiments/v6r/harvest_export.py source_strategy.py candidate.py

The exported file has its own inference parameters and needs no external data.
The proposal competes in root search only when three or more candidates fit.
The original action remains first. At most one collector changes in the alternative.
"""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def embed(source, weights=None, preserve_bank=True):
    if "def _hv_proposal(" in source:
        raise ValueError("strategy already contains the collector proposal")
    weights = weights or json.loads((HERE/"harvest-model.json").read_text())
    utilities=(HERE/"harvest.py").read_text()
    gates=json.loads((HERE/"harvest-gate.json").read_text())["gaps"]
    utilities+="\n_HV_PRESERVE_BANK = "+repr(preserve_bank)+"\n"
    utilities+="\n_HV_GAPS = "+repr(tuple(gates))+"\n"
    utilities+="\n_HV_THRESH = "+repr(tuple(weights["thresholds"]))+"\n"
    utilities+="\n_HV_MODEL = tuple(np.asarray(m,dtype=np.float32) for m in "+repr(weights["weights"])+")\n"
    anchor="def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    spot="    for T, cap in _S_GEN:"
    if source.count(anchor)!=1 or source.count(spot)!=1:
        raise ValueError("strategy search layout differs, check the proposal entry point")
    insert="""    if most >= 3 and len(cands) < most:
        extra = _hv_proposal(obs, LAST_TURN, act0, lw0, home)
        if extra is not None:
            key = tuple(sorted(extra.items()))
            if key not in keys:
                keys.add(key)
                cands.append(extra)
"""
    result=source.replace(anchor,utilities+"\n\n"+anchor).replace(spot,insert+spot)
    compile(result,"<collector-proposal>","exec")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--allow-delay-deposit",action="store_true",help="use the initial confidence-gated candidate without the bank guard")
    args=p.parse_args()
    if args.source.resolve()==args.output.resolve():
        p.error("provide a separate candidate file")
    args.output.write_text(embed(args.source.read_text(),preserve_bank=not args.allow_delay_deposit))
