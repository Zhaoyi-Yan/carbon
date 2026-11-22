"""Optional extension of tree sensing: propagate feasible integer ranges for partially resolved sums."""
import argparse
import inspect
from pathlib import Path

from sensing import generate


def _sense_ranges(bel, equations, collectors, land_hits, tax_hits, area_rate, worker_rate, est):
    limits = getattr(bel,'sensed_ranges',{})
    for income, cells, enemies in equations:
        unknown = [q for q in cells if not bel.known[q]]
        if len(unknown)<2 or any(not bel.pure[q] for q in unknown):
            continue
        soil=(income-sum(collectors[q][1]*worker_rate/(1-worker_rate*tax_hits[q]) for q in enemies))/area_rate
        residual=soil-sum(bel.val[q]/(1-area_rate*land_hits[q]) for q in cells if bel.known[q])
        if residual < -.015:
            continue
        # Each contribution is monotonic in its (unknown integer) initial carbon.
        spans={}
        for q in unknown:
            lo,hi=limits.get(CLS[q],(0,50))
            denom=1-area_rate*land_hits[q]
            spans[q]=(apply_ops(float(lo),bel.hist[q])/denom,apply_ops(float(hi),bel.hist[q])/denom)
        total_lo=sum(v[0] for v in spans.values())
        total_hi=sum(v[1] for v in spans.values())
        if residual < total_lo-.015 or residual > total_hi+.015:
            continue
        for q,(a,b) in spans.items():
            key=CLS[q]
            oldlo,oldhi=limits.get(key,(0,50))
            denom=1-area_rate*land_hits[q]
            low=(residual-total_hi+b-.015)*denom
            high=(residual-total_lo+a+.015)*denom
            lo,hi=oldlo,oldhi
            if low > a*denom and low>0:
                while lo<hi:
                    mid=(lo+hi)//2
                    if apply_ops(float(mid),bel.hist[q])<low:
                        lo=mid+1
                    else:
                        hi=mid
            newlo=lo
            lo,hi=newlo,oldhi
            if high < b*denom:
                while lo<hi:
                    mid=(lo+hi+1)//2
                    if apply_ops(float(mid),bel.hist[q])>high:
                        hi=mid-1
                    else:
                        lo=mid
            else:
                lo=oldhi
            if newlo!=oldlo or lo!=oldhi:
                limits[key]=(newlo,lo)
    bel.sensed_ranges=limits
    prior=bel.prior()
    for key,(lo,hi) in limits.items():
        bounded=max(lo,min(hi,prior))
        if bounded!=prior:
            for q in CLS_MEMBERS[key]:
                if not bel.known[q] and bel.pure[q]:
                    est[q]=apply_ops(float(bounded),bel.hist[q])
    return est


def confirm(code):
    # A baseline cell labelled known may have drifted after a hidden collision.
    # Before committing an initial mirror class, ask for a second tree if any
    # independent term was inferred rather than visible. A zero soil sum needs no subtraction.
    sites = '''    visible = set(bel.base_cells)
    for cell,_,role in players[obs['player']][2].values():
        visible.update([cell] if role == 'COLLECTOR' else NB[cell])
    sites = {}
    evidence = getattr(bel,'tree_evidence',{})
    bel.tree_evidence = evidence
    equations = []'''
    code=code.replace('    equations = []',sites,1)
    code=code.replace("            equations.append((obs['trees'][tid][1], cells, enemies))",
                      "            equations.append((obs['trees'][tid][1], cells, enemies))\n            sites[id(cells)] = tid",1)
    old='            bel._learn_class(CLS[p],options[0][1])'
    new='''            cls,v0 = CLS[p],options[0][1]
            if soil > .00001 and any(q not in visible for q in cells if q!=p):
                earlier = evidence.get(cls)
                evidence[cls] = (v0,sites[id(cells)])
                if earlier is None or earlier[0]!=v0 or earlier[1]==sites[id(cells)]:
                    continue
            bel._learn_class(cls,v0)'''
    if code.count(old)!=1:raise ValueError('Expected the unique class commit')
    return code.replace(old,new,1)


def extend(code):
    code=code.replace('def _sense_tree_carbon(',inspect.getsource(_sense_ranges)+'\n\ndef _sense_tree_carbon(',1)
    code=code.replace('    if not equations:\n        return est, known\n','')
    old='    return (bel.estimate(),list(bel.known)) if changed else (est,known)'
    new="""    if changed:
        est,known=bel.estimate(),list(bel.known)
    return _sense_ranges(bel,equations,collectors,land_hits,tax_hits,area_rate,worker_rate,est),known"""
    if code.count(old)!=1:
        raise ValueError('Expected sensing v1')
    return code.replace(old,new)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--confirmed',action='store_true',help='Require corroboration for inferred neighboring terms.')
    a=p.parse_args()
    generate(a.base,a.output)
    text=extend(a.output.read_text())
    if a.confirmed:
        text=confirm(text)
    compile(text,str(a.output),'exec')
    a.output.write_text(text)
