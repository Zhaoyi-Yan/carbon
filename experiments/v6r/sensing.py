"""Generate a standalone strategy that learns hidden carbon from public tree receipts."""
import argparse
import inspect
from pathlib import Path


def _sense_tree_carbon(bel, obs, cfg, previous, est, known):
    if not previous or not obs.get('trees') or all(known):
        return est, known
    area_rate = cfg.get('cellAbsorptionRate', .0375)
    worker_rate = cfg.get('collectorAbsorptionRate', .10)
    if area_rate <= 0 or worker_rate*9 >= 1:
        return est, known
    players = obs['players']
    mature = [(tid,pos,j) for j,pl in enumerate(players) for tid,(pos,age) in pl[3].items() if age>1]
    if not mature:
        return est, known
    tree_cells = {pos for pl in players for pos,_ in pl[3].values()}
    collectors = {}
    stationary = set()
    for j,pl in enumerate(players):
        for wid,(pos,load,role) in pl[2].items():
            if role != 'COLLECTOR':
                continue
            collectors[pos] = (j,load)
            old = previous[j][2].get(wid)
            if old is None or old[0] == pos:
                stationary.add(pos)
    land_hits = {}
    tax_hits = {}
    equations = []
    for tid,pos,side in mature:
        cells = [p for p in NB[pos] if p not in stationary and p not in tree_cells]
        enemies = [p for p in [pos]+NB8[pos] if p in collectors and collectors[p][0] != side]
        for p in enemies:
            tax_hits[p] = tax_hits.get(p,0) + 1
        for p in cells:
            land_hits[p] = land_hits.get(p,0) + 1
        if any(not known[p] for p in cells) and tid in obs['trees']:
            equations.append((obs['trees'][tid][1], cells, enemies))
    if not equations:
        return est, known

    changed = False
    # Each listed enemy's post-tax cargo equals pre-tax cargo * (1 - hits * tax_rate).
    # The land equation uses the same pre/post conversion for cells shared by trees.
    for income, cells, enemies in equations:
        soil = (income - sum(collectors[p][1] * worker_rate/(1-worker_rate*tax_hits[p]) for p in enemies))/area_rate
        if soil < -.01 or soil > len(cells)*cfg.get('maxCellCarbon',100)+.01:
            continue
        unknown = [p for p in cells if not bel.known[p]]
        if not unknown:
            continue
        if abs(soil) <= .00001:
            exact = [(p,0.) for p in unknown]
        elif len(unknown) == 1:
            p = unknown[0]
            rest = sum(bel.val[q]/(1-area_rate*land_hits[q]) for q in cells if q != p)
            value = (soil-rest)*(1-area_rate*land_hits[p])
            if value < -.012 or value > cfg.get('maxCellCarbon',100)+.012:
                continue
            exact = [(p,max(0.,value))]
        else:
            continue
        for p,value in exact:
            if bel.known[p] or not bel.pure[p]:
                continue
            # Infer a mirror class only where the integer initial carbon is identifiable.
            # Multiple possible preimages are common after saturation or many harvests.
            ops = bel.hist[p]
            left, right = 0, 50
            while left < right:
                mid = (left+right)//2
                if apply_ops(float(mid),ops) < value:
                    left = mid+1
                else:
                    right = mid
            options = [(abs(apply_ops(float(v),ops)-value),v)
                       for v in range(max(0,left-2),min(50,left+2)+1)]
            options.sort()
            if options[0][0] > .012 or options[1][0] <= .025:
                continue
            bel._learn_class(CLS[p],options[0][1])
            changed = True
    return (bel.estimate(),list(bel.known)) if changed else (est,known)


def generate(source, destination):
    anchor = "    est, known = st['tr'].step(players, step, obs['carbon'])"
    code = source.read_text()
    if code.count(anchor)!=1 or code.count('def _act(obs, configuration=None, t0=None):')!=1:
        raise ValueError('Expected the yan-v5 belief update and action entry.')
    code = code.replace('def _act(obs, configuration=None, t0=None):',
                        inspect.getsource(_sense_tree_carbon)+'\n\ndef _act(obs, configuration=None, t0=None):')
    code = code.replace(anchor,anchor+"\n    est, known = _sense_tree_carbon(st['tr'].bel, obs, configuration or {}, st['prev'], est, known)")
    compile(code,str(destination),'exec')
    destination.write_text(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    generate(args.base,args.output)
