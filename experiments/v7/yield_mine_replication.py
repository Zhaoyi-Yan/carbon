"""Freeze four outcome-independent development maps and check them against evaluation and replay pools."""
import gzip
import hashlib
import json
from pathlib import Path
from opening_probe import ROOT,HERE,OUT,map_id,sha
from benchmark import new_game
from branch_replay import write
from receipt_manifest import semantic_layout


def freeze():
    fixed=json.loads((OUT/'yield-mine-probe.json').read_text())
    if fixed['finished_maps']!=4 or sha(HERE/'yield_mine.py')!=fixed['policy_sha256']:
        raise ValueError('Exploratory policy has not been frozen')
    namespace='carbon-v7/yield-mine/replication-1/'
    seeds=[int.from_bytes(hashlib.sha256((namespace+str(i)).encode()).digest()[:4],'big')%10**9 for i in range(4)]
    manifest=json.loads((HERE/'panels.json').read_text())
    official={s for key in ('qualifying','confirmation') for panel in manifest[key] for s in panel}
    prior=set(manifest['development'])
    receipt=[json.loads(l) for l in (OUT/'receipt-games.jsonl').read_text().splitlines()]
    public={r['canonical'] for r in receipt};source_seeds={r['seed'] for r in receipt}
    def fake(oh):return {'steps':[[{'observation':oh}]]}
    src=json.loads((OUT/'branch-original.json').read_text())
    public.add(semantic_layout(fake(src['initial']['seats'][0]['observation'])));source_seeds.add(src['configuration']['randomSeed'])
    for game in json.loads(gzip.decompress((OUT/'branch-breakout-replays.json.gz').read_bytes())).values():
        public.add(semantic_layout(fake(game['initial']['seats'][0]['observation'])));source_seeds.add(game['configuration']['randomSeed'])
    for p in sorted((ROOT.parent/'v7-data/distill-public').glob('*.json.gz')):
        with gzip.open(p) as f:game=json.load(f)
        public.add(semantic_layout(game));source_seeds.add(game['configuration']['randomSeed'])
    registry={}
    for seed in sorted(official|prior|set(seeds)):
        state,_=new_game(seed,300)
        registry[seed]=(map_id(state),semantic_layout(fake(state[0].observation)))
    blocked={registry[s][1] for s in official|prior}|public
    assert len(seeds)==len(set(seeds))==4 and not (set(seeds)&(official|prior|source_seeds))
    assert len({registry[s][1] for s in seeds})==4 and not any(registry[s][1] in blocked for s in seeds)
    result=dict(seeds=seeds,namespace=namespace,rule='First four SHA-256 outputs, unsigned first 32 bits modulo 1000000000; no outcome viewed or replacement drawn.',
                policy_sha256=fixed['policy_sha256'],frozen_v6_sha256=fixed['frozen_sha256'],
                layouts={str(s):dict(exact=registry[s][0],rotation_reflection_seat_invariant=registry[s][1]) for s in seeds},
                exclusion_counts=dict(qualification_confirmation_seeds=len(official),prior_development_seeds=len(prior),public_unique_semantic_layouts=len(public)),
                overlap=dict(seed=0,semantic_layout=0))
    path=OUT/'yield-mine-replication-maps.json'
    if path.exists():
        if json.loads(path.read_text())!=result:raise ValueError('Frozen replication manifest differs')
    else:write(path,result)
    return result
