"""Report paired effects with equal weight per source map."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import mean, median


def report(roots):
    by_map = []
    for r in roots:
        diffs = [s['delta'] for s in r['streams']]
        assert len(diffs) == 3
        # Two other streams have to agree before selecting the runner-up.
        held = [d if all(x>0 for j,x in enumerate(diffs) if i!=j) else 0
                for i,d in enumerate(diffs)]
        later = [s['alternative']['marks'][str(r['turn']+10)]['margin'] -
                 s['baseline']['marks'][str(r['turn']+10)]['margin'] for s in r['streams']
                 if str(r['turn']+10) in s['baseline']['marks'] and str(r['turn']+10) in s['alternative']['marks']]
        by_map.append(dict(seed=r['seed'],seat=r['seat'],turn=r['turn'],category=r['category'],
                           scores=diffs,mean=round(mean(diffs),2),ten_turn_mean=round(mean(later),2) if later else None,
                           held_stream=held,features_collide=r['features'][0]==r['features'][1],
                           changed_units=len(r['differences']),
                           postprocessed=[a!=b for a,b in zip(r['raw'],r['executed'])]))
    assert len({r['seed'] for r in roots}) == len(roots) == 12
    by_group = {}
    for category in ('cash','cargo','tree','ordinary'):
        maps = [r for r in by_map if r['category']==category]
        assert len(maps)==3
        by_group[category] = dict(maps=3,mean=round(mean(x for r in maps for x in r['scores']),2),
                                  alternative_unanimous=sum(min(r['scores'])>0 for r in maps),
                                  baseline_unanimous=sum(max(r['scores'])<0 for r in maps))
    return dict(by_map=by_map,by_category=by_group,
                median_map_effect=round(median(mean(r['scores']) for r in by_map),2),
                cross_stream_gain=round(mean(x for r in by_map for x in r['held_stream']),2),
                cross_stream_changes=sum(all(x>0 for j,x in enumerate(r['scores']) if i!=j)
                                         for r in by_map for i in range(3)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args=parser.parse_args()
    with gzip.open(args.run/'roots.jsonl.gz','rt') as f:
        data=report(list(map(json.loads,f)))
    (args.run/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
