"""Summarize paired payoffs and source coverage using the source game as the unit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

def signature(action):
    return tuple(sorted(action.items()))


def analyze(games):
    coverage = defaultdict(Counter)
    shadows = defaultdict(Counter)
    site = defaultdict(list)
    changed = Counter()
    for game in games:
        effects = defaultdict(list)
        for root in game['roots']:
            trace = root['trace']
            admitted = {signature(a):i for i,a in enumerate(trace.get('admitted',[]))}
            scored = {int(i) for i in trace.get('scored_depth',{})}
            scores = {int(i):value for i,value in trace.get('final_scores',{}).items()}
            executed = signature(dict(trace['stages'])['executed'])
            for field, table in (('proposed',coverage), ('probes',shadows)):
                # An identical move can have multiple origins. Only an actual generator
                # receives admission credit; probe counts describe action matches.
                for source,key in {(name,signature(a)) for name,a in trace.get(field,[])}:
                    stats = table[source]
                    stats['actions'] += 1
                    i = admitted.get(key)
                    if i is None:
                        stats['outside_pool'] += 1
                    else:
                        stats['in_pool'] += 1
                        stats['evaluated'] += i in scored
                        stats['selected'] += i==trace.get('selected')
                        stats['executed'] += key==executed
            previous = None
            for name,action in trace['stages']:
                if previous is not None and action != previous:
                    changed[name]+=1
                previous=action

            results = root['results']
            final = next(row for row in results if 'executed' in row['labels'])
            search = next(row for row in results if 'search' in row['labels'])
            effects['executed_minus_search'].append(final['margin']-search['margin'])
            omitted = next((row for row in results if 'outside-pool' in row['labels']),None)
            if omitted:
                effects['outside_minus_search'].append(omitted['margin']-search['margin'])
            finalists = [(scores[admitted[key]],row['margin']) for row in results
                         if (key:=signature(row['action'])) in admitted and admitted[key] in scores]
            if len(finalists)>=2:
                # This pair is tied to its own root and budget. Equal model scores
                # are uninformative about ranking and excluded from the comparison.
                finalists.sort(reverse=True)
                if finalists[0][0]!=finalists[1][0]:
                    effects['ranked_first_minus_second'].append(finalists[0][1]-finalists[1][1])
        for name,values in effects.items():
            site[name].append(dict(seed=game['seed'],opponent=game['opponent'],
                                   roots=len(values),mean=round(mean(values),3),
                                   better=sum(v>0 for v in values),equal=sum(v==0 for v in values)))
    effects = {}
    for name, outcomes in site.items():
        effects[name] = dict(games=len(outcomes),roots=sum(x['roots'] for x in outcomes),
                             average=round(mean(x['mean'] for x in outcomes),2),by_game=outcomes)
    return dict(coverage=coverage,shadow_matches=shadows,postprocessor_changes=changed,effects=effects)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    with (args.run/'games.jsonl').open() as f:
        data=analyze(list(map(json.loads,f)))
    with (args.run/'diagnosis.json').open('x') as f:
        json.dump(data,f,indent=2)
        f.write('\n')
    print(json.dumps(data))


if __name__=='__main__':
    main()
