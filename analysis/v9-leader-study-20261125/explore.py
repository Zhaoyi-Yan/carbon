"""Descriptive replay audit; no strategies are executed and no hidden inputs are used for training."""
import argparse, collections, gzip, hashlib, json, math, statistics
from pathlib import Path

ARTEMIS='1f5b63b85d43a90ab4d4e75c294d1d62'
V8='52af7735a7677815efb48cdf94269e29'
V9='0f9896ed3dc3cd7bfad29fe2dd38d6fb'
TIMES=(10,25,30,50,75,100,150,200,225,250,280,299)
SOURCE='ddc3e6f6922474ad5894a8066be5b906bf35af55'
CURRENT_METADATA_SHA256={
    'index.json':'a328b208a39a325907fd249aceb97895fefc89322eae090309bc991ee1471096',
    'leaderboard.json':'6c9854b37e6272d2ad9c9f4ec33560a4b71b9d2a6341f325be6474b1e25ec055',
}

def sha(b):return hashlib.sha256(b).hexdigest()
def mean(xs):return statistics.mean(xs) if xs else None

def validate_current_snapshot(path):
    """This CLI only audits the frozen current snapshot; imports do not validate."""
    for name,expected in CURRENT_METADATA_SHA256.items():
        observed=sha((Path(path)/name).read_bytes())
        if observed!=expected:
            raise ValueError(f'{name}: expected {SOURCE} SHA256 {expected}, got {observed}; use history.py for the historical snapshot')

def measure(meta, replay):
    order=meta['order'];steps=replay['steps'];cfg=replay['configuration'];size=cfg['size']
    start=steps[0][0]['observation'];bases=[set(p[1].values()) for p in start['players']]
    def dist(a,b):
        r,c=divmod(a,size);rr,cc=divmod(b,size)
        return min(abs(r-rr),size-abs(r-rr))+min(abs(c-cc),size-abs(c-cc))
    home=[[min(dist(pos,b) for b in base) for pos in range(size*size)] for base in bases]
    layout=sha(json.dumps([size,start['full_carbon'],[sorted(p[1].values()) for p in start['players']]],separators=(',',':')).encode())
    sums=[collections.Counter(),collections.Counter()];samples=[{},{}];cumulative=[{},{}];first=[{},{}]
    for t in range(1,len(steps)):
        old=steps[t-1][0]['observation']['players'];obs=steps[t][0]['observation'];new=obs['players']
        all_old_trees=set(old[0][3])|set(old[1][3]);alive=sum(tree[1]<cfg['treeLifespan'] for p in old for tree in p[3].values())
        market=cfg['plantCost']+cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**alive
        for seat,(p,q) in enumerate(zip(old,new)):
            s=sums[seat];actions=steps[t][seat]['action'] or {};opp=old[1-seat]
            planted=set(q[3])-all_old_trees;seized=set(q[3])&set(opp[3]);nold=len(p[2]);cash=p[0];recruit_cost=0
            for base in p[1]:
                ac=actions.get(base)
                if ac in ('RECCOLLECTOR','RECPLANTER'):
                    role='COLLECTOR' if ac=='RECCOLLECTOR' else 'PLANTER'
                    cost=cfg['recCollectorCost' if role=='COLLECTOR' else 'recPlanterCost']
                    assert cfg['recCollectorIncreaseCost']==cfg['recPlanterIncreaseCost']==0
                    if cash>=cost and nold<cfg['workerLimit']:
                        s['recruited_'+role]+=1;recruit_cost+=cost;cash-=cost;nold+=1
                        first[seat].setdefault('recruit',t)
            plant_price=cfg['plantCost'] if len(p[3])<=cfg['playerTreeProtectiveNumber'] else market
            tree_income=sum(obs['trees'][tid][1] for tid in q[3])
            s['tree_income']+=tree_income;s['plants']+=len(planted);s['seizes']+=len(seized)
            s['recruit_cost']+=recruit_cost;s['plant_cost']+=plant_price*len(planted);s['seize_cost']+=cfg['seizeCost']*len(seized)
            s['non_tree_cash_inflow']+=q[0]-p[0]+recruit_cost+plant_price*len(planted)+cfg['seizeCost']*len(seized)-tree_income
            s['worker_turns']+=len(q[2]);s['collector_turns']+=sum(w[2]=='COLLECTOR' for w in q[2].values());s['tree_turns']+=len(q[3])
            s['cargo_turns']+=sum(w[1] for w in q[2].values())
            disappeared=set(p[2])-set(q[2]);s['disappeared_workers']+=len(disappeared);s['disappeared_worker_cargo']+=sum(p[2][u][1] for u in disappeared)
            pressure=sum(w[2]=='COLLECTOR' and w[1]<1 and home[1-seat][w[0]]<=2 for w in q[2].values())
            s['empty_collectors_near_enemy_base']+=pressure
            if t<=75:s['early_empty_collectors_near_enemy_base']+=pressure
            for u,w in p[2].items():
                if u in q[2] and w[2]=='COLLECTOR' and w[1]>0 and q[2][u][0] in bases[seat]:
                    s['loaded_return_events']+=1;s['pre_return_cargo']+=w[1]
                    first[seat].setdefault('loaded_return',t)
                if w[2]=='COLLECTOR' and w[1]<1:s['empty_collector_turns']+=1
                if actions.get(u) not in ('UP','DOWN','LEFT','RIGHT'):
                    s['stay_'+w[2]]+=1
            if planted:first[seat].setdefault('plant',t)
            if seized:first[seat].setdefault('seize',t)
            if len(q[2])>=cfg['workerLimit']:first[seat].setdefault('full_population',t)
            if t in TIMES or t==len(steps)-1:
                samples[seat][str(t)]={'cash':q[0],'collectors':sum(w[2]=='COLLECTOR' for w in q[2].values()),'planters':sum(w[2]=='PLANTER' for w in q[2].values()),'trees':len(q[3]),'cargo':sum(w[1] for w in q[2].values())}
                cumulative[seat][str(t)]={k:round(v,5) for k,v in s.items()}
    sides=[]
    for seat,slot in enumerate(order):
        assert abs(replay['rewards'][seat]-meta['scores'][slot])<.011
        sums[seat]['terminal_cargo']=sum(w[1] for w in new[seat][2].values())
        a0=steps[1][seat]['action'] or {}
        opening=''.join(sorted('C' if a=='RECCOLLECTOR' else 'P' for a in a0.values() if a in ('RECCOLLECTOR','RECPLANTER')))
        sides.append(dict(**meta['strategies'][slot],seat=seat,score=meta['scores'][slot],opening=opening,first=first[seat],samples=samples[seat],cumulative=cumulative[seat],totals={k:round(v,5) for k,v in sums[seat].items()}))
    return {'match':meta['id'],'game':meta['game'],'file':meta['file'],'seed':cfg['randomSeed'],'turns':len(steps)-1,'full_length':len(steps)==cfg['episodeSteps'],'layout_sha256':layout,'seats':sides}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot',type=Path);parser.add_argument('output',type=Path);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    validate_current_snapshot(a.snapshot)
    idx=json.loads((a.snapshot/'index.json').read_text());lb=json.loads((a.snapshot/'leaderboard.json').read_text())['strategies'];top6={r['id'] for r in lb if r['rank']<=6};targets={ARTEMIS,V8,V9};games=[];manifest=[]
    for meta in idx:
        if not any(s['id'] in targets for s in meta['strategies'].values()):continue
        assert all(meta['statuses'][s]=='DONE' for s in ('a','b'))
        data=(a.snapshot/meta['file']).read_bytes();r=json.loads(gzip.decompress(data));g=measure(meta,r);games.append(g);manifest.append(dict(meta,sha256=sha(data),layout_sha256=g['layout_sha256']))
    (a.output/'games.json').write_text(json.dumps(games,ensure_ascii=False,indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps({'snapshot':SOURCE,'index_sha256':sha((a.snapshot/'index.json').read_bytes()),'leaderboard_sha256':sha((a.snapshot/'leaderboard.json').read_bytes()),'games':manifest},ensure_ascii=False,indent=2)+'\n')
    summaries={}
    for target in targets:
        own=[(g, next(s for s in g['seats'] if s['id']==target),next(s for s in g['seats'] if s['id']!=target)) for g in games if any(s['id']==target for s in g['seats'])]
        groups={'all':own,'top6':[x for x in own if x[2]['id'] in top6]}
        for opp in sorted({x[2]['id'] for x in own}):groups[opp]=[x for x in own if x[2]['id']==opp]
        summaries[target]={}
        for key,rows in groups.items():
            if not rows:continue
            d={'n':len(rows),'wins':sum(s['score']>o['score'] for g,s,o in rows),'losses':sum(s['score']<o['score'] for g,s,o in rows),'full_length':sum(g['full_length'] for g,s,o in rows),'mean_margin':mean([s['score']-o['score'] for g,s,o in rows]),'opponent_names':sorted({o['name'] for g,s,o in rows}),'opening':dict(collections.Counter(s['opening'] for g,s,o in rows)),'first':{},'samples':{},'totals':{}}
            for field in ('recruit','plant','loaded_return','full_population'):
                v=[s['first'][field] for g,s,o in rows if field in s['first']];d['first'][field]={'n':len(v),'mean':mean(v),'median':statistics.median(v) if v else None}
            for t in TIMES:
                valid=[x for x in rows if str(t) in x[1]['samples'] and str(t) in x[2]['samples']]
                if not valid:continue
                d['samples'][str(t)]={'n':len(valid)}
                for f in ('cash','collectors','planters','trees','cargo'):
                    d['samples'][str(t)][f]={'own_mean':mean([s['samples'][str(t)][f] for g,s,o in valid]),'opponent_mean':mean([o['samples'][str(t)][f] for g,s,o in valid]),'paired_difference':mean([s['samples'][str(t)][f]-o['samples'][str(t)][f] for g,s,o in valid])}
            for f in sorted(set().union(*(set(s['totals']) for g,s,o in rows))):
                d['totals'][f]={'own_mean':mean([s['totals'].get(f,0) for g,s,o in rows]),'opponent_mean':mean([o['totals'].get(f,0) for g,s,o in rows]),'paired_difference':mean([s['totals'].get(f,0)-o['totals'].get(f,0) for g,s,o in rows])}
            summaries[target][key]=d
    (a.output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    print('Wrote',len(games),'games into',a.output)
    for target,groups in summaries.items():
        print(target,[(k,d['n'],d['wins'],round(d['mean_margin'],1),d['opening']) for k,d in groups.items()])

if __name__=='__main__':main()
