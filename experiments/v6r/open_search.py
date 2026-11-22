"""Compare canopy's feasible root joint proposal, fixed-budget search, and played move on 12 held-out maps."""
import argparse
from collections import Counter,defaultdict
import gzip,hashlib,inspect,json
from pathlib import Path
import numpy as np
from localhead_data import load
from localhead import ROOT


def fixed_search(model):
    src=inspect.getsource(model._search)
    start=src.index('    if _S_COST[0] > 0.0:')
    stop=src.index('    stages =',start)
    src=src[:start]+'    most, D2, M = 14, min(D2,5), 1\n'+src[stop:]
    begin=src.index('        need =',src.index('    def late('))
    end=src.index('    def over(',begin)
    src=src[:begin]+'        return False\n\n'+src[end:]
    env=vars(model)
    exec(compile(src,'<fixed-canopy-search>','exec'),env)
    model._S_MMAX=1
    return model._search


def decision_steps(first_tree,birth):
    if not first_tree or first_tree>34:
        return (5,13,23,33)
    before=first_tree-1
    earlier=max(1,min(before-1,(max(1,birth or 1)+before)//2))
    return (earlier,before,min(34,first_tree+3),min(34,first_tree+9))


def replay_game(m,search,name,meta,data):
    path=data/'replays'/(name+'.json.gz')
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=meta['file_sha256']:raise ValueError('Wrong source '+name)
    r=json.loads(gzip.decompress(blob));frames=r['steps'];side=meta['seat'];tree=meta['first_tree']
    moments=decision_steps(tree,meta['first_planter']);assert len(set(moments))==4
    creator=set()
    if tree is not None:
        fo=frames[tree][0]['observation']
        creator={fo['trees'][tid][0] for tid in fo['players'][side][3]}
    m._ST.clear();m._S_REC=[];m.LAST_TURN=r['configuration']['episodeSteps']-2
    records=[]
    for t,frame in enumerate(frames[:max(moments)+1]):
        shared=frame[0]['observation'];own=shared['players'][side]
        # Do not hand the agent the replay-only oracle, even if seat zero is running.
        obs=dict(step=t,player=side,players=shared['players'],trees=shared['trees'],
                 carbon=frame[side]['observation']['carbon'],remainingOverageTime=frame[side]['observation']['remainingOverageTime'])
        teacher=frames[t+1][side]['action'] or {}
        if t in moments:
            trace={}
            def spy(obs,config,est,act,lw,lb,prev,t0,home=()):
                m._S_REC.clear()
                selected=search(obs,config,est,act,lw,lb,prev,t0,home)
                trace.update(root=act.copy(),search=selected.copy(),worker=lw,base=lb,home=list(home))
                if m._S_REC:
                    row=m._S_REC[-1]
                    trace.update(candidates=len(row[2]),depth=max(row[3]),pick=int(row[-1]))
                else:trace.update(candidates=1,depth=0,pick=0)
                return selected
            m._search=spy
        else:
            # Advance the full root, style posterior, belief and stochastic policy RNG on public history;
            # search random streams are local to each turn and discarded with their action.
            m._search=lambda o,c,e,act,*args:act
        played=m.agent(obs,r['configuration'])
        m._ST['tr'].set_actions(shared['players'],teacher)
        if t not in moments:continue
        root,searched=trace['root'],trace['search']
        eligible=own[0]>=min(r['configuration'].get('recCollectorCost',30),r['configuration'].get('recPlanterCost',30)) and len(own[2])<r['configuration'].get('workerLimit',10)
        actors=[]
        for uid,pos,role in [(uid,w[0],w[2].lower()) for uid,w in own[2].items()]+[(uid,pos,'base') for uid,pos in own[1].items()]:
            isbase=role=='base';idx=m.BIDX if isbase else m.WIDX
            old,new,end,a=(idx[moves.get(uid) or 'NONE'] for moves in (root,searched,played,teacher))
            logits=trace['base' if isbase else 'worker'][pos]
            actors.append(dict(uid=uid,kind=role,root=old,search=new,played=end,teacher=a,
                               sampling_p=round(float(m._probs(logits,m._ROOT_TEMP)[old]),7),
                               softmax_p=round(float(m._probs(logits,1.)[old]),7),
                               eligible=eligible if isbase else True,created_first_tree=uid in creator))
        records.append(dict(game=name,seed=meta['seed'],seat=side,teacher=meta['name'],step=t,
                            first_tree=tree,phase='pre' if tree is None or t<tree else 'post',
                            actual_first_tree_decision=t==tree-1 if tree is not None else False,
                            root_joint=root,searched_joint=searched,played_joint=played,teacher_joint=teacher,
                            search_candidates=trace['candidates'],depth=trace['depth'],selected=trace['pick'],units=actors))
    return records


def summarize(recs):
    totals=defaultdict(Counter)
    for r in recs:
        for u in r['units']:
            groups=['all',u['kind'],r['phase']+'_'+u['kind']]
            if u['eligible']:groups+=['eligible',r['phase']+'_eligible']
            if u['eligible'] and u['kind']=='base':groups+=['eligible_bases']
            if not u['eligible']:groups+=['unaffordable_or_full_bases']
            if u['created_first_tree'] and r['phase']=='pre':groups+=['first_tree_planter_pre']
            if u['created_first_tree'] and r['actual_first_tree_decision']:groups+=['actual_first_tree_decision']
            if u['sampling_p']>=.90 and u['eligible']:
                groups+=['high90_eligible','high90_'+u['kind'],r['phase']+'_high90_'+u['kind']]
            if u['softmax_p']>=.8 and u['eligible']:groups+=['high80_unscaled_eligible']
            root,search,played,label=(u[k] for k in ('root','search','played','teacher'))
            for key in groups:
                c=totals[key];c['n']+=1;c['root_match']+=root==label;c['search_match']+=search==label
                c['played_match']+=played==label;c['changed']+=root!=search
                c['lost_match']+=root==label and search!=label
                c['gained_match']+=root!=label and search==label
                c['both_wrong_change']+=root!=label and search!=label and root!=search
    return {k:dict(v) for k,v in totals.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('canopy',type=Path);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();m=load(a.canopy);search=fixed_search(m)
    split=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())['validation']
    manifest=[json.loads(z) for z in (ROOT/'analysis/v6r-artemis-games.jsonl').read_text().splitlines()]
    records=[]
    for name in split:
        mid,game=name.rsplit('-',1)
        meta=next(g for g in manifest if g['match']==mid and g['game']==int(game) and g['name'] in ('Artemis','Apollo'))
        records.extend(replay_game(m,search,name,meta,a.data))
        print(name,len(records),flush=True)
    result={'model_sha256':hashlib.sha256(a.canopy.read_bytes()).hexdigest(),'cpu_threads':m._torch.get_num_threads(),
            'width_cap':14,'requested_depth':5,'rollouts':1,'root_sampling_temperature':m._ROOT_TEMP,
            'selection':'Two pre- and two post-first-tree states when first tree by step 34; otherwise steps 5,13,23,33.',
            'confidence':'Probability of the realized feasible root proposal at canopy action temperature; unit aggregate also gives unscaled 0.8 threshold.',
            'teacher_forcing':'Advance normal root and recorded observations every turn. Feed recorded historical own action back to the belief tracker. Run actual search only for the prespecified states; cap width/depth/rollouts and disable wall cutoff.',
            'summary':summarize(records),'samples':records}
    with a.output.open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
