"""Prospective opponent-move audit of the optional online router on common observations."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import types

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from elite_rival import build
from starvation_replay import read_replay
ROOT=Path(__file__).resolve().parents[2]
ART='226891089ba7f54b198c8f8dc771dfc2'
APOLLO='dd1f117a64c676225f0c6333b1cdd1e7'
WINDOWS=((48,72),(166,190),(246,270))


def candidate_source(code, initial):
    candidate=build(code)
    if not initial:
        return candidate
    start=candidate.index('    # Average opposite rotations')
    end=candidate.index('    for uid,(cell,_,_) in foe[2].items():',start)
    old_network="""    inputs = np.stack([other_view, other_view]).astype(np.float32)
    workers,bases,_ = _net()(inputs, np.array([21,22], dtype=np.int64))
"""
    candidate=candidate[:start]+old_network+candidate[end:]
    return (candidate
            .replace("memory['count'] < 19 or gain < (.008 if previous else .028)",
                     "memory['count'] < 12 or gain < (.025 if previous else .075)")
            .replace("min(.72,.30+2.4*(gain-.008))","min(.78,.30+2.4*(gain-.025))"))


def collect_games(args):
    recent=json.loads((args.recent/'index.json').read_text())
    # Three held-out Artemis maps selected by distinct opponents in frozen manifest order.
    rows=[recent['rows'][j] for j in ((1,3,5,6,7,8) if args.confirm else (0,2,4))]
    for row in rows:
        f=args.recent/row['file'];g=read_replay(f)
        for s,slot in enumerate(row['order']):
            yield row['id'], 'post_training', row['strategies'][slot]['name'], s, g
    if args.confirm:
        return
    hist=json.loads((args.historical/'index.json').read_text())
    selection=json.loads((ROOT/'experiments/v6r/elite-validation.json').read_text())
    old=set(m.split('-')[0] for m in selection['validation'])
    ap=next(r for r in hist if r['id'] in old and any(v['id']==APOLLO for v in r['strategies'].values()))
    s=next(j for j,k in enumerate(ap['order']) if ap['strategies'][k]['id']==APOLLO)
    yield ap['id'],'v6r_epoch_selection','CarbonEdge_Apollo',s,read_replay(args.historical/ap['file'])
    if args.screenshot:
        g=read_replay(args.screenshot)
        for i,name in enumerate(('claude-AlphaCarbon-Master7sE','yan-QQ-v6')):
            yield '4063a34afedb6d4767a9118e1dee9e5d','post_training',name,i,g


def summarize(counter):
    def avg(k,d):
        return round(counter[k]/max(1,counter[d]),5)
    return dict(decisions=counter['turns'],worker_observations=counter['workers'],
                selected_apollo=counter['style21'],selected_artemis=counter['style22'],
                selected_legacy=counter['style0'], changed_argmax=counter['switched'],
                worker_accuracy=[avg(k,'workers') for k in ('legacy_correct','patched_correct')],
                worker_nll=[avg(k,'workers') for k in ('legacy_nll','patched_nll')],
                eligible_base_observations=counter['bases'],
                base_accuracy=[avg(k,'bases') for k in ('legacy_base','patched_base')],
                median_block_warmup=6)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--historical',type=Path,required=True)
    a.add_argument('--initial',action='store_true',help='Reproduce the original one-rotation development test.')
    a.add_argument('--confirm',action='store_true',help='Disjoint six-map confirmation split; skip development and historical games.')
    a.add_argument('--recent',type=Path,required=True);a.add_argument('--screenshot',type=Path)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    src=(ROOT/'yan-v6.py').read_text()
    mod=types.ModuleType('elite_rival_audit_module')
    exec(compile(candidate_source(src,args.initial),'<elite-rival-generated>','exec'),mod.__dict__)
    assert mod._SEARCH_OK
    assert len(mod._POLICY.p['emb'])==23
    tiny_rows=len(mod._V6R_TINY.emb);assert tiny_rows<=21
    snapshot={}
    original=mod._v7_route_rival
    def capture(state,pl,side,t,x):
        snapshot['legacy']=state['opponent']
        start=time.perf_counter()
        original(state,pl,side,t,x)
        snapshot['ms']=(time.perf_counter()-start)*1000
    mod._v7_route_rival=capture
    all_rows=[];grand=Counter();timings=[];counterfactual_checked=False
    for uid,cohort,name,side,g in collect_games(args):
        viewer=1-side;frames=g['steps'];cfg=g['configuration'];mod._ST.clear()
        tr=mod.BeliefTracker(mod._Bel,viewer);prev=None
        count=Counter()
        for t,fr in enumerate(frames[:-1]):
            if t>=WINDOWS[-1][1]:break
            board=fr[0]['observation'];pl=board['players'];obs=dict(players=pl,player=viewer,step=t,trees=board['trees'],carbon=fr[viewer]['observation']['carbon'])
            stock,known=tr.step(pl,t,obs['carbon'])
            stock,known=mod._sense_tree_carbon(tr.bel,obs,cfg,prev,stock,known)
            block=next((a for a,b in WINDOWS if a<=t<b),None)
            if block is not None:
                if t==block: mod._ST.pop('v5_history',None)
                x=mod._quant(mod.encode(pl,t,viewer,stock,known,prev))
                mod._v5_root(obs,stock,x,prev)
                state=mod._ST['v5_history'];memory=state['elite_rival']
                chosen=memory['selected']
                assert state['simulation_ids'][side]==(chosen or memory['legacy_id'])
                assert memory['legacy_id']<tiny_rows
                if t>=block+6:
                    timings.append(snapshot['ms']);count['turns']+=1;count['style'+str(chosen)]+=1
                    act=frames[t+1][side]['action'] or {}
                    for u,(pos,load,k) in pl[side][2].items():
                        action=mod.WIDX.get(act.get(u),0)
                        predicted=[]
                        for label,logits in [('legacy',snapshot['legacy'][0][pos]),('patched',state['opponent'][0][pos])]:
                            loss=-mod._v7_rival_logprobs(logits[None,:])[0,action]
                            pick=int(np.argmax(logits))
                            count[label+'_nll']+=float(loss)
                            count[label+'_correct']+=int(pick==action);predicted.append(pick)
                        count['workers']+=1;count['switched']+=int(predicted[0]!=predicted[1])
                    if len(pl[side][2])<cfg['workerLimit'] and pl[side][0]>=30:
                        for u,pos in pl[side][1].items():
                            action=mod.BACT.index(act[u]) if u in act else 0
                            count['bases']+=1
                            for label,logits in [('legacy',snapshot['legacy'][1][pos]),('patched',state['opponent'][1][pos])]:
                                count[label+'_base']+=int(int(np.argmax(logits))==action)
                    if chosen and not counterfactual_checked:
                        # Exercise the real search wrapper with two opponent rows sharing one teacher reference.
                        views=np.stack((x,x));dummy=[(0,0),(1,0)]
                        mod._v6r_counterfactuals(mod._net(),views,None,side,dummy)
                        counterfactual_checked=True
            tr.set_actions(pl,frames[t+1][viewer]['action'] or {})
            prev=pl
        all_rows.append(dict(match=uid,opponent=name,seat=side,cohort=cohort,seed=cfg['randomSeed'],metrics=summarize(count)))
        grand.update(count)
        print(name,uid[:7],summarize(count),flush=True)
    out=dict(frozen_source_sha=hashlib.sha256(src.encode()).hexdigest(),windows=WINDOWS,
             select_rule='One rotation, 12 effective labels, enter 0.075, keep 0.025.' if args.initial else 'Two rotations. EWMA of worker likelihood gain, 19 effective labels, enter at 0.028 nat, keep at 0.008; previous-style hysteresis 0.025. Chosen on the prior development split before opening confirmation predictions.',
             validation='Opponent actions from next frame only used as scoring labels; online evidence only inferred from surviving positions. Both candidates use viewer fog. Original shadow+v5 root is the paired legacy prediction. Blocks restart all opponent-router memory at beginning. First 6 turns in each 24-turn block are warmup. No paired games run.',
             teacher_embedding_rows=23,tiny_embedding_rows=tiny_rows,student_indices_checked_in_real_search=counterfactual_checked,
             extra_ms_median=round(float(np.median(timings)),3),extra_ms_p95=round(float(np.percentile(timings,95)),3),
             pooled=summarize(grand),selected_map_partition='confirmation' if args.confirm else 'development',rows=all_rows)
    args.output.write_text(json.dumps(out,indent=1,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
