"""Verify actual receipt labels with official cash mutations, test the completely excluded replay, and time inference."""
import argparse,base64,gzip,hashlib,io,json,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import receipts as impl
from receipt_data import ROOT,TARGET_LAYOUT,TARGET_MATCH,layout,targets
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'experiments/v6r'))
from benchmark import configure_engine
from trace_economy import CASH_RULES,rule_lines


def real_target(path):
    data=path.read_bytes()
    if data[:2]==b'\x1f\x8b':data=gzip.decompress(data)
    if data.lstrip().startswith(b'{'):
        obj=json.loads(data);return obj.get('environment',obj)
    return json.JSONDecoder().raw_decode(data.decode().split('window.kaggle =',1)[1].lstrip())[0]['environment']


def validate(g,steps):
    configure_engine()
    from zerosum_env.envs.carbon import helpers,idgen
    from zerosum_env import utils
    cfg=utils.structify(g['configuration']);records,_=targets(g)
    lines=rule_lines(helpers.Board.next,CASH_RULES);code=helpers.Board.next.__code__
    errors=[];cash_errors=[]
    def record(pl,k,val):
        frame=sys._getframe(1)
        if k=='_cash' and frame.f_code is code and lines.get(frame.f_lineno) in ('bank','base_collision'):
            booked[pl.id]+=val-pl.cash
        return object.__setattr__(pl,k,val)
    with patch.object(helpers.Player,'__setattr__',record):
        for t in steps:
            f,nxt=g['steps'][t:t+2]
            public=f[0]['observation'];full={**public,'carbon':public['full_carbon']}
            actions=[z['action'] or {} for z in nxt];booked=np.zeros(2)
            counters={f'player-{s}-{role}-':100000 for s in (0,1) for role in ('tree','worker','recrtCenter')}
            with patch.dict(idgen._id_dict,counters,clear=True):
                actual=helpers.Board(full,cfg,actions).next().observation
            errors.append(float(abs(booked-records[t]).max()))
            # Elimination cleanup is external to Board.next; only compare non-eliminated sides here.
            truth=nxt[0]['observation']['players']
            cash_errors.extend(abs(actual['players'][s][0]-truth[s][0]) for s in (0,1) if truth[s][0]!=0 or truth[s][2])
    return len(errors),max(errors),max(cash_errors,default=0.)


def predict_env():
    root=(ROOT/'yan-v5.py').read_text()
    code=impl.build(root)
    start=code.index('_V7_RECEIPT_DIST=');end=code.index('def _act(obs,',start)
    env=dict(np=np,base64=base64,io=io);exec(compile(code[start:end],'<standalone-receipts>','exec'),env)
    return env['_v7_receipts']


def metrics(y,p):
    error=p-y
    return dict(mae=np.mean(abs(error),axis=(0,1)).round(4).tolist(),
                rmse=np.sqrt(np.mean(error**2,axis=(0,1))).round(4).tolist())


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('target',type=Path);a.add_argument('cache',type=Path)
    a.add_argument('validation',type=Path);a.add_argument('output',type=Path);args=a.parse_args()
    g=real_target(args.target);fp=layout(g);assert fp==TARGET_LAYOUT
    report=json.loads(args.validation.read_text());assert fp not in report['validation_layouts']
    index=json.loads((args.cache/'index.json').read_text())['games']
    assert all(r['layout']!=fp and TARGET_MATCH not in r['file'] for r in index)
    fn=predict_env();frames=g['steps'];last=g['configuration']['episodeSteps']-2
    increments,_=targets(g);cumul=np.vstack((np.zeros(2),increments.cumsum(0)))
    rows=[];predictions=[];truth=[];naive=[];elapsed=[];timed_total=0
    geo=lambda a,b:min(abs(a//15-b//15),15-abs(a//15-b//15))+min(abs(a%15-b%15),15-abs(a%15-b%15))
    for t,f in enumerate(frames):
        players=f[0]['observation']['players'];end=len(frames)-1
        true=np.stack([cumul[min(end,t+h)]-cumul[t] for h in (5,10)],axis=1)
        start=time.perf_counter_ns();p=fn(players,t,last)
        elapsed.append((time.perf_counter_ns()-start)*1e-6)
        base=[]
        for pl in players:
            rr=[]
            for horizon in (5,10):
                h=min(horizon,max(0,last+1-t))
                total=sum(v*max(0.,1-min((geo(q,b) for b in pl[1].values()),default=16)/(h+1))
                          for q,v,k in pl[2].values() if k=='COLLECTOR') if h else 0.
                rr.append(total)
            base.append(rr)
        assert p.shape==(2,2) and np.all(p>=0) and np.all(p[:,1]>=p[:,0]-1e-4)
        if t in (10,50,100,150,175,180,185,190,195,200,210,250,285,290,295,297,298,299):
            rows.append(dict(step=t,actual=true.round(2).tolist(),forecast=p.round(2).tolist(),distance_discount=np.round(base,2).tolist()))
        predictions.append(p);truth.append(true);naive.append(base)
    at=190
    _,logs=targets(g)
    starting={uid:(s,pos,load) for s,pl in enumerate(frames[at][0]['observation']['players']) for uid,(pos,load,k) in pl[2].items()}
    events=[]
    for t in range(at,min(at+10,len(logs))):
        for uid,(bank,money) in logs[t].items():
            s,pos,load=starting.get(uid,(None,None,0.))
            events.append(dict(credited_state=t+1,worker=uid,base_owner=bank,source_side=s,
                               position_at_190=pos,cargo_at_190=round(load,3),paid=round(money,3)))
    compare=[]
    for r in [v for v in index if v['layout'] in report['validation_layouts']][::max(1,len(report['validation_layouts'])//10)][:10]:
        with gzip.open(Path(r['source'])/'replays'/r['file']) as f:other=json.load(f)
        compare.append(validate(other,range(0,len(other['steps'])-1,13)))
    compare.insert(0,validate(g,range(len(g['steps'])-1)))
    assert max(v[1] for v in compare)<.001 and max(v[2] for v in compare)<.02
    y=np.array(truth);p=np.array(predictions);base=np.array(naive)
    out={'target':TARGET_MATCH,'layout':fp,'config_seed':g['configuration']['randomSeed'],'excluded_entire_initial_layout':True,
         'target_metrics':{'fitted':metrics(y,p),'distance_discount':metrics(y,base)},
         'target_rows':rows,'deposits_from_state_190':events,'median_p95_p99_ms':np.percentile(elapsed,[50,95,99]).round(4).tolist(),
         'referee_validation':{'original_frames':sum(k[0] for k in compare),'maps':len(compare),'max_receipt_error':max(k[1] for k in compare),'max_cash_error':max(k[2] for k in compare)}}
    with args.output.open('x') as f:json.dump(out,f,indent=2)
    print(out['target_metrics'],next(v for v in rows if v['step']==190),out['median_p95_p99_ms'],out['referee_validation'])

if __name__=='__main__':main()
