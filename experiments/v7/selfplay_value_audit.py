"""Verify archive map hashes, model byte equality, and compare against carrying today's cash forward."""
import selfplay_value_data as d
import argparse,base64,hashlib,io,json,re
from pathlib import Path
import numpy as np
from arena import engine_fingerprint
from carrier_action_fit import canonical
from selfplay_value_fit import MODEL,read


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--cache',required=True,type=Path);cli.add_argument('--candidate',required=True,type=Path);a=cli.parse_args();d.configure_engine()
    from zerosum_env.envs.carbon import carbon
    engine=engine_fingerprint(carbon);verified=set();canonical_maps=set();transitions=0
    for n in range(1,9):
        for line in (d.OUT/f'{n:02d}-qualifying.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['status']!='completed':continue
            st,env=d.new_game(row['seed'],300);b=st[0].observation
            assert row['engine_sha256']==engine
            for field,bits in [('map_sha256',[b.full_carbon,[list(p[1].values()) for p in b.players]]),
                              ('layout_sha256',[15,b.full_carbon,[sorted(p[1].values()) for p in b.players]])]:
                assert row[field]==hashlib.sha256(json.dumps(bits,separators=(',',':')).encode()).hexdigest()
            verified.add(row['layout_sha256']);canonical_maps.add(canonical(b));transitions+=row['turns']
    assert len(verified)==len(canonical_maps)==64
    rank=json.loads((d.OUT/'selfplay-value-ranking.json').read_text());outside=set()
    for case in rank['cases']:
        state,_=d.new_game(case['seed'],300);outside.add(canonical(state[0].observation))
    assert len(outside)==12 and not outside&canonical_maps
    src=d.SOURCE.read_text();dst=a.candidate.read_text();regex=r'^_VALUE = "([^"]+)"$'
    assert re.sub(regex,'VALUE',src,flags=re.M)==re.sub(regex,'VALUE',dst,flags=re.M)
    def head(code):
        return dict(np.load(io.BytesIO(base64.b64decode(re.search(regex,code,re.M)[1])),allow_pickle=False))
    o,n=head(src),head(dst);changes={k:int(np.count_nonzero(o[k]!=n[k])) for k in o};assert all(v==0 for k,v in changes.items() if not k.startswith('v3.'))
    assert all(np.array_equal(o[k],n[k]) for k in ('v1.weight','v1.bias','v2.weight','v2.bias'))
    manifest=json.loads((a.cache/'index.json').read_text());valid=read(a.cache,[r for r in manifest['maps'] if r['batch']>=7])
    mod=dict(np.load(MODEL));new=valid['old']+valid['h']@mod['dw']+mod['db'];carry=valid['sc'][:,0]
    subsets={}
    for title,at in [('all',np.full(len(new),True)),('early_0_99',valid['t']<100),('mid_100_199',(valid['t']>=100)&(valid['t']<200)),('late_200_plus',valid['t']>=200)]:
        subsets[title]={'n':int(at.sum()),**{name:round(float(np.mean(np.abs(pred[at]-valid['y'][at]))*3000),3) for name,pred in [('current_cash',carry),('lifeline',valid['old']),('selfplay',new)]}}
    out=dict(engine_sha=engine,verified_layouts=len(verified),rotationally_distinct_qualification_and_branch_maps=[len(canonical_maps),len(outside)],verified_taped_referee_transitions=transitions,source_hash=d.sha(d.SOURCE),
             candidate_hash=d.sha(a.candidate),only_changed_line='_VALUE',parameters_changed=changes,current_cash_comparison=subsets)
    (d.OUT/'selfplay-value-audit.json').write_text(json.dumps(out,indent=1)+'\n');print(out,flush=True)

if __name__=='__main__':main()
