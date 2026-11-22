"""Profile each search budget at replay positions using the acting seat's partial map."""
import argparse
import gzip
import json
from pathlib import Path
import time
from backbone import configure_engine,limit_math_threads,load_strategy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay',type=Path);p.add_argument('output',type=Path)
    p.add_argument('models',nargs='+',type=Path)
    a=p.parse_args();configure_engine();limit_math_threads()
    replay=json.loads(gzip.decompress(a.replay.read_bytes()));frames=replay['steps'];records=[]
    for path in a.models:
        model=load_strategy('probe_'+path.stem,path.resolve())
        side=0;probe={20,48,96,145,190,235,280,295}
        for t,(frame,after) in enumerate(zip(frames,frames[1:])):
            common=frame[0]['observation'];pl=common['players']
            obs=dict(step=t,player=side,players=pl,trees=common.get('trees',{}),carbon=frame[side]['observation']['carbon'])
            start=time.perf_counter()
            model._act(obs,replay['configuration'] if t in probe else None,start)
            if t in probe:
                width,depth,replies,elapsed,_=model._S_STATS['last']
                records.append(dict(model=path.stem,step=t,candidates=width,depth=depth,replies=replies,
                                    seconds=round(elapsed,4)))
            model._ST['tr'].set_actions(pl,after[side]['action'])
    a.output.write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
