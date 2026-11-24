"""Validate hidden-input isolation, collector recovery and measured rollout throughput."""
from pathlib import Path
import sys
import json
import time

from pilot import Actor, Collector, Encoder, Game, STYLE, load_ft3, np, torch


def main(source,out):
    teacher=load_ft3(source)
    style=teacher._POLICY.sid(STYLE)
    cpu=Actor(teacher._POLICY,style)
    obs_game=Game(332461)
    ref=Encoder(teacher,0).encode(obs_game.observations()[0])
    board=obs_game.state[0].observation
    board.full_carbon=[j+1. for j in board.full_carbon]
    changed=Encoder(teacher,0).encode(obs_game.observations()[0])
    assert np.array_equal(ref,changed)

    data={}
    for dev in ('cpu','mps') if torch.backends.mps.is_available() else ('cpu',):
        net=cpu.to(dev)
        collector=Collector(teacher,16,562781,dev,coordinated=True)
        collector.collect(net,net,.4,4)
        snapshot=collector.snapshot()
        # The second collector reconstructs independent ID and belief states.
        other=Collector(teacher,16,0,dev,coordinated=True)
        import copy
        other.recover(copy.deepcopy(snapshot))
        left=collector.collect(net,net,.4,4)
        right=other.collect(net,net,.4,4)
        for a,b in zip(left,right,strict=True):
            assert np.array_equal(a,b)
        start=time.perf_counter()
        collector.collect(net,net,.4,32)
        dt=time.perf_counter()-start
        data[dev]=dict(E=512,seconds=dt,steps_per_second=512/dt,restored_window_exact=True)
    out.write_text(json.dumps(dict(hidden_input_invariant=True,devices=data),indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    main(Path(sys.argv[1]),Path(sys.argv[2]))
