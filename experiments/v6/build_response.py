"""Build the online-replay response on top of the frozen winner-weighted action heads."""
import sys
from pathlib import Path
from learn_export import export

HERE = Path(__file__).resolve().parent

def build(output):
    export(HERE.parents[1]/'yan-v5.py',HERE/'learned-heads.npz',output)
    source=output.read_text()
    insert='def _act(obs,'
    hook='    workers, _, bases, cash = parse(players)\n    act = {}'
    assert source.count(insert)==1 and source.count(hook)==1
    source=source.replace(insert,(HERE/'initiative.py').read_text()+'\n'+(HERE/'response.py').read_text()+'\n'+insert)
    source=source.replace(hook,'''    if configuration is not None and _SEARCH_OK:
        lw,lb = _v6_response(obs,est,configuration,lw,lb)
'''+hook)
    compile(source,str(output),'exec')
    output.write_text(source)

if __name__=='__main__':
    build(Path(sys.argv[1]))
