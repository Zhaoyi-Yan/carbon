"""Revalue current cargo when scoring the final collision chooser's simulated joint replies."""
import argparse
import inspect
from pathlib import Path

if __package__:
    from . import receipts
    from .postcargo_model import _post_cargo_adjust
else:
    import receipts
    from postcargo_model import _post_cargo_adjust


def build(code):
    anchor='def _reply_assets(board,side,config):'
    old='            vals.append(_reply_assets(_turn(root,config,pair),me,config))'
    if code.count(anchor)!=1 or code.count(old)!=1 or 'def _post_cargo_adjust(' in code:
        raise ValueError('Expected the original final collision critic and branch sampler')
    if 'def _v7_receipts(' not in code:
        code=receipts.build(code)
    code=code.replace(anchor,inspect.getsource(_post_cargo_adjust)+'\n\n'+anchor)
    return code.replace(old,'            future=_turn(root,config,pair)\n'
                            '            vals.append(_reply_assets(future,me,config)+_post_cargo_adjust(future,me))')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();src=build(a.parent.read_text());compile(src,str(a.output),'exec')
    with a.output.open('x') as f:f.write(src)
