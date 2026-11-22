"""Apply the competition elimination rule and price low-cash positions in the search."""
import sys
from pathlib import Path
from build_initiative import build

def patch(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1, old[:60]
        source=source.replace(old,new)
    runtime=Path(__file__).with_name('liquidity.py').read_text()
    replace('def _search(obs,',runtime+'\n\ndef _search(obs,')
    replace('    most = 2\n','''    hire_floor = min(configuration.get('recCollectorCost',30), configuration.get('recPlanterCost',30))
    most = 2
''')
    replace('        return b[2][\'step\'] > LAST_TURN','''        board = b[2]
        return board['step'] > LAST_TURN or any(_v6_eliminated(p, hire_floor) for p in board['players'])''')
    replace("        c = [float(pl[0]) for pl in b[2]['players']]",'''        board = b[2]
        out = [_v6_eliminated(pl, hire_floor) for pl in board['players']]
        if any(out):
            # The judge freezes the survivor's reward at the preceding turn.
            c = [board['step'] - (LAST_TURN+2) - 1 if lost else float(b[3][side][0])
                 for side, lost in enumerate(out)]
        else:
            c = [float(pl[0]) for pl in board['players']]''')
    replace('                v.append(float(value))', '''                if min(pl[0] for pl in future['players']) < 2 * hire_floor:
                    value += _v6_liquidity(future, me, hire_floor)
                v.append(float(value))''')
    return source

if __name__ == '__main__':
    output = Path(sys.argv[1])
    build(output, Path(sys.argv[2]))
    s=patch(output.read_text())
    compile(s,str(output),'exec')
    output.write_text(s)
