"""Repeat the twelve predeclared interventions with the validated stochastic Apollo proxy."""
import json
import os
import sys
import branch_run as runner


def main():
    if os.environ.get("PYTHONHASHSEED")!="0":
        os.environ["PYTHONHASHSEED"]="0"
        os.execv(sys.executable,[sys.executable,*sys.argv])
    runner.limit_math_threads()
    runner.initialize()
    import surrogate
    from surrogate_variation import make
    surrogate.make = make
    results=[]
    for seed in runner.SEEDS:
        for mode in runner.MODES:
            row=runner.run(("apollo-varied",seed,mode,False,299))
            results.append(row)
            print(row["key"],row["milestones"]["299"]["margin"],flush=True)
    source=json.loads((runner.OUT/"branch-results.json").read_text())
    recorded=next(r for r in source["results"] if r["mode"]=="recorded")
    baseline={k:v for k,v in source.items() if k not in ("results","verified_repetition")}
    for name,expected in baseline["module_sha256"].items():
        path=(runner.ROOT/"experiments/v6r/checkpoints/grove.py" if name=="grove.py" else runner.HERE/name)
        if runner.sha(path)!=expected:
            raise ValueError(f"Changed since primary comparison: {name}")
    baseline["module_sha256"]["surrogate_variation.py"]=runner.sha(runner.HERE/"surrogate_variation.py")
    runner.write(runner.OUT/"branch-varied-results.json",dict(baseline,results=results+[recorded]))


if __name__=="__main__":
    main()
