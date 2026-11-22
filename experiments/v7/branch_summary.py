"""Summarize a paired intervention experiment on a single historical board."""
import argparse
import json
from pathlib import Path
from statistics import mean, median
from branch_replay import OUT, write


def summary(rows):
    recorded = next(r for r in rows if r["mode"]=="recorded")
    controls = {(r["proxy"],r["seed"]):r for r in rows if r["mode"]=="control"}
    paired = []
    for r in rows:
        if r["mode"] in ("recorded","control"):
            continue
        c = controls[r["proxy"],r["seed"]]
        effects = {}
        for t,im in r["milestones"].items():
            cm = c["milestones"][t]
            effects[t] = dict(cash_margin=round(im["margin"]-cm["margin"],2),
                              bank=[round(a-b,2) for a,b in zip(im["deposits"],cm["deposits"])],
                              opposing_bank_denied=round(cm["deposits"][1]-im["deposits"][1],2))
        paired.append(dict(mode=r["mode"], proxy=r["proxy"],seed=r["seed"],edited_turns=len(r["edits"]),
                           edited_commands=sum(len(x[1]) for x in r["edits"]), effects=effects,
                           margin=r["milestones"][str(r["last_step"])]["margin"],
                           control_margin=c["milestones"][str(c["last_step"])]["margin"]))
    groups = []
    for mode in sorted({x["mode"] for x in paired}):
        for proxy in sorted({x["proxy"] for x in paired}) + ["pooled_on_one_board"]:
            group = [x for x in paired if x["mode"]==mode and (x["proxy"]==proxy or proxy=="pooled_on_one_board")]
            stages = {}
            for t in group[0]["effects"]:
                deltas = [x["effects"][t]["cash_margin"] for x in group]
                stages[t] = dict(mean_margin_delta=round(mean(deltas),2), median_margin_delta=round(median(deltas),2),
                                 range_margin_delta=[min(deltas),max(deltas)], positive_margin_delta=sum(d>0 for d in deltas),
                                 mean_bank_delta=[round(mean(x["effects"][t]["bank"][s] for x in group),2) for s in (0,1)])
            groups.append(dict(mode=mode,proxy=proxy,replicates=len(group),terminal_wins=sum(x["margin"]>0 for x in group),
                               control_wins=sum(x["control_margin"]>0 for x in group), stages=stages))
    return dict(real_recorded=recorded["milestones"], paired_differences=paired, groups=groups,
                scope="One map, one actual side and starting state. Paired seeds vary policy sampling, not maps. "
                      "The two opponents are behavior-cloned policies of the same author. Treat terminal wins as conditional simulations.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=OUT/"branch-results.json")
    p.add_argument("--out",type=Path,default=OUT/"branch-summary.json")
    a=p.parse_args()
    result=summary(json.loads(a.input.read_text())["results"])
    write(a.out,result)
    for g in result["groups"]:
        if g["proxy"]=="pooled_on_one_board":
            print(g["mode"],g["terminal_wins"],g["control_wins"],[(t,v["mean_margin_delta"],v["mean_bank_delta"]) for t,v in g["stages"].items()])
