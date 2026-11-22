"""Summarize recorded late-game balance, geometric delivery deadlines and fixed-action interventions."""
import argparse
import json
from pathlib import Path
from statistics import mean,median

HERE=Path(__file__).resolve().parent


def cash_categories(row):
    return dict(deposits=row.get("bank",0),trees=sum(row.get(k,0) for k in ("tree_yield","tree_collision","plant","seize")),
                hires=sum(row.get(k,0) for k in ("hire_collector","hire_planter")),base_collision=row.get("base_collision",0),rounding=row.get("rounding",0))


def category(m):
    if m<0:return "close_loss" if m>=-350 else "larger_loss"
    return "close_win" if m<=350 else "larger_win"


def summarize():
    ledger=json.loads((HERE/"close-ledger.json").read_text())["cases"]
    cohorts=json.loads((HERE/"close-cohort.json").read_text())["cases"]
    forks={g["key"]:g for g in json.loads((HERE/"close-counterfactual.json").read_text())["cases"]}
    games=[]
    for g in ledger:
        s=g["seat"];o=1-s;c=cohorts[g["key"]];marks=g["milestones"];label=category(g["margin"])
        margins={k:round(v[s]["cash"]-v[o]["cash"],2) for k,v in marks.items()}
        sides=[]
        for side in (s,o):
            start=c["initial_carbon"][side];deaths=g["deaths25"][side];r=g["receipts25"][side]
            feasible=[x for x in r if x["prior_approach_lighter_interceptors"] and x["amount"]>=30]
            first=c["disposition"][side].get("bank_seat"+str(side),0)
            terminal=c["terminal_workers"][side]
            sides.append(dict(original_seat=side,start_carried=start,initial_stock_banked_by_own_side=first,
                          initial_stock_banked_fraction=first/start if start else None,terminal_cargo=marks["299"][side]["cargo"],
                          terminal_cargo_with_no_reachable_loaded_decision_on_final_trip=sum(v["cargo"] for v in terminal if v["last_move_with_reachable_carried_stock"] is None),
                          funds25=cash_categories(g["cash25"][side]),gross_death_cargo=sum(d["cargo"] for d in deaths),
                          collector_deaths=sum(d["role"]=="COLLECTOR" for d in deaths),planter_deaths=sum(d["role"]=="PLANTER" for d in deaths),
                          hires25=len(g["hires25"][side]),hires_last8=sum(d["turn"]>=291 for d in g["hires25"][side]),
                          direct_bank_receipts_ge30=sum(v["amount"]>=30 for v in r),hindsight_one_move_interceptable_receipts=len(feasible),
                          hindsight_one_move_interceptable_cash=sum(v["amount"] for v in feasible)))
        delta={k:sides[0]["funds25"][k]-sides[1]["funds25"][k] for k in sides[0]["funds25"]}
        if abs(margins["274"]+sum(delta.values())-g["margin"])>.011:raise ValueError("endgame components do not add")
        branches={}
        for name in ("early_bank","preblock"):
            f=forks[g["key"]][name]
            money=[cash_categories(p) for p in f["cash25"]]
            diff=[{k:money[seat][k]-sides[pid]["funds25"][k] for k in money[seat]} for pid,seat in enumerate((s,o))]
            if abs(sum(diff[0].values())-sum(diff[1].values())-f["change"])>.011:raise ValueError("branch components do not add")
            branches[name]=dict(edits=len(f["edits"]),margin=f["alternative_margin"],change=f["change"],funds_changes=diff,
                                absent_opponent_commands=f["command_mapping"][o].get("absent_order",0),
                                reattributed_opponent_commands=f["command_mapping"][o].get("different_birth_order",0))
        games.append(dict(key=g["key"],type=label,candidate=g["candidate"],seat=s,seed=g["seed"],margin=g["margin"],
                          checkpoint_margin=margins,late_cash_delta=delta,sides_candidate_first=sides,branches=branches))
    summary={}
    for label in ("close_loss","larger_loss","close_win","larger_win","all"):
        xs=[g for g in games if label=="all" or g["type"]==label];n=len(xs)
        side=[g["sides_candidate_first"][0] for g in xs];rivals=[g["sides_candidate_first"][1] for g in xs]
        summary[label]=dict(games=n,behind_at_274=sum(g["checkpoint_margin"]["274"]<0 for g in xs),
            median_margin_274=median(g["checkpoint_margin"]["274"] for g in xs),median_final_margin=median(g["margin"] for g in xs),
            median_candidate_terminal_cargo=median(x["terminal_cargo"] for x in side),
            losses_covered_by_candidate_terminal_cargo=sum(g["margin"]<0 and v["terminal_cargo"]>=-g["margin"] for g,v in zip(xs,side)),
            candidate_carried_at_274=sum(x["start_carried"] for x in side),candidate_274_stock_banked=sum(x["initial_stock_banked_by_own_side"] for x in side),
            candidate_274_stock_banked_fraction=sum(x["initial_stock_banked_by_own_side"] for x in side)/sum(x["start_carried"] for x in side),
            candidate_total_terminal_cargo=sum(x["terminal_cargo"] for x in side),
            candidate_terminal_cargo_without_geometrically_feasible_loaded_move=sum(x["terminal_cargo_with_no_reachable_loaded_decision_on_final_trip"] for x in side),
            candidate_hires_last8=sum(x["hires_last8"] for x in side),games_behind_on_deposits25=sum(g["late_cash_delta"]["deposits"]<0 for g in xs),
            median_cash25={k:median(g["late_cash_delta"][k] for g in xs) for k in xs[0]["late_cash_delta"]},
            median_own_enemy_deaths=[[median(x[k] for x in sides) for k in ("collector_deaths","planter_deaths","gross_death_cargo")] for sides in (side,rivals)],
            enemy_hindsight_blockable_count=sum(r["hindsight_one_move_interceptable_receipts"] for r in rivals),
            enemy_receipts_ge30=sum(r["direct_bank_receipts_ge30"] for r in rivals),
            enemy_hindsight_blockable_cash=sum(r["hindsight_one_move_interceptable_cash"] for r in rivals))
        branches={}
        for name in ("early_bank","preblock"):
            bs=[g["branches"][name] for g in xs]
            branches[name]=dict(changed=sum(bool(b["edits"]) for b in bs),positive=sum(b["change"]>0 for b in bs),negative=sum(b["change"]<0 for b in bs),
                                mean_delta=mean(b["change"] for b in bs),median_delta=median(b["change"] for b in bs),wins_before=sum(g["margin"]>0 for g in xs),
                                wins_after=sum(b["margin"]>0 for b in bs))
        summary[label]["alternatives"]=branches
    return dict(definitions=dict(window="Decision steps 274..298 inclusive, 25 turns ending at state 299.",subset="Every final loss plus every win with absolute final margin <=350. All other wins retained as a descriptive control.",
                                geometric_reach="Shortest toroidal path with no other workers or blockers. The 15x15 board has maximum toroidal distance 14; all loads are geometrically reachable at steps 250,275 and 285. Steps 290,294,297,298 also saved as reachability summaries.",
                                initial_carbon="Only carbon already carried at state 274. Tag proportions follow transfers among workers, bank payouts and tree taxes. Ground spills end attribution.",
                                blocked="Hindsight candidate only. For opponent return of >=30 at t, look at the preceding t-1 decision: a lighter defender could step to the opponent's recorded non-depot approach cell. Same-turn collisions on the depot can still bank for its owner.",
                                counterfactual="Both rules are public-information rules; other own and all opponent orders are frozen, with missing or reassigned archived IDs counted. Counterfactual outcomes are not wins against a responding opponent."),
                groups=summary,games=games)


def tidy(x):
    if isinstance(x,float):return round(x,5)
    if isinstance(x,list):return list(map(tidy,x))
    if isinstance(x,dict):return {k:tidy(v) for k,v in x.items()}
    return x


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=HERE/"close-results.json")
    a=p.parse_args();a.output.write_text(json.dumps(tidy(summarize()),indent=2)+"\n")
