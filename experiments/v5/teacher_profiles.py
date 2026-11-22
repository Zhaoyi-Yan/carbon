"""Publish per-seat behavior summaries by game phase, giving each map one vote."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','split','output'):p.add_argument(k,type=Path)
    a=p.parse_args()
    here=Path(__file__).parent;rank=json.loads((here/'design.json').read_text())['ranking']
    split=json.loads(a.split.read_text())
    rows=[[] for _ in rank]
    manifest=[]
    for partition,key in [('train','training_matches'),('validation','validation_matches')]:
        for name in sorted(split[key]):
            with np.load(a.cache/(name+'.npz')) as z:
                ids=z['J'];by_teacher=[]
                for teacher in np.unique(ids):
                    teacher=int(teacher);mine=ids==teacher
                    phases=[]
                    for phase in range(4):
                        mask=mine & (z['P']==phase)
                        phases.append(z['B'][mask].mean(0).tolist())
                    rows[teacher].append(phases);by_teacher.append(teacher+1)
                manifest.append(dict(match=name,seed=int(z['seed']),layout=str(z['layout']),
                                     teacher_ranks=by_teacher,partition=partition))
    counts=[len(v) for v in rows]
    assert len(manifest)==len({g['seed'] for g in manifest})
    lines=['# 排行榜前二十的风格差异','',
           '272 张地图共 471 个前二十席位。下表先对每位高手在每张地图上的表现求平均，再对地图求平均。每五步采样一次。开局指第 0–74 步，中盘指第 75–149 步。深入对方半场指距敌方最近基地比距己方最近基地更近，轻载指携带少于 30 碳。树收入采用当步公开的实际收入。','',
           '| 当时排名 | 策略 | 席位 | 开局种植员占比 | 开局采集员深入敌方半场 | 中盘轻载采集员占比 | 中盘平均树数 | 中盘树收入/步 |',
           '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r,units in zip(rank,rows):
        avg=np.array(units).mean(0)
        fmt=lambda w:f'{w:.0%}'
        lines.append(f"| {r['rank']} | {r['name']} | {len(units)} | {fmt(avg[0,0])} | {fmt(avg[0,1])} | {fmt(avg[1,2])} | {avg[1,6]*10:.1f} | {avg[1,7]*50:.1f} |")
    lines.extend(['','树数和树收入分别按每步最多 20 棵、100 碳截断，与训练时用到的观测特征一致。各席位的对手和地图不同，表中的差异只能用来描述风格。策略选择、预测以及换边实战分别验收。',
                  '','[冻结的排名与策略 ID](design.json) · [逐局地图、参赛教师与数据划分](replay-manifest.json)'])
    a.output.write_text('\n'.join(lines)+'\n')
    (a.output.parent/'replay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('maps',len(manifest),'seats',sum(counts))


if __name__=='__main__':
    main()
