"""Independently replay the suffix actions, summarize paired score changes, and separate map cohorts."""
import gzip
import hashlib
import json
import os
import statistics

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from branch_replay import OUT, ROOT, HERE, board_digest, digest, step, write
from benchmark import configure_engine, new_game, validate_actions
from close_replay import archive_state, restore


def summarize(cases, names):
    output = {}
    for name in names:
        a = [c['arms'][name] for c in cases]
        margins = [v['score_margin'] for v in a]
        output[name] = dict(wins=sum(v['won'] == 1 for v in a), losses=sum(v['won'] == -1 for v in a),
            ties=sum(v['won'] == 0 for v in a), mean_final=round(statistics.mean(margins),2),
            mean_second_half=round(statistics.mean(m-c['initial_margin'] for m,c in zip(margins,cases)),2))
    comparisons = {}
    for first,other in (('soundings','tidewatch'),('soundings','estuary'),('tidewatch','estuary')):
        rows = [(c['arms'][first], c['arms'][other]) for c in cases]
        delta = [round(a['score_margin']-b['score_margin'],2) for a,b in rows]
        comparisons[first+'-minus-'+other] = dict(
            wins_gained=sum(a['won']==1 and b['won']!=1 for a,b in rows), wins_lost=sum(a['won']!=1 and b['won']==1 for a,b in rows),
            improved=sum(x>0 for x in delta), declined=sum(x<0 for x in delta), equal=delta.count(0),
            mean=round(statistics.mean(delta),2), median=round(statistics.median(delta),2), individual=delta)
    # Categories are controlled-vs-rival net cash contributions since the fork.
    trajectory={}
    for end in (175,200,225,250,275,299):
        arms={}
        for name in names:
            gaps={k:[] for k in ('margin','transport_income','tree_income','tree_payment','recruit_payment')}
            for c in cases:
                r=c['arms'][name];s=c['side'];m=r['milestones'][str(min(end,r['end']))];ledger=m['cash_flow']
                gaps['margin'].append(m['our_margin'])
                for key,events in (('transport_income',('bank','base_collision')),('tree_income',('tree_yield','tree_collision')),
                        ('tree_payment',('plant','seize')),('recruit_payment',('hire_collector','hire_planter'))):
                    gaps[key].append(sum(ledger[s].get(e,0)-ledger[1-s].get(e,0) for e in events))
            arms[name]={k:round(statistics.mean(v),2) for k,v in gaps.items()}
        trajectory[str(end)]=dict(arms=arms, soundings_minus_tidewatch={k:round(arms['soundings'][k]-arms['tidewatch'][k],2) for k in arms['soundings']})
    return dict(arms=output, paired=comparisons, trajectory=trajectory)


def render(cases, groups):
    lines = ['# 第 150 步更换策略的配对实验', '',
        '从 8 张旧比赛地图还原第 150 步，同一盘面分别交给 Soundings、Tidewatch、v6 Estuary。另一席始终由实时 v6 接手，每回合按新的棋面决策。开发组为 v6r 第五轮资格赛，留出组为 v6r 第五轮确认赛，各选四张独立地图，均含席 0、席 1 各两次。按种子哈希和席位预先选择，未按原局输赢或前半盘现金差筛选。', '',
        '| 分组 | 种子 | 接手席位 | 第 150 步分差 | Soundings 终局 | Tidewatch 终局 | v6 终局 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    labels = {'development':'开发', 'holdout':'留出', 'all':'合计'}
    for c in cases:
        r = c['arms']
        lines.append(f"| {labels[c['split']]} | {c['seed']} | {c['side']} | {c['initial_margin']:+,.2f} | "
            + ' | '.join(f"{r[n]['score_margin']:+,.2f}" for n in ('soundings','tidewatch','estuary'))+' |')
    lines += ['', '| 分组 | 接手策略 | 胜–负–平 | 平均终局分差 | 从第 150 步到终局的平均分差变化 |',
              '| --- | --- | ---: | ---: | ---: |']
    for g, v in groups.items():
        for n,a in v['arms'].items():
            lines.append(f"| {labels[g]} | {n} | {a['wins']}–{a['losses']}–{a['ties']} | {a['mean_final']:+,.2f} | {a['mean_second_half']:+,.2f} |")
    lines+=['','| Soundings 对 Tidewatch | 换成 Soundings 后多赢 | 换成 Soundings 后少赢 | 终局分差上升/下降/相等 | 平均分差效果 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for g,v in groups.items():
        q=v['paired']['soundings-minus-tidewatch']
        lines.append(f"| {labels[g]} | {q['wins_gained']} | {q['wins_lost']} | {q['improved']}/{q['declined']}/{q['equal']} | {q['mean']:+,.2f} |")
    lines+=['', '## 适用范围', '',
        '八张图里七张的指定席位在中盘领先，一张落后。主样本的八个接手席位在真实前半场都由 v6 操作，对手为 v5。此设计衡量在这些棋面上交给不同策略的直接效果；它没有覆盖由 Tidewatch 或 Soundings 实际打出前半场后形成的全部局面。每个棋面仅有一个共同决策种子。', '',
        '每一席只看到自身合法视野，从第 0 步以真实历史动作预热信念、资金跟踪与短期路线。每次分叉复原工人和树 ID 计数。双方每回合搜索最多四个候选、两层、一个应手及三步尾评估。推演使用固定工作量，运行时核实 Torch 两种并行线程及已加载的数学线程池均为一。正式线上搜索会按时间调整预算。', '',
        '逐席分数、每 25 步现金分类账、策略 SHA、布局 SHA 和种子见 `phase-switch.json`。所有分支逐步的双方动作保存在 `phase-switch-actions.json.gz`。资格与确认原局逐步核对现金及原终局分数。另用 `experiments/v7/phase_switch_report.py` 独立重播全部 24 个后半局，对照动作摘要、中间现金和终局盘面。']
    (OUT/'phase-switch.md').write_text('\n'.join(lines)+'\n')



def render_deficit(cases, groups):
    g=groups['all'];order=('soundings','tidewatch','estuary')
    lines=['# 第 150 步逆风接手压力测试','',
      '预定的主八图只有一局在第 150 步落后。这里从旧 v6r 第五轮资格、确认与回归记录中，在开跑前按第 150 步资金落后、整图哈希顺序和席位平衡另选四图。四张图与主八图互不重复；选样只用第 150 步的现金，未用历史终盘结果。三个模型从完全相同的棋面接手，同样对阵逐步重新决策的冻结 v6。','',
      '| 地图种子 | 席位 | 前半场策略 | 起始分差 | Soundings 终局 | Tidewatch 终局 | 冻结 v6 终局 |',
      '| ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for c in cases:
        lines.append(f"| {c['seed']} | {c['side']} | {c['historical_names'][c['side']]} | {c['initial_margin']:+,.2f} | "+
            ' | '.join(f"{c['arms'][n]['score_margin']:+,.2f}" for n in order)+' |')
    lines+=['','| 接手策略 | 翻盘胜/负/平 | 平均终局分差 | 比接手时平均追上 |','| --- | ---: | ---: | ---: |']
    for name in order:
        a=g['arms'][name]
        lines.append(f"| {name} | {a['wins']}/{a['losses']}/{a['ties']} | {a['mean_final']:+,.2f} | {a['mean_second_half']:+,.2f} |")
    lines+=['','## 如何解读','',
      '主八图均从 v6 历史席接手，而本组有三席由原 v5 打出前半场，一席由 v6 打出。压力测试刻意覆盖落后局面，其胜率只能按本组报告。这些棋面没有给出在第 150 步固定更换策略或采用一个现金分差阈值的独立验证。','',
      '每条分支自第 150 步演到终局，每回合双方最多四个候选、两层、一种对手回复和三步尾评估；每次决策给两组相同的随机数种子。单核且在导入前限制数学库线程，实际运行线程数见原始摘要。全部 12 次分支均已从原动作独立重播，核对中间现金和终局完整盘面。','',
      '固定选样见 `phase-deficit-schedule.json`，分支与账本见 `phase-deficit.json`，全部双方逐步动作见 `phase-deficit-actions.json.gz`，复核与各阶段均值见 `phase-deficit-summary.json`。']
    (OUT/'phase-deficit.md').write_text('\n'.join(lines)+'\n')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deficit', action='store_true')
    deficit=parser.parse_args().deficit
    configure_engine()
    prefix='phase-deficit' if deficit else 'phase-switch'
    runner='phase_deficit.py' if deficit else 'phase_switch.py'
    d=json.loads((OUT/(prefix+'.json')).read_text())
    assert hashlib.sha256((HERE/runner).read_bytes()).hexdigest()==d['runner_sha256']
    traces=json.loads(gzip.decompress((OUT/(prefix+'-actions.json.gz')).read_bytes()))
    names=tuple(d['sha256']);cases=d['cases']
    if deficit:
        path=OUT/'phase-deficit-schedule.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==d['schedule_sha256']
        assert hashlib.sha256((HERE/'phase_switch.py').read_bytes()).hexdigest()==d['shared_runner_sha256']
        schedule=[('deficit',c['seed'],c['side'],c['layout']) for c in json.loads(path.read_text())['cases']]
    else:
        schedule=d['schedule']
    assert len(cases)==len(schedule)==len(set(c['layout'] for c in cases))==(4 if deficit else 8)
    for c, expected in zip(cases,schedule):
        split,seed,side,layout=expected
        assert [c['split'],c['seed'],c['side'],c['layout']]==list(expected) and set(c['arms'])==set(names)
        state,env=new_game(seed,300)
        fname=c['source'] if deficit else 'v6r-'+('qual' if split=='development' else 'confirm')+'-05-estuary.jsonl'
        rows=[json.loads(raw) for raw in (ROOT/'analysis'/fname).open()]
        row,=[r for r in rows if r['seed']==seed and [q['name'] for q in r['players']]==c['historical_names']]
        for actions,post in row['actions_and_post_cash'][:150]:
            step(state,env,actions)
            assert post==[p[0] for p in state[0].observation.players]
        assert board_digest(state)==c['initial_board_sha256']
        snapshot=archive_state(state);config=dict(env.configuration)
        assert snapshot['id_counters']==c['initial_id_counters']
        for name in names:
            s,env=restore(snapshot,config);record=c['arms'][name]
            plays=traces[f'{seed}/{side}'][name]
            assert digest(plays)==record['actions_sha256']
            for i,actions in enumerate(plays,151):
                for seat,action in enumerate(actions):validate_actions(action,s[0].observation.players[seat])
                step(s,env,actions)
                if str(i) in record['milestones']:
                    assert [p[0] for p in s[0].observation.players]==record['milestones'][str(i)]['cash']
            assert record['end']==i and (i==299 or all(p.status=='DONE' for p in s))
            assert [p.reward for p in s]==record['scores'] and board_digest(s)==record['terminal_hash']
    groups=() if deficit else ('development','holdout')
    summaries={group:summarize([c for c in cases if c['split']==group], names) for group in groups}
    summaries['all']=summarize(cases,names)
    write(OUT/(prefix+'-summary.json'),dict(verified_suffixes=sum(len(c['arms']) for c in cases),
        measure='Positive margin and a win are from the branch-taking seat, including seat 1.',groups=summaries))
    if deficit:render_deficit(cases,summaries)
    else:render(cases,summaries)
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    main()
