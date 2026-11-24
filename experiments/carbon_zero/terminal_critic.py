"""Calibrate a critic on frozen-policy Monte Carlo returns before updating the actor."""
import argparse
import copy
from pathlib import Path
import time

from pilot import Actor, FT3_SHA256, STYLE, load_ft3, np, torch, write
from policy import public_state_features
from terminal_games import group_matches, play


def dataset(teacher, actor, device, batches, namespace):
    features, labels, gaps, turns = [], [], [], []
    audits, match_rows, costs, groups = [], [], [], []
    for batch in range(batches):
        matches = group_matches(namespace, batch)
        played, timing = play(teacher, actor, actor, matches, device, capture='critic')
        for row, trace, _ in played:
            for embedding, gap, turn, observation in trace:
                label = row['utility'] - 0.5 * np.tanh(gap / 1000.)
                groups.append(len(match_rows))
                features.append(embedding)
                labels.append(label)
                gaps.append(gap)
                turns.append(turn)
                if observation is not None and len(audits) < 96:
                    audits.append((observation, public_state_features(gap, turn), label))
            match_rows.append(row)
        costs.append(timing)
    return (np.stack(features), np.asarray(labels, dtype=np.float32),
            np.asarray(gaps), np.asarray(turns), audits, match_rows, costs,
            np.asarray(groups, dtype=np.int64))


def metrics(pred, truth):
    residual = truth - pred
    var = float(np.var(truth))
    correlation = float(np.corrcoef(pred, truth)[0, 1]) if np.ptp(pred) and np.ptp(truth) else None
    return dict(MSE=float(np.mean(residual ** 2)), bias=float(np.mean(-residual)),
                explained_variance=1. - float(np.var(residual)) / var if var else None,
                correlation=correlation, N=len(truth))


def scores(predictions, labels, gap, turns):
    phases = {'all': np.ones(len(turns), dtype=bool), 'opening': turns <= 60,
              'middle': (turns > 60) & (turns < 230), 'ending': turns >= 230}
    phases['leading'] = gap > 0
    phases['trailing'] = gap < 0
    return {
        title: {name: metrics(pred[mask], labels[mask]) for name, pred in predictions.items()}
        for title, mask in phases.items() if mask.any()
    }


def public_features(gap, turn):
    # A tuned cash-only comparator. Cash magnitude and time interact late in the game.
    return public_state_features(gap, turn).astype(np.float64)


def fit_value(actor, train_x, train_y, seed, *, train_pub=None, epochs=12, weight_decay=1e-3):
    device = next(actor.parameters()).device
    features = torch.as_tensor(train_x, device=device)
    labels = torch.as_tensor(train_y, device=device)
    pub = torch.as_tensor(train_pub, dtype=torch.float32, device=device) if train_pub is not None else None
    opt = torch.optim.AdamW(actor.value.parameters(), lr=0.001, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(features))
        total = 0.
        for start in range(0, len(order), 1024):
            ids = torch.as_tensor(order[start:start + 1024], device=device)
            pred = actor.predict_value(features[ids], pub[ids] if pub is not None else None)
            loss = (pred - labels[ids]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(ids)
        losses.append(total / len(features))
    return losses


def head_vs_trunk(actor, audit, device):
    """Replay the same value-only update under head-only, shared-trunk, and detached-trunk modes."""
    x, pub, y = zip(*audit[:32], strict=True)
    x = torch.as_tensor(np.stack(x), device=device)
    pub_t = torch.as_tensor(np.stack(pub), dtype=torch.float32, device=device) if actor.residual_critic else None
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    with torch.no_grad():
        before = [z.detach() for z in (actor(x, pub=pub_t) if actor.residual_critic else actor(x))[:2]]
    result = {}
    modes = [('head_only', False, False), ('shared_trunk', True, False)]
    if actor.residual_critic:
        modes.append(('detached_trunk', True, True))
    for label, shared, detach in modes:
        model = copy.deepcopy(actor)
        model.detach_value_trunk = detach
        params = model.parameters() if shared else model.value.parameters()
        opt = torch.optim.AdamW(params, lr=3e-5)
        out = model(x, pub=pub_t) if model.residual_critic else model(x)
        loss = (out[2] - y).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        with torch.no_grad():
            after = (model(x, pub=pub_t) if model.residual_critic else model(x))[:2]
            shift = [(a - b).abs().max().item() for a, b in zip(after, before, strict=True)]
        result[label] = dict(
            max_worker_logit_shift=shift[0], max_base_logit_shift=shift[1],
            gradient_norm_before_clip=float(norm))
        if (not shared or detach) and shift != [0., 0.]:
            raise AssertionError(f'Value update in mode {label} changed actor logits: {shift}')
    return result


def main(a):
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=False)
    teacher = load_ft3(a.ft3)
    style = teacher._POLICY.sid(STYLE)
    assert style == 122
    actor = Actor(teacher._POLICY, style, residual_critic=a.residual).to(a.device)
    fit = dataset(teacher, actor, a.device, a.fit_batches, ('terminal-critic-fit', a.seed))
    check = dataset(teacher, actor, a.device, a.test_batches, ('terminal-critic-check', a.seed, a.validation_panel))
    x, y, gap, turn, audit, train_games, train_cost, _ = fit
    hx, hy, hg, ht, _, held_games, held_cost, held_groups = check
    if {g['layout_sha256'] for g in train_games} & {g['layout_sha256'] for g in held_games}:
        raise ValueError('Critic training / validation layout overlap')
    train_pub = public_features(gap, turn)
    held_pub = public_features(hg, ht)
    cash_model = np.linalg.lstsq(train_pub, y, rcond=None)[0]
    if a.residual:
        with torch.no_grad():
            actor.value_baseline.weight.copy_(torch.as_tensor(cash_model[None, :], dtype=torch.float32, device=a.device))
    cold_audit = head_vs_trunk(actor, audit, a.device)
    start = time.perf_counter()
    losses = fit_value(
        actor, x, y, a.seed,
        train_pub=train_pub.astype(np.float32) if a.residual else None,
        epochs=a.epochs, weight_decay=a.weight_decay,
    )
    fit_seconds = time.perf_counter() - start
    audit_result = head_vs_trunk(actor, audit, a.device)
    with torch.no_grad():
        hx_t = torch.as_tensor(hx, device=a.device)
        hp_t = torch.as_tensor(held_pub, dtype=torch.float32, device=a.device) if a.residual else None
        prediction = actor.predict_value(hx_t, hp_t).cpu().numpy()
    preds = dict(value=prediction, initial_zero=np.zeros_like(hy),
                 constant=np.full_like(hy, y.mean()), cash_only=held_pub @ cash_model)
    np.savez(a.out / 'heldout.npz', labels=hy, value=prediction, cash=preds['cash_only'],
             constant=preds['constant'], gap=hg, turn=ht, group=held_groups)
    held = scores(preds, hy, hg, ht)
    # Require residual critic to beat both constant and cash_only on all, middle, and ending.
    required_phases = ('all', 'middle', 'ending') if a.residual else ('all', 'middle')
    qualified = all(held[phase]['value']['MSE'] < min(
        held[phase]['constant']['MSE'], held[phase]['cash_only']['MSE'])
        for phase in required_phases)
    if a.residual:
        critic_state = dict(
            residual_critic=True,
            value_baseline={k: v.cpu() for k, v in actor.value_baseline.state_dict().items()},
            value={k: v.cpu() for k, v in actor.value.state_dict().items()},
        )
    else:
        critic_state = {k: v.cpu() for k, v in actor.value.state_dict().items()}
    torch.save(critic_state, a.out / 'critic.pt')
    report = dict(source_sha256=FT3_SHA256, style=style, seed=a.seed, device=a.device,
                  residual_critic=a.residual, epochs=a.epochs, weight_decay=a.weight_decay,
                  required_phases=list(required_phases),
                  train_E=len(x), heldout_E=len(hx), fit_seconds=fit_seconds,
                  fit_losses=losses, fit_collection=train_cost, heldout_collection=held_cost,
                  heldout=held, cold_gradient_diagnostic=cold_audit,
                  shared_gradient_diagnostic=audit_result, qualified=qualified,
                  training_games=train_games, heldout_games=held_games)
    write(a.out / 'critic.json', report)
    print('Critic complete', len(x), len(hx), 'residual', a.residual, 'qualified', qualified, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ft3', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--seed', type=int, default=24000000)
    parser.add_argument('--fit-batches', type=int, default=7)
    parser.add_argument('--test-batches', type=int, default=2)
    parser.add_argument('--validation-panel', default='initial')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    main(parser.parse_args())
