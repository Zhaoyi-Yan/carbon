"""Use the last position of our own collectors to record completed trips between depots."""


def _v8_observe_trips(memory, own, previous, turn):
    if turn == 0:
        memory.clear()
    active = memory.setdefault('starts', {})
    trips = memory.setdefault('completed', [])
    ports = set(own[1].values())
    workers = own[2]
    for uid in list(active):
        if uid not in workers:
            del active[uid]
    for uid, (pos, _, kind) in workers.items():
        if kind != 'COLLECTOR' or pos not in ports:
            continue
        last = previous[2].get(uid) if previous else None
        if uid in active and last and last[0] not in ports and last[1] > 0:
            start, origin = active[uid]
            trips.append((turn, origin, turn-start, float(last[1])))
            active[uid] = (turn, pos)
        elif uid not in active:
            active[uid] = (turn, pos)
    return trips
