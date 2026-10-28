"""Plan one dish for one chef as a short list of macro-actions.

The chef carries a plate the whole time and builds the dish on it. A macro is
one trip and one action: fetch a plate, take a stored ingredient onto the
plate, put the plate's food into a station, chop it, take a station's food
back onto the plate (combining it with what is there), bin the plate's food,
or deliver. A uniform-cost search over (time, tile, plate, my stations) finds
the plan that delivers soonest. Stations that cook by themselves keep cooking
while the chef walks, so the search interleaves naturally, for example
chopping chocolate while the cake bakes.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from controllers.open.submission.kitchen import Layout, Tile

PLATE = "plate"
TAKE = "take"
PUT = "put"
PRESS = "press"
GRAB = "grab"
DELIVER = "deliver"
BIN = "bin"


@dataclass(frozen=True)
class Macro:
    kind: str
    target: Tile
    sid: str | None = None
    food: str | None = None  # the stored food taken, or the station's output


# One of my items in a station: (station id, tile, output, ready step, presses left)
Mine = tuple[str, Tile, str, int, int]
# A usable station: (station id, tile, kind, cook time, cooks by itself, free from step)
FreeStation = tuple[str, Tile, str, int, bool, int]


@dataclass
class Plan:
    finish: int  # the step the dish is delivered
    macros: list[Macro]
    expanded: int


def plan_dish(
    layout: Layout,
    dish: str,
    now: int,
    pos: Tile,
    has_plate: bool,
    content: str | None,
    mine: tuple[Mine, ...],
    free: list[FreeStation],
    *,
    parked: tuple[tuple[Tile, str | None], ...] = (),
    max_nodes: int = 2500,
    per_kind: int = 2,
) -> Plan | None:
    """Find the quickest way to deliver ``dish``.

    ``parked`` lists plates this chef left on counters, with what they hold;
    a chef with empty hands may pick one up instead of a fresh plate.
    """
    tree = layout.tree(dish)
    takeable = [food for food in tree if food in layout.storages]
    kinds = layout.station_kinds
    cook = layout.cook
    add = layout.add

    def reach(target: Tile, at: Tile) -> tuple[int, Tile] | None:
        return layout.reach(target).get(at)

    def nearest(tiles: list[Tile], at: Tile) -> tuple[int, Tile, Tile] | None:
        best = None
        for tile in tiles:
            r = reach(tile, at)
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], r[1], tile)
        return best

    free_by_kind: dict[str, list[FreeStation]] = {}
    for station in free:
        free_by_kind.setdefault(station[2], []).append(station)

    start = (pos, has_plate, content, tuple(sorted(mine)))
    best_time: dict[tuple, int] = {start: now}
    parent: dict[tuple, tuple[tuple, Macro] | None] = {start: None}
    heap: list[tuple[int, int, tuple]] = [(now, 0, start)]
    counter = 0
    expanded = 0
    goal_key = None
    goal_time = 0

    def push(t: int, key: tuple, prev: tuple, macro: Macro) -> None:
        nonlocal counter
        old = best_time.get(key)
        if old is not None and old <= t:
            return
        best_time[key] = t
        parent[key] = (prev, macro)
        counter += 1
        heapq.heappush(heap, (t, counter, key))

    while heap:
        t, _, key = heapq.heappop(heap)
        if best_time.get(key, 1 << 30) < t:
            continue
        if key[0] == "GOAL":
            goal_key, goal_time = key, t
            break
        expanded += 1
        if expanded > max_nodes:
            return None
        at, plate, food, stations = key

        if not plate:
            got = nearest(layout.dispensers, at)
            if got is not None:
                d, tile, target = got
                push(t + d + 1, (tile, True, None, stations), key, Macro(PLATE, target))
            for spot, held in parked:
                if held is not None and held not in tree:
                    continue
                r = reach(spot, at)
                if r is not None:
                    push(
                        t + r[0] + 1,
                        (r[1], True, held, stations),
                        key,
                        Macro(PLATE, spot, food=held),
                    )
            continue

        pressing = next((s for s in stations if s[4] > 0), None)
        if pressing is not None:
            r = reach(pressing[1], at)
            if r is not None:
                done = t + r[0] + pressing[4]
                rest = tuple(
                    sorted(
                        (s if s is not pressing else (s[0], s[1], s[2], done, 0))
                        for s in stations
                    )
                )
                push(
                    done,
                    (r[1], True, food, rest),
                    key,
                    Macro(PRESS, pressing[1], pressing[0], pressing[2]),
                )
            continue

        if food is not None and food not in tree:
            got = nearest(layout.bins, at)
            if got is not None:
                d, tile, target = got
                push(t + d + 1, (tile, True, None, stations), key, Macro(BIN, target))
            continue

        if food == dish:
            got = nearest(layout.deliveries, at)
            if got is not None:
                d, tile, target = got
                push(t + d + 1, ("GOAL",), key, Macro(DELIVER, target, food=dish))
            continue

        outs = {s[2] for s in stations}
        used = {s[0] for s in stations}

        # Take a stored ingredient onto the plate.
        for raw in takeable:
            new = add(food, raw)
            if new is None or new not in tree or new in outs:
                continue
            got = nearest(layout.storages[raw], at)
            if got is None:
                continue
            d, tile, target = got
            push(
                t + d + 1,
                (tile, True, new, stations),
                key,
                Macro(TAKE, target, food=raw),
            )

        # Take something I left in a station back onto the plate.
        for station in stations:
            new = add(food, station[2])
            if new is None or new not in tree:
                continue
            r = reach(station[1], at)
            if r is None:
                continue
            done = max(t + r[0], station[3]) + 1
            rest = tuple(s for s in stations if s is not station)
            push(
                done,
                (r[1], True, new, rest),
                key,
                Macro(GRAB, station[1], station[0], station[2]),
            )

        # Put the plate's food into a free station to cook.
        if food is not None:
            for kind in kinds:
                out = cook.get((food, kind))
                if out is None or out not in tree or out in outs:
                    continue
                options = []
                for station in free_by_kind.get(kind, ()):
                    if station[0] in used:
                        continue
                    r = reach(station[1], at)
                    if r is not None:
                        options.append((max(t + r[0], station[5]), r[1], station))
                options.sort(key=lambda o: o[0])
                for step, tile, station in options[:per_kind]:
                    if station[4]:
                        item = (station[0], station[1], out, step + station[3], 0)
                    else:
                        item = (
                            station[0],
                            station[1],
                            out,
                            step + station[3] + 1,
                            station[3],
                        )
                    rest = tuple(sorted(stations + (item,)))
                    push(
                        step + 1,
                        (tile, True, None, rest),
                        key,
                        Macro(PUT, station[1], station[0], out),
                    )

    if goal_key is None:
        return None
    macros: list[Macro] = []
    key = goal_key
    while True:
        link = parent.get(key)
        if link is None:
            break
        key, macro = link
        macros.append(macro)
    macros.reverse()
    return Plan(finish=goal_time - 1, macros=macros, expanded=expanded)
