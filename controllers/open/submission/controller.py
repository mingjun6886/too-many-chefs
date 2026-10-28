"""Level 4 controller: one chef in a 2v2 Open Kitchen match.

Every tick the controller reads the kitchen (``kitchen.read_snapshot``), keeps
or picks a dish to cook, plans that dish from the current state
(``planner.plan_dish``) and carries out the first step of the plan: walk one
tile towards the next station, or face it and act. Replanning every tick makes
the chef react at once when a teammate or an opponent takes an order, a
station or the food in it.

Choosing a dish: each visible order the team is not already making is planned,
and the chef takes the one with the best reward per step that can be delivered
before its ticket expires. The teammate's dish is read from what it carries.

Any exception inside ``get_actions`` is caught, and the chef then waits for one
tick, since a crash would lose the whole set.
"""

from __future__ import annotations

import heapq
import os
import random
import time
from collections import Counter as Tally

from controllers.open.submission.kitchen import (
    FACING,
    Chef,
    Layout,
    Snapshot,
    Tile,
    read_snapshot,
)
from controllers.open.submission.planner import (
    BIN,
    DELIVER,
    GRAB,
    PLATE,
    PRESS,
    PUT,
    TAKE,
    Macro,
    Mine,
    plan_dish,
)
from simulator.context import Context
from simulator.controller import Controller
from simulator.entities import Agent, Equipment, Food, Plate
from simulator.environment import Environment
from simulator.mutations import (
    Combine,
    Cook,
    Deliver,
    Discard,
    MoveAgent,
    Mutation,
    PickUp,
    Place,
    TakeFromStorage,
    TurnAgent,
)
from simulator.mutations.mutation import IllegalMutationError


def _knob(name: str, default: float) -> float:
    """A tuning value, overridable from the environment for local experiments."""
    try:
        value = float(os.environ.get(name, default))
    except TypeError, ValueError:
        return default
    return value if -1e6 < value < 1e6 else default


LOG_PATH = os.environ.get("OPEN_CHEF_LOG")
TRACE = bool(os.environ.get("OPEN_CHEF_TRACE"))
# Stop evaluating extra dishes once a tick has used this much time.
SOFT_BUDGET = 0.09
# Games last 300 steps; the last action lands in step 299.
LAST_STEP = 299
# How many of the orders still to be revealed are worth cooking ahead. The
# kitchen snapshot never predicts unrevealed orders, so this has no effect.
UPCOMING = int(_knob("OPEN_CHEF_UPCOMING", 4))
# Value kept by an order an opponent is already making, and by one not yet shown.
RACED = _knob("OPEN_CHEF_RACED", 0.5)
AHEAD = _knob("OPEN_CHEF_AHEAD", 0.9)
# Steps after which food left in a station counts as abandoned.
ABANDONED = int(_knob("OPEN_CHEF_ABANDONED", 12))
# The spare pot is fetched early by a chef this close to it; after this step
# any chef with free hands may fetch it.
SPARE_REACH = 6
SPARE_FALLBACK = 15
SPARE_UNTIL = 40
# Both measured worse over 8 order seeds against v1: a second pot and cleared
# stations help the other team as much as this one. Kept switchable.
SPARE_EARLY = _knob("OPEN_CHEF_SPARE", 0) == 1
CLEAN_JUNK = _knob("OPEN_CHEF_JUNK", 0) == 1
# Leftovers this close are collected and binned.
JUNK_REACH = 10
# Steps another chef's parked dish may wait, with its order up, before it is
# taken to be abandoned.
STALE_PARKED = 15
# Most plates one chef leaves on counters.
MAX_PARKED = 3
PLATE_NAME = "catalog/equipment/plate"


class _ChefState:
    """What one chef remembers between ticks."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.job: str | None = None
        # Stations holding my food: id -> (food put in, food it cooks into).
        self.mine: dict[str, tuple[str, str]] = {}
        # Puts sent last tick, checked against my plate this tick.
        self.unconfirmed: dict[str, str] = {}
        self.last_tile: Tile | None = None
        self.still = 0
        self.rng = random.Random(agent_id)
        # Plates I left on counters: tile -> food on them.
        self.parked: dict[Tile, str | None] = {}
        # Whether this chef has already moved a spare pot.
        self.spare_done = False


class OpenController(Controller):
    """Control one chef in a 2v2 match."""

    def __init__(self) -> None:
        self._layout: Layout | None = None
        self._chefs: dict[str, _ChefState] = {}
        self._times: list[float] = []
        self._log: list[str] = []
        self._deadline = 0.0
        # What each teammate seems to be cooking, and what it held last tick.
        self._mate_guess: dict[str, str | None] = {}
        self._mate_item: dict[str, str | None] = {}
        self._station_since: dict[str, tuple[str | None, int]] = {}
        self._stations_seen_at = -1
        # Others' parked dishes: (tile, dish) -> step first seen with its order up.
        self._others_parked: dict[tuple[Tile, str], int] = {}

    # Controller API ------------------------------------------------------

    def get_actions(self, environment: Environment, context: Context) -> list[Mutation]:
        started = time.perf_counter()
        self._deadline = started + SOFT_BUDGET
        actions: list[Mutation] = []
        try:
            agents = self.controllable_agents(environment)
        except Exception:
            return []
        for agent in agents:
            chosen: list[Mutation] = []
            try:
                chosen = self._act(environment, agent)
            except Exception as error:  # noqa: BLE001 - a crash loses the set
                self._note(
                    f"t={environment.timestep} error {type(error).__name__}: {error}"
                )
                chosen = []
            if TRACE:
                self._note(
                    "  -> "
                    + ",".join(
                        type(m).__name__ for m in chosen if not isinstance(m, TurnAgent)
                    )
                )
            if not chosen:
                chosen = [TurnAgent(agent_id=agent.id, orientation=agent.orientation)]
            actions.extend(chosen)
        self._times.append(time.perf_counter() - started)
        return actions

    def is_busy(self) -> bool:
        return False

    def has_finished(self, environment: Environment) -> bool:
        return False

    def shutdown(self) -> None:
        if not LOG_PATH:
            return
        try:
            times = sorted(self._times)
            if times:

                def pct(p: float) -> float:
                    return times[min(len(times) - 1, int(p * len(times)))] * 1000

                self._log.append(
                    f"ticks={len(times)} median={pct(0.5):.2f}ms p99={pct(0.99):.2f}ms"
                    f" max={times[-1] * 1000:.2f}ms"
                )
            with open(LOG_PATH, "a") as handle:
                handle.write("\n".join(self._log) + "\n")
        except Exception:
            pass

    # One chef --------------------------------------------------------------

    def _note(self, line: str) -> None:
        if LOG_PATH and len(self._log) < 5000:
            self._log.append(line)

    def _act(self, environment: Environment, agent: Agent) -> list[Mutation]:
        if self._layout is None:
            self._layout = Layout(environment)
        layout = self._layout
        snap = read_snapshot(environment, layout)
        me = snap.chefs.get(agent.id)
        if me is None:
            return []
        state = self._chefs.setdefault(agent.id, _ChefState(agent.id))
        if state.last_tile == me.tile:
            state.still += 1
        else:
            state.still = 0
        state.last_tile = me.tile

        self._refresh_mine(state, snap, me)
        self._refresh_parked(state, snap)
        self._track_stations(snap)

        # Hands holding something the plan cannot use: clear them first.
        if me.held_kind == "food":
            return self._go_and(
                environment, snap, me, state, layout.bins, Discard(agent_id=me.id)
            )
        if me.held_kind == "equipment":
            state.spare_done = True
            return self._put_back_equipment(environment, snap, me, state)
        if me.held_kind == "other":
            return []

        has_plate = me.held_kind == "plate"
        content = me.plate_content if has_plate else None
        mine = self._mine_tuple(state, snap)
        free = self._free_stations(state, snap)
        parked = tuple(state.parked.items())

        if (
            SPARE_EARLY
            and me.held_kind is None
            and not state.spare_done
            and snap.timestep < SPARE_UNTIL
        ):
            task = self._spare_equipment_task(environment, snap, me, state, early=True)
            if task is not None:
                return task
        if CLEAN_JUNK and has_plate and content is None:
            task = self._junk_task(environment, snap, me, state, mine)
            if task is not None:
                return task

        choice = self._choose_job(
            snap, me, state, has_plate, content, mine, free, parked
        )
        if choice is None:
            if has_plate and content is not None:
                if self._worth_keeping(snap, state, content):
                    return self._park(environment, snap, me, state, content)
                return self._go_and(
                    environment, snap, me, state, layout.bins, Discard(agent_id=me.id)
                )
            return self._idle(environment, snap, me, state)
        dish, deadline, mode = choice

        if has_plate:
            visible = {name for name, _, _ in snap.orders}
            if content == dish and dish not in visible:
                # Made ahead of its order: leave it by the hatch until it shows.
                return self._park(
                    environment, snap, me, state, content, near_hatch=True
                )
            if mode == "stock" and content != dish:
                return self._park(environment, snap, me, state, content)
            if (
                content is not None
                and content not in layout.tree(dish)
                and self._worth_keeping(snap, state, content)
            ):
                return self._park(environment, snap, me, state, content)

        plan = plan_dish(
            layout,
            dish,
            snap.timestep,
            me.tile,
            has_plate,
            content,
            mine,
            free,
            parked=parked,
        )
        if plan is None or not plan.macros:
            state.job = None
            return self._idle(environment, snap, me, state)
        macro = plan.macros[0]
        if TRACE:
            self._note(
                f"t={snap.timestep} {me.id}@{me.tile} {dish[13:]}/{mode} c={content}"
                f" fin={plan.finish} n={plan.expanded} {macro.kind}:{macro.target}:"
                f"{macro.food} mine={[(m[2][13:], m[3], m[4]) for m in mine]}"
                f" parked={parked}"
            )
        return self._execute(environment, snap, me, state, macro, content)

    # Bookkeeping -------------------------------------------------------------

    def _refresh_mine(self, state: _ChefState, snap: Snapshot, me: Chef) -> None:
        """Drop stations that no longer hold my food.

        A put is only trusted once my plate no longer holds what I put: another
        chef may have filled the station first in the same tick.
        """
        for sid, food_in in list(state.unconfirmed.items()):
            del state.unconfirmed[sid]
            if me.held_kind == "plate" and me.plate_content == food_in:
                state.mine.pop(sid, None)
        for sid, (food_in, food_out) in list(state.mine.items()):
            station = snap.stations.get(sid)
            if station is None or station.food not in (food_in, food_out):
                del state.mine[sid]

    def _mine_tuple(self, state: _ChefState, snap: Snapshot) -> tuple[Mine, ...]:
        now = snap.timestep
        items: list[Mine] = []
        for sid, (food_in, food_out) in state.mine.items():
            station = snap.stations[sid]
            if station.food == food_out:
                items.append((sid, station.tile, food_out, now, 0))
            elif station.self_cook:
                left = max(0, station.cook_time - station.elapsed)
                items.append((sid, station.tile, food_out, now + left, 0))
            else:
                left = max(1, station.cook_time - station.elapsed)
                items.append((sid, station.tile, food_out, now + left, left))
        return tuple(items)

    def _track_stations(self, snap: Snapshot) -> None:
        """Remember since when each station has held what it holds now."""
        if self._stations_seen_at == snap.timestep:
            return
        self._stations_seen_at = snap.timestep
        for station in snap.stations.values():
            seen = self._station_since.get(station.id)
            if seen is None or seen[0] != station.food_id:
                self._station_since[station.id] = (station.food_id, snap.timestep)

    def _free_stations(self, state: _ChefState, snap: Snapshot) -> list:
        """Stations the plan may use, each with the step it should be free from.

        A station busy with someone else's food is expected to be emptied a few
        steps after that food is done; one that has sat untouched for a long
        time is treated as abandoned and left out.
        """
        layout = self._layout
        assert layout is not None
        now = snap.timestep
        free = []
        for station in snap.stations.values():
            if not station.supported or station.id in state.mine:
                continue
            free_at = now
            if station.food is not None:
                age = now - self._station_since.get(station.id, (None, now))[1]
                cooks_on = layout.cook.get((station.food, station.name))
                tended = station.self_cook or any(
                    chef.tile in layout.access(station.tile)
                    and _ahead(chef.tile, chef.orientation) == station.tile
                    for chef in snap.chefs.values()
                )
                if cooks_on is not None and tended:
                    left = max(1, station.cook_time - station.elapsed)
                    free_at = now + left + 4
                elif age > ABANDONED:
                    continue
                else:
                    # Done food waiting to be collected: the longer it waits,
                    # the less likely anyone comes for it soon.
                    free_at = now + 4 + age
            free.append(
                (
                    station.id,
                    station.tile,
                    station.name,
                    station.cook_time,
                    station.self_cook,
                    free_at,
                )
            )
        return free

    # Choosing a dish -----------------------------------------------------------

    def _held_item(self, chef: Chef) -> str | None:
        if chef.held_kind == "plate":
            return chef.plate_content
        if chef.held_kind == "food":
            return chef.held_name
        return None

    def _infer(self, snap: Snapshot, chef: Chef, wanted: dict[str, int]) -> str | None:
        """Guess which wanted dish ``chef`` is making from what it holds.

        ``wanted`` maps each dish on order or coming soon to its reward. The
        guess is remembered, so an ambiguous item (meat goes into both burgers)
        keeps pointing at the dish first inferred, and it is dropped once the
        chef's hands empty after carrying a finished dish.
        """
        layout = self._layout
        assert layout is not None
        item = self._held_item(chef)
        previous = self._mate_guess.get(chef.id)
        last_item = self._mate_item.get(chef.id)
        self._mate_item[chef.id] = item
        if previous is not None:
            delivered = item is None and last_item == previous
            if (
                not delivered
                and previous in wanted
                and (item is None or item in layout.tree(previous))
            ):
                return previous
        guess = None
        if item is not None:
            if item in wanted:
                guess = item
            else:
                best = None
                for name, reward in wanted.items():
                    if item in layout.tree(name) and (best is None or reward > best[1]):
                        best = (name, reward)
                guess = best[0] if best else None
        self._mate_guess[chef.id] = guess
        return guess

    def _progress(self, content: str | None, mine: tuple[Mine, ...], dish: str) -> int:
        layout = self._layout
        assert layout is not None
        tree = layout.tree(dish)
        total = layout.leaves(content) if content in tree else 0
        total += sum(layout.leaves(item[2]) for item in mine if item[2] in tree)
        return total

    def _choose_job(
        self,
        snap: Snapshot,
        me: Chef,
        state: _ChefState,
        has_plate: bool,
        content: str | None,
        mine: tuple[Mine, ...],
        free: list,
        parked: tuple,
    ) -> tuple[str, int, str] | None:
        """Pick the dish to work on: (dish, deadline, mode).

        Mode ``stock`` delivers a dish already made and parked on a counter;
        ``make`` cooks one. Demand is the visible orders plus the next few the
        generator will reveal; supply is parked dishes and what the teammate
        is making. Of the demand left over, the dish with the best reward per
        step until it can be paid is chosen.
        """
        layout = self._layout
        assert layout is not None
        now = snap.timestep
        visible = snap.orders
        if not visible and not snap.upcoming:
            state.job = None
            return None

        # A parked dish that is now on order goes out first.
        stock: Tally[str] = Tally(
            held for _, held in parked if held is not None and held in snap.menu
        )
        for name, _, deadline in visible:
            if content == name:
                return self._set_job(state, snap, me, name, "make", deadline)
        for name, _, deadline in visible:
            if stock[name] > 0:
                return self._set_job(state, snap, me, name, "stock", deadline)

        # Demand: visible orders, then the next ones to be revealed.
        demands: list[tuple[str, int, int, int, bool]] = []
        for name, reward, deadline in visible:
            demands.append((name, reward, min(deadline, LAST_STEP), now, True))
        gap = 20 if snap.events < 2 else max(8, now // snap.events)
        expiries = sorted(deadline + 1 for _, _, deadline in visible)
        for k, (name, reward, limit) in enumerate(snap.upcoming[:UPCOMING]):
            reveal = now + (k + 1) * gap
            if k < len(expiries):
                reveal = min(reveal, expiries[k])
            demands.append(
                (name, reward, min(reveal + limit, LAST_STEP), reveal, False)
            )
        wanted: dict[str, int] = {}
        for name, reward, *_ in demands:
            wanted[name] = max(wanted.get(name, 0), reward)

        # Supply: parked dishes (mine, and others' unless left unserved while
        # their order is up) and what the teammate seems to be making.
        supply: Tally[str] = Tally(stock)
        visible_names = {name for name, _, _ in visible}
        for tile, (_, held) in snap.counter_plates.items():
            if tile in state.parked or held is None or held not in snap.menu:
                continue
            key = (tile, held)
            if held in visible_names:
                since = self._others_parked.setdefault(key, now)
                if now - since > STALE_PARKED:
                    continue
            supply[held] += 1
        teammates = [
            chef
            for chef in snap.chefs.values()
            if chef.team == me.team and chef.id != me.id
        ]
        for mate in teammates:
            guess = self._infer(snap, mate, wanted)
            if guess is None:
                continue
            if guess == state.job:
                # Both on one dish: the one further behind gives way.
                item = self._held_item(mate)
                mate_progress = layout.leaves(item) if item else 0
                my_progress = self._progress(content, mine, guess)
                if my_progress > mate_progress or (
                    my_progress == mate_progress and me.id < mate.id
                ):
                    continue
            supply[guess] += 1
        # Opponents racing for a visible order make it worth less.
        racing: Tally[str] = Tally()
        for chef in snap.chefs.values():
            if chef.team == me.team:
                continue
            item = self._held_item(chef)
            if item is None:
                continue
            for name, _, _ in visible:
                if item == name or item in layout.tree(name):
                    racing[name] += 1
                    break

        if now < 2 and state.job is None:
            # First decision: a teammate with a lower id is taken to be a copy
            # of this controller that picks first.
            for mate in teammates:
                if mate.id < me.id and mate.held_kind is None:
                    pick = self._best_dish(
                        snap,
                        mate.tile,
                        False,
                        None,
                        (),
                        free,
                        (),
                        demands,
                        supply,
                        racing,
                        None,
                    )
                    if pick is not None:
                        supply[pick[0]] += 1

        best = self._best_dish(
            snap,
            me.tile,
            has_plate,
            content,
            mine,
            free,
            parked,
            demands,
            supply,
            racing,
            state.job,
        )
        if best is None:
            return self._set_job(state, snap, me, None, "make", 0)
        return self._set_job(state, snap, me, best[0], "make", best[1])

    def _set_job(
        self,
        state: _ChefState,
        snap: Snapshot,
        me: Chef,
        dish: str | None,
        mode: str,
        deadline: int,
    ) -> tuple[str, int, str] | None:
        if dish != state.job:
            self._note(
                f"t={snap.timestep} {me.id} job {state.job} -> {dish} ({mode})"
                f" orders={[o[0][13:] for o in snap.orders]}"
                f" next={[u[0][13:] for u in snap.upcoming[:UPCOMING]]}"
            )
        state.job = dish
        if dish is None:
            return None
        return (dish, deadline, mode)

    def _best_dish(
        self,
        snap: Snapshot,
        tile: Tile,
        has_plate: bool,
        content: str | None,
        mine: tuple[Mine, ...],
        free: list,
        parked: tuple,
        demands: list[tuple[str, int, int, int, bool]],
        supply: Tally,
        racing: Tally,
        current: str | None,
    ) -> tuple[str, int] | None:
        """The open demand with the best reward per step: (dish, deadline)."""
        layout = self._layout
        assert layout is not None
        now = snap.timestep
        left = Tally(supply)
        open_demands = []
        seen: set[str] = set()
        visible_count = Tally(name for name, *_ in demands if _[-1])
        for demand in demands:
            name = demand[0]
            if left[name] > 0:
                left[name] -= 1
                continue
            if name in seen:
                continue
            seen.add(name)
            open_demands.append(demand)
        if current is not None:
            # Keep the current dish while it is still wanted and on time.
            for name, reward, deadline, ready, is_visible in open_demands:
                if name != current:
                    continue
                plan = plan_dish(
                    layout,
                    name,
                    now,
                    tile,
                    has_plate,
                    content,
                    mine,
                    free,
                    parked=parked,
                )
                if plan is not None and plan.finish <= deadline:
                    return (name, deadline)
        best: tuple[float, str, int] | None = None
        for name, reward, deadline, ready, is_visible in open_demands:
            if time.perf_counter() > self._deadline and best is not None:
                break
            plan = plan_dish(
                layout, name, now, tile, has_plate, content, mine, free, parked=parked
            )
            if plan is None or plan.finish > deadline:
                continue
            paid = max(plan.finish, ready)
            value = reward
            if is_visible:
                value += 0.5 * len(plan.macros)
                if racing[name] >= visible_count[name]:
                    value *= RACED
            else:
                value *= AHEAD
            rate = value / max(1, paid - now + 1)
            if best is None or rate > best[0]:
                best = (rate, name, deadline)
        return (best[1], best[2]) if best else None

    # Parking plates ------------------------------------------------------------

    def _refresh_parked(self, state: _ChefState, snap: Snapshot) -> None:
        for tile, held in list(state.parked.items()):
            seen = snap.counter_plates.get(tile)
            if seen is None or seen[1] != held:
                del state.parked[tile]

    def _worth_keeping(self, snap: Snapshot, state: _ChefState, content: str) -> bool:
        """Whether food on my plate is worth parking rather than binning."""
        layout = self._layout
        assert layout is not None
        if len(state.parked) >= MAX_PARKED:
            return False
        wanted = {name for name, _, _ in snap.orders} | {
            name for name, _, _ in snap.upcoming[:UPCOMING]
        }
        if content in snap.menu:
            return content in wanted
        return layout.leaves(content) >= 2 and any(
            content in layout.tree(name) for name in wanted
        )

    def _park(
        self,
        environment: Environment,
        snap: Snapshot,
        me: Chef,
        state: _ChefState,
        content: str | None,
        *,
        near_hatch: bool = False,
    ) -> list[Mutation]:
        """Set my plate down on a free counter (by the hatch for a finished dish)."""
        layout = self._layout
        assert layout is not None
        hatch = [a for d in layout.deliveries for a in layout.access(d)]
        best = None
        for counter in snap.free_counters:
            to_me = layout.reach(counter).get(me.tile)
            if to_me is None:
                continue
            score = float(to_me[0])
            if near_hatch:
                to_hatch = min(
                    (
                        layout.dist[a].get(h, 99)
                        for a in layout.access(counter)
                        for h in hatch
                    ),
                    default=99,
                )
                score = to_hatch * 3 + to_me[0]
            if best is None or score < best[0]:
                best = (score, counter)
        if best is None:
            if content is None:
                return []
            return self._go_and(
                environment, snap, me, state, layout.bins, Discard(agent_id=me.id)
            )
        counter = best[1]
        if counter in _neighbours_of(me.tile):
            facing = FACING[(counter[0] - me.tile[0], counter[1] - me.tile[1])]
            turn = TurnAgent(agent_id=me.id, orientation=facing)
            place = Place(agent_id=me.id, expected_held_equipment_name=PLATE_NAME)
            if _simulate(environment, turn, place) is not None:
                state.parked[counter] = content
                if TRACE:
                    self._note(
                        f"t={snap.timestep} {me.id} parks {content} at {counter}"
                    )
                return [turn, place] if me.orientation != facing else [place]
            return []
        return self._walk(snap, me, state, layout.access(counter))

    # Acting -----------------------------------------------------------------------

    def _idle(
        self, environment: Environment, snap: Snapshot, me: Chef, state: _ChefState
    ) -> list[Mutation]:
        if me.held_kind is None:
            task = self._spare_equipment_task(environment, snap, me, state)
            if task is not None:
                return task
        return []

    def _execute(
        self,
        environment: Environment,
        snap: Snapshot,
        me: Chef,
        state: _ChefState,
        macro: Macro,
        content: str | None,
    ) -> list[Mutation]:
        layout = self._layout
        assert layout is not None
        target = macro.target
        if target not in _neighbours_of(me.tile):
            return self._walk(snap, me, state, layout.access(target))

        facing = FACING[(target[0] - me.tile[0], target[1] - me.tile[1])]
        turn = TurnAgent(agent_id=me.id, orientation=facing)
        station = snap.stations.get(macro.sid) if macro.sid else None
        mutation: Mutation
        if macro.kind == PLATE:
            if target in layout.dispensers:
                mutation = TakeFromStorage(agent_id=me.id)
            else:
                mutation = PickUp(
                    agent_id=me.id,
                    expected_held_equipment_name=PLATE_NAME,
                    expected_held_equipment_contents_name=macro.food,
                )
        elif macro.kind == TAKE:
            mutation = TakeFromStorage(agent_id=me.id, expected_input_name=macro.food)
        elif macro.kind == PUT:
            if station is None:
                return [turn]
            mutation = Combine(
                agent_id=me.id, expected_target_equipment_name=station.name
            )
        elif macro.kind == PRESS:
            if station is None or station.food is None:
                return [turn]
            mutation = Cook(agent_id=me.id, expected_held_food=station.food)
        elif macro.kind == GRAB:
            if station is None:
                return [turn]
            mutation = Combine(
                agent_id=me.id,
                expected_target_equipment_name=station.name,
                expected_target_equipment_contents_name=macro.food,
            )
        elif macro.kind == DELIVER:
            mutation = Deliver(agent_id=me.id)
        elif macro.kind == BIN:
            mutation = Discard(agent_id=me.id)
        else:
            return []

        after = _simulate(environment, turn, mutation)
        if after is None or not self._outcome_ok(after, me, macro, content, station):
            # Not possible yet (still cooking, say): face the target and wait.
            return [turn] if me.orientation != facing else []

        if macro.kind == PUT and station is not None and content is not None:
            state.mine[station.id] = (content, macro.food or content)
            state.unconfirmed[station.id] = content
        if macro.kind == GRAB and macro.sid:
            state.mine.pop(macro.sid, None)
        result: list[Mutation] = [mutation]
        if me.orientation != facing:
            result.insert(0, turn)
        return result

    def _outcome_ok(
        self, after: Environment, me: Chef, macro: Macro, content, station
    ) -> bool:
        layout = self._layout
        assert layout is not None
        agent = after.get_entity_as(me.id, Agent)
        if agent is None:
            return False
        held = after.get_entity(agent.held_item_id) if agent.held_item_id else None
        plate_food = None
        if isinstance(held, Plate) and held.held_item_id:
            food = after.get_entity_as(held.held_item_id, Food)
            plate_food = food.name if food else None
        if macro.kind == PLATE:
            return isinstance(held, Plate) and plate_food == macro.food
        if macro.kind == TAKE:
            return isinstance(held, Plate) and plate_food == layout.add(
                content, macro.food or ""
            )
        if macro.kind == PUT:
            if not isinstance(held, Plate) or plate_food is not None or station is None:
                return False
            eq = after.get_entity_as(station.id, Equipment)
            food = (
                after.get_entity_as(eq.held_item_id, Food)
                if eq and eq.held_item_id
                else None
            )
            return food is not None and food.name == content
        if macro.kind == GRAB:
            return isinstance(held, Plate) and plate_food == layout.add(
                content, macro.food or ""
            )
        if macro.kind == BIN:
            return isinstance(held, Plate) and plate_food is None
        return True

    def _go_and(
        self,
        environment: Environment,
        snap: Snapshot,
        me: Chef,
        state: _ChefState,
        targets: list[Tile],
        mutation: Mutation,
    ) -> list[Mutation]:
        layout = self._layout
        assert layout is not None
        for target in targets:
            if target in _neighbours_of(me.tile):
                facing = FACING[(target[0] - me.tile[0], target[1] - me.tile[1])]
                turn = TurnAgent(agent_id=me.id, orientation=facing)
                if _simulate(environment, turn, mutation) is not None:
                    return [turn, mutation] if me.orientation != facing else [mutation]
        goals = [tile for target in targets for tile in layout.access(target)]
        return self._walk(snap, me, state, goals)

    def _put_back_equipment(
        self, environment: Environment, snap: Snapshot, me: Chef, state: _ChefState
    ) -> list[Mutation]:
        """Set down a pan, pot or bowl this chef ended up holding.

        It goes on a free support it needs (a pot on a bare stove), or else on
        an empty counter.
        """
        held = environment.get_entity_as(me.held_id or "", Equipment)
        targets = _free_supports(environment, held) if held is not None else []
        if not targets:
            targets = list(snap.free_counters)
        return self._go_and(
            environment, snap, me, state, targets, Place(agent_id=me.id)
        )

    def _spare_equipment_task(
        self,
        environment: Environment,
        snap: Snapshot,
        me: Chef,
        state: _ChefState,
        *,
        early: bool = False,
    ) -> list[Mutation] | None:
        """Lift an empty pot or pan that is off its stove, to set it on a free one.

        The kitchen has one pot on a stove and a spare one on each team's
        bench; a second pot doubles how much rice can cook. ``early`` is the
        proactive version: only the teammate nearest the spare goes for it.
        """
        layout = self._layout
        assert layout is not None
        best = None
        for station in snap.stations.values():
            if station.supported or station.food is not None:
                continue
            eq = environment.get_entity_as(station.id, Equipment)
            if eq is None or not eq.can_pick_up or not eq.requires:
                continue
            if station.tile not in layout.counters:
                continue
            if not _free_supports(environment, eq):
                continue
            reach = layout.reach(station.tile)
            mine = reach.get(me.tile)
            if mine is None:
                continue
            if early:
                if mine[0] > SPARE_REACH:
                    continue
                if snap.timestep < SPARE_FALLBACK and any(
                    chef.team == me.team
                    and chef.id != me.id
                    and chef.held_kind is None
                    and (reach.get(chef.tile, (999,))[0], chef.id) < (mine[0], me.id)
                    for chef in snap.chefs.values()
                ):
                    continue
            if best is None or mine[0] < best[0]:
                best = (mine[0], station.tile, eq.name)
        if best is None:
            return None
        return self._go_and(
            environment,
            snap,
            me,
            state,
            [best[1]],
            PickUp(agent_id=me.id, expected_held_equipment_name=best[2]),
        )

    def _junk_task(
        self,
        environment: Environment,
        snap: Snapshot,
        me: Chef,
        state: _ChefState,
        mine: tuple[Mine, ...],
    ) -> list[Mutation] | None:
        """Collect my leftover food that no wanted dish needs, to bin it.

        Leftovers from abandoned dishes would otherwise block a pan, pot or
        cutboard for the rest of the game.
        """
        layout = self._layout
        assert layout is not None
        wanted = [name for name, _, _ in snap.orders] + [
            name for name, _, _ in snap.upcoming[:UPCOMING]
        ]
        if state.job:
            wanted.append(state.job)
        best = None
        for sid, tile, out, ready, presses in mine:
            if presses or ready > snap.timestep:
                continue
            if any(out in layout.tree(name) for name in wanted):
                continue
            r = layout.reach(tile).get(me.tile)
            if r is None or r[0] > JUNK_REACH:
                continue
            if best is None or r[0] < best[0]:
                best = (r[0], Macro(GRAB, tile, sid, out))
        if best is None:
            return None
        if TRACE:
            self._note(
                f"t={snap.timestep} {me.id} clears {best[1].food} at {best[1].target}"
            )
        return self._execute(environment, snap, me, state, best[1], None)

    def _walk(
        self, snap: Snapshot, me: Chef, state: _ChefState, goals: list[Tile]
    ) -> list[Mutation]:
        """Take one step towards the nearest goal tile.

        Tiles with a chef on them cost extra rather than being walls, so a chef
        waits a tick for someone to pass instead of taking a long detour; the
        longer it has been stuck, the more a detour is preferred.
        """
        layout = self._layout
        assert layout is not None
        blocked = snap.occupied - {me.tile}
        penalty = 3 + 2 * state.still
        step = _first_step(layout.walk, me.tile, set(goals), blocked, penalty)
        if step is not None and step in blocked:
            step = None
        if step is None and state.still >= 3:
            options = [
                n
                for n in _neighbours_of(me.tile)
                if n in layout.walk and n not in blocked
            ]
            if options:
                step = state.rng.choice(options)
        if step is None or step == me.tile:
            return []
        return [
            MoveAgent(agent_id=me.id, dx=step[0] - me.tile[0], dy=step[1] - me.tile[1])
        ]


def _free_supports(environment: Environment, held: Equipment) -> list[Tile]:
    """Cells holding the station ``held`` needs, with nothing portable on them."""
    if not held.requires:
        return []
    supports: list[Tile] = []
    portable: set[Tile] = set()
    for eq in environment.get_entities_of_type(Equipment):
        if eq.x is None or eq.y is None or eq.id == held.id:
            continue
        if eq.can_pick_up:
            portable.add((eq.x, eq.y))
        if eq.name == held.requires:
            supports.append((eq.x, eq.y))
    return [tile for tile in supports if tile not in portable]


def _ahead(tile: Tile, orientation: str) -> Tile:
    dx, dy = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}[orientation]
    return (tile[0] + dx, tile[1] + dy)


def _neighbours_of(tile: Tile) -> list[Tile]:
    x, y = tile
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def _first_step(
    walk: frozenset[Tile],
    start: Tile,
    goals: set[Tile],
    blocked: set[Tile],
    penalty: int,
) -> Tile | None:
    """Cheapest path to any goal, where tiles in ``blocked`` cost ``penalty`` more.

    Return the first tile of that path (``start`` when already on a goal).
    """
    goals = {g for g in goals if g in walk}
    if not goals:
        return None
    if start in goals:
        return start
    best = {start: 0}
    first: dict[Tile, Tile] = {}
    heap: list[tuple[int, int, Tile]] = [(0, 0, start)]
    count = 0
    while heap:
        cost, _, tile = heapq.heappop(heap)
        if cost > best.get(tile, 1 << 30):
            continue
        if tile in goals:
            return first[tile]
        for n in _neighbours_of(tile):
            if n not in walk:
                continue
            step_cost = cost + 1 + (penalty if n in blocked else 0)
            if step_cost < best.get(n, 1 << 30):
                best[n] = step_cost
                first[n] = n if tile == start else first[tile]
                count += 1
                heapq.heappush(heap, (step_cost, count, n))
    return None


def _simulate(environment: Environment, *mutations: Mutation) -> Environment | None:
    try:
        for mutation in mutations:
            environment = mutation.run(environment)
        return environment
    except IllegalMutationError:
        return None
    except Exception:
        return None
