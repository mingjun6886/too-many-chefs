"""Decide what the next planning problem is about.

Planning every visible order at once makes problems that Fast Downward
cannot ground in time. Instead, Python picks the next few orders ("jobs"),
gives each job its own crew of chefs and, as far as it can, its own food,
plates and cookware, and leaves everything else out of the problem:

- Orders are ranked by the kitchen's rules: under strict ordering or
  sequential reveal only the head of the queue can be served, so it is the
  only job; otherwise the earliest deadline (then the highest reward) first.
- An order only becomes a job if the chefs can make it: ``_route`` walks the
  recipes back to storages and loose food the crew can reach, and the job
  needs a plate (clean, dirty with a sink, or from a dispenser) and every
  station on the route.
- In an open kitchen each order gets its own lead chef, the one with the
  shortest walk to the stations it needs, so the problem splits into
  nearly independent parts (one per job) and stays small; spare chefs
  help. Jobs share a station only when it is a watched one (a pan that
  burns what is left in it, which no plan leaves holding food); they may
  also share counters and ``Service.shared`` items (food in hands, junk in
  the way). In a
  divided kitchen every job needs the whole brigade, and one order is
  planned at a time (two at once made Fast Downward's plans worse).
- When no order passes these checks, one order is planned with every chef
  and every item before the service waits or stops, since the checks are
  stricter than the domain.
- When the previous plan stalled without moving the kitchen, ``attempt``
  rises and the next problem plans a single order with every chef, cycling
  through the orders, so one impossible order cannot block the others.
- When the kitchen keeps standing still, ``watch=False`` plans without the
  watch rules for food that burns (see ``problem_generator``'s escape).

The result is a ``Service``: the jobs, the items each job owns, the items
every job may touch (junk in the way, food in a chef's hands), and the tiles
worth describing (an open kitchen has far more counters than any plan uses).
"""

from __future__ import annotations

from collections import Counter as Tally
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field

from controllers.pddl.submission.kitchen import Kitchen, Order, Vessel, pddl_name
from simulator.entities import OvercookedState

# Orders planned together when chefs must cooperate on every dish.
JOINT_ORDERS = 1
# Empty counters described per job.
COUNTERS_PER_JOB = 3
# Counters described between two parts of a divided kitchen.
HANDOFF_COUNTERS = 3
# A station another job already uses counts as this many steps further, so
# jobs share one only when there is no other.
_BUSY_STEPS = 100
# A plate holding waste counts as this much worse than an empty one.
_EMPTYING_STEPS = 3
_MAX_DEPTH = 12
_INFINITY = 10**6


@dataclass
class Job:
    symbol: str
    order: Order
    crew: list[str]
    # Items (food and vessels) only this job may use.
    items: set[str] = field(default_factory=set)
    # Food a storage will hand out for this job: token symbol -> food name.
    tokens: dict[str, str] = field(default_factory=dict)
    # Whether the job keeps to its own workspace (an open kitchen, one chef
    # per order) rather than using every tile the problem describes.
    local: bool = False
    # The tiles the job may act on.
    slots: set[str] = field(default_factory=set)


@dataclass
class Service:
    jobs: list[Job] = field(default_factory=list)
    # Items every job may use: food in a chef's hands or in the way.
    shared: set[str] = field(default_factory=set)
    slots: set[str] = field(default_factory=set)
    # Food names the problem needs as kinds.
    kinds: set[str] = field(default_factory=set)
    # With no job: "wait" lets time pass (more orders are coming or will
    # expire), "stop" ends the service.
    rest: str | None = None


@dataclass
class _Route:
    cost: int
    pulls: tuple[str, ...] = ()
    items: frozenset[str] = frozenset()
    stations: frozenset[str] = frozenset()


class _Recipes:
    def __init__(self, state: OvercookedState) -> None:
        self.state = state
        self.cooked_from: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.combined_from: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for recipe in state.cook_recipes:
            self.cooked_from[recipe.output].append(
                (recipe.ingredient, recipe.equipment)
            )
        for recipe in state.combine_recipes:
            ingredients = sorted(recipe.ingredients)
            if len(ingredients) == 1:
                ingredients *= 2
            if len(ingredients) == 2:
                self.combined_from[recipe.output].append(
                    (ingredients[0], ingredients[1])
                )

    def needed(self, dishes: Collection[str]) -> set[str]:
        return needed(self.state, dishes)

    def ancestors(self, dish: str) -> set[str]:
        """Return every food that can go into making ``dish``, and the dish."""
        seen: set[str] = set()
        stack = [dish]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            for ingredient, _ in self.cooked_from.get(name, ()):
                stack.append(ingredient)
            for first, second in self.combined_from.get(name, ()):
                stack += [first, second]
        return seen


def needed(state: OvercookedState, dishes: Collection[str]) -> set[str]:
    """Return every food any recipe could put into ``dishes``, and them.

    Like ``_Recipes.ancestors`` but through recipes of any size: this is the
    food the problem does not mark as waste.
    """
    seen: set[str] = set()
    stack = list(dishes)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for recipe in state.cook_recipes:
            if recipe.output == name:
                stack.append(recipe.ingredient)
        for recipe in state.combine_recipes:
            if recipe.output == name:
                stack += list(recipe.ingredients)
    return seen


def burns(
    state: OvercookedState,
    self_cooking: Collection[str],
    wanted: Collection[str],
    illegal: bool = True,
) -> set[tuple[str, str]]:
    """Return the (equipment, food) pairs where food left too long is ruined.

    Equipment that cooks by itself keeps cooking what is left in it, so food
    it would turn into something no visible order wants burns there. With
    illegal recipes on, it also cooks food it has no recipe for into
    garbage, so whatever it has finished making burns there too;
    ``illegal=False`` leaves that out, as main did before the watch rules.
    ``problem_generator`` states these as ``(burns ?e ?k)``.
    """
    pairs = {
        (recipe.equipment, recipe.ingredient)
        for recipe in state.cook_recipes
        if recipe.equipment in self_cooking and recipe.output not in wanted
    }
    if illegal and state.allow_illegal_recipes:
        pairs |= {
            (recipe.equipment, recipe.output)
            for recipe in state.cook_recipes
            if recipe.equipment in self_cooking
            and not state.can_cook_with(recipe.output, recipe.equipment)
        }
    return pairs


def watched_foods(
    state: OvercookedState, self_cooking: Collection[str], wanted: Collection[str]
) -> set[tuple[str, str]]:
    """Return the (equipment, food) pairs a chef must watch in that station.

    That is food that burns there and food that cooks there into food that
    burns, like raw and grilled patties in a pan that burns them.
    """
    burning = burns(state, self_cooking, wanted)
    return burning | {
        (recipe.equipment, recipe.ingredient)
        for recipe in state.cook_recipes
        if (recipe.equipment, recipe.output) in burning
    }


def plan_service(
    kitchen: Kitchen, state: OvercookedState, attempt: int = 0, watch: bool = True
) -> Service:
    """Choose the jobs for the next problem and what each one may use.

    ``watch=False`` drops the burns that only illegal recipes cause, as in
    the escape from the watch rules.
    """
    recipes = _Recipes(state)
    planner = _Planner(kitchen, recipes, attempt, watch)
    candidates = planner.candidates()
    chefs = list(kitchen.chefs)
    service = Service()
    if candidates and chefs:
        if attempt > 0:
            # The last plan went nowhere: one order, every chef, and a
            # different order on each retry.
            order = candidates[(attempt - 1) % len(candidates)]
            planner.add_job(service, order, chefs)
        elif planner.open_kitchen():
            planner.split_jobs(service, candidates, chefs)
        else:
            for order in candidates:
                if len(service.jobs) >= JOINT_ORDERS:
                    break
                planner.add_job(service, order, chefs)
        if not service.jobs and attempt <= len(candidates):
            # The checks above are stricter than the domain. Before giving
            # up, let the planner try one order with every chef and every
            # item; each order gets one such try.
            order = (
                candidates[(attempt - 1) % len(candidates)]
                if attempt
                else candidates[0]
            )
            planner.add_open_job(service, order, chefs)
    if not service.jobs:
        pending = bool(state.order_queue.pending) or any(
            order.deadline is not None for order in kitchen.orders
        )
        service.rest = "wait" if pending and chefs else "stop"
        return service
    planner.finish(service)
    return service


class _Planner:
    def __init__(
        self,
        kitchen: Kitchen,
        recipes: _Recipes,
        attempt: int = 0,
        watch: bool = True,
    ) -> None:
        self.kitchen = kitchen
        self.recipes = recipes
        self.taken: set[str] = set()
        self.plates_taken: set[str] = set()
        self.cookware_used: Tally[str] = Tally()
        # Loose food by name, for routes.
        self.pool: dict[str, list[str]] = defaultdict(list)
        for symbol, name in kitchen.foods.items():
            self.pool[name].append(symbol)
        wanted = recipes.needed([order.dish for order in kitchen.orders])
        # Watched stations: every wanted food made there burns if left, like
        # the pan that burns a patty. The domain only lets a chef cook such
        # food while watching it (cook_watch), and the station accepts it no
        # other way (``problem_generator.describe_equipment``), so plans do
        # not leave food in a watched station between steps, and jobs may
        # share it. Without the watch rules (``watch=False``) the station
        # accepts it again, as before those rules.
        burning = burns(recipes.state, kitchen.self_cooking, wanted, illegal=watch)
        makes: dict[str, set[str]] = defaultdict(set)
        for recipe in recipes.state.cook_recipes:
            if recipe.output in wanted:
                makes[recipe.equipment].add(recipe.output)
        self.watched = {
            station
            for station, foods in makes.items()
            if all((station, food) in burning for food in foods)
        }
        # Plates whose food no visible order can use: a chef may bin it.
        # After two plans in a row went nowhere anything may be binned (as in
        # ``problem_generator.describe_kinds``), so a plate holding food that
        # got stuck there can be emptied too.
        keep = wanted if attempt < 2 else set()
        contents: dict[str, list[str | None]] = defaultdict(list)
        for item, vessel in kitchen.in_vessel.items():
            contents[vessel].append(kitchen.foods.get(item))
        self.junk_plates = {
            vessel
            for vessel, names in contents.items()
            if vessel in kitchen.vessels
            and kitchen.vessels[vessel].plate
            and all(name is not None and name not in keep for name in names)
        }

    # --- Which orders -------------------------------------------------------

    def candidates(self) -> list[Order]:
        live = [
            order
            for order in self.kitchen.orders
            if order.deadline is None or order.deadline >= self.kitchen.timestep
        ]
        if self.kitchen.strict or self.kitchen.sequential:
            return live[:1]
        return sorted(
            live,
            key=lambda order: (
                order.due if order.due is not None else _INFINITY,
                -order.reward,
                order.index,
            ),
        )

    def open_kitchen(self) -> bool:
        """Return whether every chef can reach every tile any chef reaches."""
        reaches = {frozenset(self.kitchen.reach[chef]) for chef in self.kitchen.chefs}
        return len(reaches) == 1

    def split_jobs(
        self, service: Service, candidates: list[Order], chefs: list[str]
    ) -> None:
        """Give each order its own lead chef; spare chefs join the jobs.

        Jobs never share a station: two orders going through one cutboard at
        once block each other, because a plan leaves food in a station while
        its chef does other work. An order whose route needs a station every
        copy of which another job already uses waits for the next plan. The
        exception is a watched station (see ``watched``): every wanted food
        made there burns if left, so a chef only cooks it while watching and
        takes it out at once. Who gets the station first is not decided by
        deadline; the first chef to reach it uses it.

        The planner does not model walking, so who works where is decided
        here. The lead is the chef with the shortest walk to the stations the
        dish needs, counting only copies no earlier job claimed when there
        are any, so jobs keep to their own side of the kitchen. A spare chef
        joins the smallest crew whose lead stands nearest.
        """
        reach = set().union(*(self.kitchen.reach[chef] for chef in chefs))
        in_use: Tally[str] = Tally()
        claimed: set[str] = set()
        free = list(chefs)
        for order in candidates:
            if not free:
                break
            whole = self._route(order.dish, reach, chefs, frozenset(), 0)
            if whole is None or any(
                in_use[name] >= self._copies(name, reach)
                for name in whole.stations - self.watched
            ):
                continue
            if self._assign(service, order, free, claimed):
                in_use.update(whole.stations)
        for chef in free:
            if not service.jobs:
                break
            x, y = self.kitchen.chef_tiles[chef]
            job = min(
                service.jobs,
                key=lambda job: (
                    len(job.crew),
                    abs(self.kitchen.chef_tiles[job.crew[0]][0] - x)
                    + abs(self.kitchen.chef_tiles[job.crew[0]][1] - y),
                ),
            )
            job.crew.append(chef)
        if not service.jobs:
            # No chef can make an order alone: plan the first one together.
            for order in candidates:
                if self.add_job(service, order, chefs):
                    break

    def _assign(
        self, service: Service, order: Order, free: list[str], claimed: set[str]
    ) -> bool:
        """Give ``order`` to the free chef who makes it with the least work.

        Recipe steps first (a chef holding part of the dish needs fewer),
        then the walk to the dish's stations. The chosen chef claims the
        nearest copy of each station it needs.
        """
        ranked: list[tuple[int, int, int, str, _Route]] = []
        for index, chef in enumerate(free):
            route = self._route(
                order.dish, self.kitchen.reach[chef], [chef], frozenset(), 0
            )
            if route is not None:
                walk = sum(
                    self._distance_to(chef, self._station_tiles(name, chef, claimed))
                    for name in route.stations
                )
                ranked.append((route.cost, walk, index, chef, route))
        for _, _, _, chef, route in sorted(ranked, key=lambda row: row[:3]):
            if self.add_job(service, order, [chef]):
                service.jobs[-1].local = True
                for name in route.stations:
                    tiles = self._station_tiles(name, chef, claimed)
                    if tiles:
                        claimed.add(
                            min(tiles, key=lambda s: self._distance_to(chef, [s]))
                        )
                free.remove(chef)
                return True
        return False

    def _station_tiles(self, name: str, chef: str, claimed: set[str]) -> list[str]:
        """Return the tiles of stations called ``name`` the chef reaches.

        A portable station (a pan, a pot) counts where it stands. Copies no
        other job has claimed are returned when there are any.
        """
        reach = self.kitchen.reach[chef]
        tiles = [
            symbol
            for symbol in reach
            if self.kitchen.slots[symbol].equipment == name
            and not self.kitchen.slots[symbol].blocked
        ]
        for vessel in self.kitchen.vessels.values():
            if vessel.name == name and not vessel.plate:
                place = self.kitchen.on_slot.get(vessel.symbol)
                if self._usable(place, reach):
                    tiles.append(place)
        unclaimed = [symbol for symbol in tiles if symbol not in claimed]
        return unclaimed or tiles

    def _distance_to(self, chef: str, tiles: Collection[str]) -> int:
        """Return the chef's distance to the nearest of ``tiles`` (0 if none)."""
        distance = self.kitchen.distance[chef]
        return min((distance.get(symbol, _INFINITY) for symbol in tiles), default=0)

    def _copies(self, name: str, reach: set[str]) -> int:
        """Return how many stations called ``name`` the chefs can use."""
        fixed = sum(
            1
            for symbol in reach
            if self.kitchen.slots[symbol].equipment == name
            and not self.kitchen.slots[symbol].blocked
        )
        portable = sum(
            1
            for vessel in self.kitchen.vessels.values()
            if vessel.name == name and not vessel.plate and vessel.entity_id
        )
        return max(1, fixed, portable)

    # --- One job --------------------------------------------------------------

    def add_open_job(self, service: Service, order: Order, crew: list[str]) -> bool:
        """Add a job for ``order`` that may use everything the crew reaches.

        The fallback when ``add_job`` finds no way to make any order: its
        route and plate checks are stricter than the domain (a station or
        plate with something else in it, say), so here the planner gets
        every food and vessel within reach and decides for itself.
        """
        reach = set().union(*(self.kitchen.reach[chef] for chef in crew))
        if not any(self.kitchen.slots[s].delivery for s in reach):
            return False
        job = Job(symbol=f"j_{len(service.jobs)}", order=order, crew=list(crew))
        for symbol in self.kitchen.foods:
            place = self.kitchen.place_of(symbol)
            if self.kitchen.holder_of(symbol) in crew or self._usable(place, reach):
                job.items.add(symbol)
        for symbol, vessel in self.kitchen.vessels.items():
            place = self.kitchen.on_slot.get(symbol)
            if (
                self.kitchen.held_by.get(symbol) in crew
                or self._usable(place, reach)
                or (vessel.dispenser is not None and vessel.dispenser in reach)
            ):
                job.items.add(symbol)
        pulls = self._pulls(order.dish, reach, crew, 0)
        if pulls is None:
            return False
        for name, count in Tally(pulls).items():
            for index in range(count):
                job.tokens[f"t_{job.symbol[2:]}_{_short(name)}_{index}"] = name
        service.jobs.append(job)
        return True

    def _pulls(
        self, dish: str, reach: set[str], crew: list[str], depth: int
    ) -> list[str] | None:
        """Return the fewest storage pulls that make ``dish``, ignoring stations.

        Food already in the kitchen costs none. None means the crew cannot
        get some ingredient at all (an order that is out of stock).
        """
        if depth > _MAX_DEPTH:
            return None
        for item in self.pool.get(dish, ()):
            place = self.kitchen.place_of(item)
            if self.kitchen.holder_of(item) in crew or (
                place is not None and place in reach
            ):
                return []
        options: list[list[str]] = []
        if any(self.kitchen.slots[s].storage == dish for s in reach):
            options.append([dish])
        for ingredient, _ in self.recipes.cooked_from.get(dish, ()):
            inner = self._pulls(ingredient, reach, crew, depth + 1)
            if inner is not None:
                options.append(inner)
        for first, second in self.recipes.combined_from.get(dish, ()):
            left = self._pulls(first, reach, crew, depth + 1)
            right = self._pulls(second, reach, crew, depth + 1)
            if left is not None and right is not None:
                options.append(left + right)
        return min(options, key=len) if options else None

    def add_job(self, service: Service, order: Order, crew: list[str]) -> bool:
        """Add a job for ``order`` if ``crew`` can make and serve it."""
        reach = set().union(*(self.kitchen.reach[chef] for chef in crew))
        if not any(self.kitchen.slots[s].delivery for s in reach):
            return False
        route = self._route(order.dish, reach, crew, frozenset(), 0)
        if route is None:
            return False
        held = {item for item, chef in self.kitchen.held_by.items() if chef in crew}
        plate = self._plate(route, reach, crew, held)
        if plate is None:
            return False
        job = Job(symbol=f"j_{len(service.jobs)}", order=order, crew=list(crew))
        job.items |= set(route.items)
        job.items.add(plate)
        self.plates_taken.add(plate)
        self.taken |= set(route.items)
        # The vessels holding the job's food come with it.
        for item in list(job.items):
            if item in self.kitchen.in_vessel:
                job.items.add(self.kitchen.in_vessel[item])
        # Cookware for every station on the route that is portable.
        for name in route.stations:
            vessel = self._cookware(name, reach, crew)
            if vessel is not None:
                job.items.add(vessel)
                self.cookware_used[vessel] += 1
        for name, count in Tally(route.pulls).items():
            for index in range(count):
                job.tokens[f"t_{job.symbol[2:]}_{_short(name)}_{index}"] = name
        service.jobs.append(job)
        return True

    def _route(
        self,
        dish: str,
        reach: set[str],
        crew: list[str],
        used: frozenset[str],
        depth: int,
    ) -> _Route | None:
        """Return the cheapest way to make ``dish`` from what the crew reaches.

        Loose food of the right kind costs nothing; a storage pull costs one;
        cooking or combining costs one plus its ingredients.
        """
        if depth > _MAX_DEPTH:
            return None
        for item in self._loose(dish, reach, crew, used):
            return _Route(cost=0, items=frozenset({item}))
        options: list[_Route] = []
        if any(self.kitchen.slots[s].storage == dish for s in reach):
            options.append(_Route(cost=1, pulls=(dish,)))
        for ingredient, station in self.recipes.cooked_from.get(dish, ()):
            if not self._station_usable(station, reach, crew):
                continue
            inner = self._route(ingredient, reach, crew, used, depth + 1)
            if inner is not None:
                options.append(
                    _Route(
                        cost=inner.cost + 1,
                        pulls=inner.pulls,
                        items=inner.items,
                        stations=inner.stations | {station},
                    )
                )
        for first, second in self.recipes.combined_from.get(dish, ()):
            left = self._route(first, reach, crew, used, depth + 1)
            if left is None:
                continue
            right = self._route(second, reach, crew, used | left.items, depth + 1)
            if right is None:
                continue
            options.append(
                _Route(
                    cost=left.cost + right.cost + 1,
                    pulls=left.pulls + right.pulls,
                    items=left.items | right.items,
                    stations=left.stations | right.stations,
                )
            )
        if not options:
            return None
        return min(options, key=lambda route: route.cost)

    def _loose(
        self, name: str, reach: set[str], crew: list[str], used: frozenset[str]
    ) -> list[str]:
        """Return loose food called ``name`` the crew can get, nearest first."""
        found: list[tuple[int, str]] = []
        for item in self.pool.get(name, ()):
            if item in self.taken or item in used:
                continue
            holder = self.kitchen.holder_of(item)
            if holder is not None:
                if holder in crew:
                    found.append((0, item))
                continue
            place = self.kitchen.place_of(item)
            if place is None or place not in reach or self.kitchen.slots[place].blocked:
                continue
            vessel = self.kitchen.in_vessel.get(item)
            if vessel is not None and vessel in self.plates_taken:
                continue
            found.append((self._distance(place, crew), item))
        return [item for _, item in sorted(found)]

    def _station_usable(self, station: str, reach: set[str], crew: list[str]) -> bool:
        for symbol in reach:
            slot = self.kitchen.slots[symbol]
            if slot.equipment == station and not slot.blocked:
                return True
        return self._cookware(station, reach, crew) is not None

    def _cookware(self, name: str, reach: set[str], crew: list[str]) -> str | None:
        """Return the nearest portable station called ``name`` the crew can use.

        One no other job uses yet comes first.
        """
        best: tuple[int, int, str] | None = None
        for vessel in self.kitchen.vessels.values():
            if vessel.plate or vessel.name != name or vessel.entity_id is None:
                continue
            if vessel.requires and not any(
                self.kitchen.slots[s].equipment == vessel.requires for s in reach
            ):
                continue
            holder = self.kitchen.held_by.get(vessel.symbol)
            if holder is not None:
                distance = 0 if holder in crew else None
            else:
                place = self.kitchen.on_slot.get(vessel.symbol)
                distance = (
                    self._distance(place, crew) if self._usable(place, reach) else None
                )
            if distance is None:
                continue
            choice = (self.cookware_used[vessel.symbol], distance, vessel.symbol)
            if best is None or choice < best:
                best = choice
        return best[2] if best else None

    def _plate(
        self, route: _Route, reach: set[str], crew: list[str], held: set[str]
    ) -> str | None:
        """Return the plate the job serves on, or None if there is none."""
        sink = any(self.kitchen.slots[s].sink for s in reach)

        def usable(vessel: Vessel) -> bool:
            return (
                vessel.plate
                and vessel.symbol not in self.plates_taken
                and (sink or not vessel.dirty)
            )

        # A plate that already holds the job's food.
        for item in route.items:
            vessel = self.kitchen.vessels.get(self.kitchen.in_vessel.get(item, ""))
            if vessel is not None and vessel.plate and usable(vessel):
                return vessel.symbol
        holds_food = set(self.kitchen.in_vessel.values())
        has_bin = any(self.kitchen.slots[s].bin for s in reach)
        options: list[tuple[int, int, str]] = []
        for vessel in self.kitchen.vessels.values():
            if not usable(vessel):
                continue
            # A plate holding food no order can use is emptied in the bin
            # first, so it comes after the empty ones.
            emptying = 0
            if vessel.symbol in holds_food:
                if vessel.symbol not in self.junk_plates or not has_bin:
                    continue
                emptying = _EMPTYING_STEPS
            if vessel.symbol in held:
                options.append((emptying, 0, vessel.symbol))
            elif vessel.dispenser is not None:
                if vessel.dispenser not in reach:
                    continue
                # Plates leave a dispenser in order.
                earlier = [
                    other
                    for other in self.kitchen.vessels.values()
                    if other.dispenser == vessel.dispenser
                    and other.position < vessel.position
                    and other.symbol not in self.plates_taken
                ]
                if earlier:
                    continue
                options.append(
                    (
                        1 + vessel.dirty,
                        self._distance(vessel.dispenser, crew),
                        vessel.symbol,
                    )
                )
            else:
                place = self.kitchen.on_slot.get(vessel.symbol)
                if (
                    place is None
                    or place not in reach
                    or self.kitchen.slots[place].blocked
                ):
                    continue
                options.append(
                    (
                        1 + 2 * vessel.dirty + emptying,
                        self._distance(place, crew),
                        vessel.symbol,
                    )
                )
        if not options:
            return None
        return min(options)[2]

    def _usable(self, place: str | None, reach: set[str]) -> bool:
        """Return whether something standing on ``place`` can be used.

        The tile must be in reach and one the model describes: a tile with
        two fixed stations (a stove on an oven) is blocked and left out of
        the problem, so nothing on it can be planned with.
        """
        return (
            place is not None
            and place in reach
            and not self.kitchen.slots[place].blocked
        )

    def _distance(self, slot: str, crew: list[str]) -> int:
        return min(
            (self.kitchen.distance[chef].get(slot, _INFINITY) for chef in crew),
            default=_INFINITY,
        )

    # --- Everything else in the problem ---------------------------------------

    def finish(self, service: Service) -> None:
        """Describe the rest of the problem: shared items, tiles and kinds."""
        kitchen = self.kitchen
        described = self._share_items(service)
        crews = {chef for job in service.jobs for chef in job.crew}
        reach = set().union(*(kitchen.reach[chef] for chef in crews))
        slots = self._occupied_slots(service, reach, described)
        empty = [
            symbol
            for symbol in reach
            if kitchen.slots[symbol].counter
            and kitchen.slots[symbol].item is None
            and not kitchen.slots[symbol].blocked
        ]
        stations = [
            kitchen.slots[s].location
            for s in reach
            if kitchen.slots[s].equipment is not None
        ]
        counters = self._counters(service, empty, stations)
        for chosen in counters.values():
            slots |= chosen
        slots |= self._handoffs(crews, empty, stations)
        load: Tally[str] = Tally()
        for job in service.jobs:
            if job.local:
                job.slots = self._workspace(
                    job, slots, counters[job.symbol], load, service.shared
                )
            else:
                job.slots = set(slots)
        service.slots = slots
        service.kinds = self._kinds(service, described)

    def _share_items(self, service: Service) -> set[str]:
        """Add what every job may touch to ``service.shared``.

        What crew chefs hold is theirs to use or get rid of, and food in a
        described vessel comes with it. Returns every item the problem
        describes.
        """
        kitchen = self.kitchen
        crews = {chef for job in service.jobs for chef in job.crew}
        owned = {item for job in service.jobs for item in job.items}
        for item, chef in kitchen.held_by.items():
            if chef in crews and item not in owned:
                service.shared.add(item)
        for item, vessel in kitchen.in_vessel.items():
            if kitchen.held_by.get(vessel) in crews and item not in owned:
                service.shared.add(item)
        described = owned | service.shared
        for item, vessel in kitchen.in_vessel.items():
            if vessel in described and item not in described:
                service.shared.add(item)
        return owned | service.shared

    def _occupied_slots(
        self, service: Service, reach: set[str], described: set[str]
    ) -> set[str]:
        """Return the tiles in reach that are not bare counters or hold a
        described item.

        Junk in a station is in the way, so anyone may bin it: it joins
        ``service.shared`` and ``described``.
        """
        kitchen = self.kitchen
        slots: set[str] = set()
        for symbol in reach:
            slot = kitchen.slots[symbol]
            if slot.blocked:
                continue
            if slot.item is not None:
                if slot.item not in described:
                    if slot.equipment is not None and slot.item in kitchen.foods:
                        service.shared.add(slot.item)
                        described.add(slot.item)
                    else:
                        continue
                slots.add(symbol)
            elif not slot.counter:
                slots.add(symbol)
        return slots

    def _counters(
        self,
        service: Service,
        empty: list[str],
        stations: list[tuple[int, int]],
    ) -> dict[str, set[str]]:
        """Return a few empty counters for each job: near its chefs and
        near the kitchen's stations.

        Every job ranks counters by the same nearness to stations, so two
        jobs may share a counter, though they rarely share a station.
        """
        counters: dict[str, set[str]] = {}
        for job in service.jobs:
            job_reach = set().union(*(self.kitchen.reach[chef] for chef in job.crew))
            mine = [s for s in empty if s in job_reach]
            near_chef = sorted(mine, key=lambda s: (self._distance(s, job.crew), s))
            near_work = sorted(
                mine, key=lambda s: (_near(self.kitchen.slots[s].location, stations), s)
            )
            counters[job.symbol] = set(near_chef[: COUNTERS_PER_JOB - 1])
            counters[job.symbol] |= set(near_work[:COUNTERS_PER_JOB])
        return counters

    def _handoffs(
        self, crews: set[str], empty: list[str], stations: list[tuple[int, int]]
    ) -> set[str]:
        """Return a few empty counters where the parts of a divided kitchen
        meet, nearest the stations first."""
        groups: dict[frozenset[str], list[str]] = defaultdict(list)
        for symbol in empty:
            sharing = frozenset(
                frozenset(self.kitchen.reach[chef])
                for chef in crews
                if symbol in self.kitchen.reach[chef]
            )
            if len(sharing) > 1:
                groups[sharing].append(symbol)
        handoffs: set[str] = set()
        for group in groups.values():
            group.sort(
                key=lambda s: (_near(self.kitchen.slots[s].location, stations), s)
            )
            handoffs.update(group[:HANDOFF_COUNTERS])
        return handoffs

    def _kinds(self, service: Service, described: set[str]) -> set[str]:
        """Return the food kinds the problem needs: every food that can go
        into a job's dish, and the kinds of the described food."""
        kinds: set[str] = set()
        for job in service.jobs:
            kinds |= self.recipes.ancestors(job.order.dish)
        for item in described:
            if item in self.kitchen.foods:
                kinds.add(self.kitchen.foods[item])
        return kinds

    def _workspace(
        self,
        job: Job,
        slots: set[str],
        counters: set[str],
        load: Tally[str],
        shared: set[str],
    ) -> set[str]:
        """Return the tiles a job in an open kitchen works at.

        Where its items are, its counters, and the nearest storage, station,
        delivery spot, bin and sink of each kind it may need. Stations are
        spread over the jobs: one already chosen counts as a few steps
        further away.
        """
        kitchen = self.kitchen
        mine = set(counters)
        for symbol in slots:
            item = kitchen.slots[symbol].item
            if item is not None and (item in job.items or item in shared):
                mine.add(symbol)
        for item in job.items:
            vessel = kitchen.vessels.get(item)
            if vessel is not None and vessel.dispenser is not None:
                mine.add(vessel.dispenser)
        foods = self.recipes.ancestors(job.order.dish)
        stations_needed = {
            station
            for food in foods
            for _, station in self.recipes.cooked_from.get(food, ())
        }
        bases = {
            vessel.requires
            for item in job.items
            if (vessel := kitchen.vessels.get(item)) is not None and vessel.requires
        }
        placed_bases = {
            kitchen.slots[place].equipment
            for item in job.items
            if (place := kitchen.on_slot.get(item)) is not None
        }
        best: dict[str, tuple[int, str]] = {}
        for symbol in slots:
            slot = kitchen.slots[symbol]
            distance = self._distance(symbol, job.crew)
            keys: list[str] = []
            if slot.storage is not None and slot.storage in foods:
                keys.append("storage:" + slot.storage)
            if slot.equipment is not None and slot.equipment in stations_needed:
                keys.append("station:" + slot.equipment)
                distance += _BUSY_STEPS * load[symbol]
            if slot.equipment is not None and slot.equipment in bases - placed_bases:
                keys.append("base:" + slot.equipment)
            for flag in ("delivery", "bin", "sink"):
                if getattr(slot, flag):
                    keys.append(flag)
            for key in keys:
                choice = (distance, symbol)
                if key not in best or choice < best[key]:
                    best[key] = choice
        for key, (_, symbol) in best.items():
            mine.add(symbol)
            if key.startswith("station:"):
                load[symbol] += 1
        return mine


def _near(location: tuple[int, int], stations: list[tuple[int, int]]) -> int:
    return min(
        (abs(location[0] - x) + abs(location[1] - y) for x, y in stations),
        default=0,
    )


def _short(name: str) -> str:
    return pddl_name(name.rsplit("/", 1)[-1])
