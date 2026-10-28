"""What the kitchen looks like: the fixed layout once per game, the moving parts every tick.

``Layout`` is built on the first tick and cached: walkable floor, all-pairs
walking distances, where each storage, dispenser, bin and hatch is, and the
recipe trees of every dish. ``Snapshot`` reads the parts that change (chefs,
held items, what each station holds and how far it has cooked) in one pass
over the entities, so the planner never touches pydantic models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from simulator.entities import (
    Agent,
    Bin,
    Bounds,
    Counter,
    Delivery,
    Equipment,
    Food,
    OvercookedState,
    Plate,
    PlateDispenser,
    Storage,
)
from simulator.environment import Environment

Tile = tuple[int, int]

DELTAS: dict[str, Tile] = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}
FACING: dict[Tile, str] = {delta: name for name, delta in DELTAS.items()}


class Layout:
    """Static facts about one game's kitchen."""

    def __init__(self, environment: Environment) -> None:
        bounds = environment.get_first_entity_of_type(Bounds)
        width = bounds.width if bounds else 0
        height = bounds.height if bounds else 0
        x0 = (bounds.x or 0) if bounds else 0
        y0 = (bounds.y or 0) if bounds else 0
        blocked: set[Tile] = set()
        self.storages: dict[str, list[Tile]] = {}
        self.dispensers: list[Tile] = []
        self.bins: list[Tile] = []
        self.deliveries: list[Tile] = []
        self.counters: set[Tile] = set()
        for entity in environment.entities.values():
            x = getattr(entity, "x", None)
            y = getattr(entity, "y", None)
            if x is None or y is None:
                continue
            if isinstance(entity, Agent):
                continue
            if getattr(entity, "blocks_movement", False):
                blocked.add((x, y))
            if isinstance(entity, Storage):
                self.storages.setdefault(entity.food_name, []).append((x, y))
            elif isinstance(entity, PlateDispenser):
                self.dispensers.append((x, y))
            elif isinstance(entity, Bin):
                self.bins.append((x, y))
            elif isinstance(entity, Delivery):
                self.deliveries.append((x, y))
            elif isinstance(entity, Counter):
                self.counters.add((x, y))
        self.walk: frozenset[Tile] = frozenset(
            (x, y)
            for x in range(x0, x0 + width)
            for y in range(y0, y0 + height)
            if (x, y) not in blocked
        )
        self.dist: dict[Tile, dict[Tile, int]] = {
            tile: self._bfs(tile) for tile in self.walk
        }
        self._access_cache: dict[Tile, list[Tile]] = {}
        self._reach_cache: dict[Tile, dict[Tile, tuple[int, Tile]]] = {}

        state = environment.get_first_entity_of_type(OvercookedState)
        self.cook: dict[tuple[str, str], str] = {}
        self.combine: dict[frozenset[str], str] = {}
        self.raw: set[str] = set()
        if state is not None:
            for recipe in state.cook_recipes:
                self.cook[(recipe.ingredient, recipe.equipment)] = recipe.output
            for recipe in state.combine_recipes:
                self.combine[frozenset(recipe.ingredients)] = recipe.output
            self.raw = {
                name for name, food in state.food_definitions.items() if food.raw
            }
        self.station_kinds: frozenset[str] = frozenset(e for _, e in self.cook)
        self._trees: dict[str, frozenset[str]] = {}
        self._leaves: dict[str, int] = {}

    # Distances -----------------------------------------------------------

    def _bfs(self, start: Tile) -> dict[Tile, int]:
        seen = {start: 0}
        queue = deque([start])
        walk = self.walk
        while queue:
            x, y = queue.popleft()
            d = seen[(x, y)] + 1
            for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if n in walk and n not in seen:
                    seen[n] = d
                    queue.append(n)
        return seen

    def access(self, target: Tile) -> list[Tile]:
        """Walkable tiles from which a chef can face ``target``."""
        cached = self._access_cache.get(target)
        if cached is None:
            x, y = target
            cached = [
                n
                for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                if n in self.walk
            ]
            self._access_cache[target] = cached
        return cached

    def reach(self, target: Tile) -> dict[Tile, tuple[int, Tile]]:
        """For every floor tile, the steps to the nearest access tile of ``target``."""
        cached = self._reach_cache.get(target)
        if cached is None:
            cached = {}
            tiles = self.access(target)
            for tile in self.walk:
                row = self.dist[tile]
                best: tuple[int, Tile] | None = None
                for a in tiles:
                    d = row.get(a)
                    if d is not None and (best is None or d < best[0]):
                        best = (d, a)
                if best is not None:
                    cached[tile] = best
            self._reach_cache[target] = cached
        return cached

    # Recipes ---------------------------------------------------------------

    def add(self, content: str | None, food: str) -> str | None:
        """What a plate holding ``content`` holds once ``food`` joins it."""
        if content is None:
            return food
        return self.combine.get(frozenset((content, food)))

    def tree(self, dish: str) -> frozenset[str]:
        """Every food on any recipe route to ``dish``, the dish included."""
        cached = self._trees.get(dish)
        if cached is None:
            found = {dish}
            todo = [dish]
            while todo:
                out = todo.pop()
                for (ingredient, _), output in self.cook.items():
                    if output == out and ingredient not in found:
                        found.add(ingredient)
                        todo.append(ingredient)
                for ingredients, output in self.combine.items():
                    if output == out:
                        for ingredient in ingredients:
                            if ingredient not in found:
                                found.add(ingredient)
                                todo.append(ingredient)
            cached = frozenset(found)
            self._trees[dish] = cached
        return cached

    def leaves(self, food: str) -> int:
        """How many stored ingredients go into ``food`` (a rough progress measure)."""
        cached = self._leaves.get(food)
        if cached is not None:
            return cached
        self._leaves[food] = 1  # guards against cycles
        best = 1 if food in self.storages else 0
        for (ingredient, _), output in self.cook.items():
            if output == food:
                best = max(best, self.leaves(ingredient))
        for ingredients, output in self.combine.items():
            if output == food:
                best = max(best, sum(self.leaves(i) for i in ingredients))
        self._leaves[food] = best
        return best


@dataclass
class Station:
    id: str
    name: str
    tile: Tile
    food: str | None
    food_id: str | None
    elapsed: int
    cook_time: int
    self_cook: bool
    supported: bool


@dataclass
class Chef:
    id: str
    tile: Tile
    orientation: str
    team: str | None
    held_kind: str | None  # "plate", "food", "equipment" or None
    held_id: str | None
    held_name: str | None  # food name, or equipment name
    plate_content: str | None  # food on a held plate


@dataclass
class Snapshot:
    timestep: int
    chefs: dict[str, Chef] = field(default_factory=dict)
    stations: dict[str, Station] = field(default_factory=dict)
    # Counters with nothing on them (tile -> counter id).
    free_counters: dict[Tile, str] = field(default_factory=dict)
    # Plates resting on counters: tile -> (plate id, food name or None).
    counter_plates: dict[Tile, tuple[str, str | None]] = field(default_factory=dict)
    orders: list[tuple[str, int, int]] = field(default_factory=list)
    # The next orders the generator will reveal: (dish, reward, time limit).
    upcoming: list[tuple[str, int, int]] = field(default_factory=list)
    # Every dish that can be ordered.
    menu: set[str] = field(default_factory=set)
    # Orders delivered or expired so far, by either team.
    events: int = 0
    occupied: set[Tile] = field(default_factory=set)


def read_snapshot(environment: Environment, layout: Layout) -> Snapshot:
    snap = Snapshot(timestep=environment.timestep)
    foods: dict[str, str] = {}
    equipment: dict[str, Equipment] = {}
    agents: list[Agent] = []
    counters: list[Counter] = []
    state: OvercookedState | None = None
    for entity in environment.entities.values():
        if isinstance(entity, Food):
            foods[entity.id] = entity.name
        elif isinstance(entity, Equipment):
            equipment[entity.id] = entity
        elif isinstance(entity, Agent):
            agents.append(entity)
        elif isinstance(entity, Counter):
            counters.append(entity)
        elif isinstance(entity, OvercookedState):
            state = entity

    at_cell: dict[Tile, list[str]] = {}
    for eq in equipment.values():
        if eq.x is not None and eq.y is not None:
            at_cell.setdefault((eq.x, eq.y), []).append(eq.name)

    for eq in equipment.values():
        if isinstance(eq, Plate) or eq.name not in layout.station_kinds:
            continue
        if eq.x is None or eq.y is None:
            continue
        tile = (eq.x, eq.y)
        food_id = eq.held_item_id
        food = foods.get(food_id) if food_id else None
        elapsed = 0
        if eq.cooking is not None and eq.cooking.food_id == food_id:
            elapsed = eq.cooking.elapsed
        supported = not eq.requires or eq.requires in at_cell.get(tile, [])
        snap.stations[eq.id] = Station(
            id=eq.id,
            name=eq.name,
            tile=tile,
            food=food,
            food_id=food_id,
            elapsed=elapsed,
            cook_time=eq.cook_time,
            self_cook=eq.cooks_by_itself,
            supported=supported,
        )

    for counter in counters:
        if counter.x is None or counter.y is None:
            continue
        tile = (counter.x, counter.y)
        if counter.held_item_id is None:
            if tile not in at_cell:
                snap.free_counters[tile] = counter.id
        else:
            item = equipment.get(counter.held_item_id)
            if isinstance(item, Plate) and not item.dirty:
                content = foods.get(item.held_item_id) if item.held_item_id else None
                snap.counter_plates[tile] = (item.id, content)

    for agent in agents:
        if agent.x is None or agent.y is None:
            continue
        held_kind = held_name = content = None
        if agent.held_item_id:
            if agent.held_item_id in foods:
                held_kind, held_name = "food", foods[agent.held_item_id]
            else:
                eq = equipment.get(agent.held_item_id)
                if isinstance(eq, Plate):
                    held_kind, held_name = "plate", eq.name
                    if eq.held_item_id:
                        content = foods.get(eq.held_item_id)
                    if eq.dirty:
                        held_kind = "equipment"
                elif eq is not None:
                    held_kind, held_name = "equipment", eq.name
                    if eq.held_item_id:
                        content = foods.get(eq.held_item_id)
                else:
                    held_kind = "other"
        snap.chefs[agent.id] = Chef(
            id=agent.id,
            tile=(agent.x, agent.y),
            orientation=agent.orientation,
            team=agent.team,
            held_kind=held_kind,
            held_id=agent.held_item_id,
            held_name=held_name,
            plate_content=content,
        )
        snap.occupied.add((agent.x, agent.y))

    if state is not None:
        queue = state.order_queue
        now = environment.timestep
        for entry in queue.visible:
            limit = entry.time_limit
            deadline = entry.revealed_at + limit if limit is not None else 10**9
            if now <= deadline:
                snap.orders.append((entry.name, entry.reward, deadline))
        snap.events = queue.delivered_count + queue.expired_count
        generator = queue.generator
        if generator is not None and generator.pool:
            # Only the menu is read. Orders not yet revealed are not predicted:
            # rebuilding them from the generator's seed would use information
            # the game hides, so ``snap.upcoming`` stays empty.
            snap.menu = {entry.name for entry in generator.pool}
        snap.menu |= {name for name, _, _ in snap.orders}
    return snap
