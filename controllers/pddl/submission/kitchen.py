"""Read the kitchen into the objects the PDDL problem talks about.

The problem generator and the plan translator both build a ``Kitchen`` from
the same ``Environment`` snapshot, so they agree on every symbol: the
generator writes facts about ``Kitchen`` objects, and the translator turns
the symbols in a plan back into chef ids, tiles and catalog names.

The model is instance-level: each chef, each tile a chef can act on, each
piece of food, each plate or pan and each visible order is one PDDL object.
Food keeps its identity while it is cooked or combined; its current kind is
a fluent. Storages hand out new food, so each storage kind gets a few
"token" food objects that do not exist yet, sized to what the orders could
need.

Names:

- chefs ``a_<id>``; tiles ``loc_<x>_<y>``; food ``f_<id>``; tokens
  ``t_<kind>_<n>``; vessels (plates and other portable equipment)
  ``v_<id>``; food kinds ``k_<name>``; equipment kinds ``e_<name>``; orders
  ``o_<n>``.
"""

from __future__ import annotations

import re
from collections import Counter as Tally
from collections import deque
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from simulator.entities import (
    Agent,
    Bin,
    Bounds,
    Counter,
    Delivery,
    Equipment,
    Food,
    GameObject,
    OvercookedState,
    Plate,
    Storage,
)
from simulator.environment import Environment
from simulator.types import Location

# Recipe trees are shallow; this only guards against cyclic recipes.
_MAX_RECIPE_DEPTH = 16


def pddl_name(value: str) -> str:
    """Convert a kitchen name to a lower-case PDDL-safe fragment.

    ``catalog/food/coconut_juice`` becomes ``catalog_food_coconut_juice``.
    """
    return re.sub(r"[^0-9a-z_]+", "_", value.lower()).strip("_") or "x"


def kind_symbol(food_name: str) -> str:
    return "k_" + pddl_name(food_name)


def equipment_symbol(equipment_name: str) -> str:
    return "e_" + pddl_name(equipment_name)


def tile_symbol(location: Location) -> str:
    return f"loc_{location[0]}_{location[1]}"


@dataclass
class Slot:
    """A tile a chef acts on: counter, station, storage, delivery or bin."""

    symbol: str
    location: Location
    # Bare counter: takes raw food or a vessel.
    counter: bool = False
    # Name of the fixed equipment on the tile (cutboard, stove, ...), if any.
    equipment: str | None = None
    storage: str | None = None
    delivery: bool = False
    bin: bool = False
    # The symbol of the item on the tile, if the model can describe it.
    item: str | None = None
    # Whether something the model cannot describe occupies the tile.
    blocked: bool = False


@dataclass
class Vessel:
    """A portable piece of equipment: a plate, pan, pot or mixing bowl."""

    symbol: str
    entity_id: str
    name: str
    plate: bool


@dataclass
class Kitchen:
    chefs: dict[str, str] = field(default_factory=dict)  # symbol -> agent id
    slots: dict[str, Slot] = field(default_factory=dict)
    vessels: dict[str, Vessel] = field(default_factory=dict)
    foods: dict[str, str] = field(default_factory=dict)  # symbol -> food name
    tokens: dict[str, str] = field(default_factory=dict)  # symbol -> food name
    kinds: dict[str, str] = field(default_factory=dict)  # symbol -> food name
    equipment: dict[str, str] = field(default_factory=dict)  # symbol -> name
    orders: dict[str, str] = field(default_factory=dict)  # symbol -> dish name
    # Where each described item is: chef holding it, slot it sits on, or the
    # vessel it is in. Values are symbols.
    held_by: dict[str, str] = field(default_factory=dict)
    on_slot: dict[str, str] = field(default_factory=dict)
    in_vessel: dict[str, str] = field(default_factory=dict)
    # Static reachability: chef symbol -> slot symbols it can act on.
    reach: dict[str, set[str]] = field(default_factory=dict)

    def slot_at(self, location: Location) -> Slot | None:
        return self.slots.get(tile_symbol(location))


def read_kitchen(
    environment: Environment,
    controlled_agent_ids: Collection[str] | None = None,
) -> Kitchen:
    """Build the kitchen model for ``environment``.

    ``controlled_agent_ids`` limits the chefs described; ``None`` means all.
    """
    state = environment.get_first_entity_of_type(OvercookedState)
    if state is None:
        raise ValueError("The environment has no OvercookedState")
    kitchen = Kitchen()
    _read_kinds(environment, state, kitchen)
    _read_slots(environment, kitchen)
    _read_items(environment, kitchen)
    _read_chefs(environment, controlled_agent_ids, kitchen)
    _read_orders(state, kitchen)
    _read_tokens(state, kitchen)
    return kitchen


# --- Kinds --------------------------------------------------------------------


def _read_kinds(
    environment: Environment, state: OvercookedState, kitchen: Kitchen
) -> None:
    names: list[str] = list(state.food_definitions)
    names += [food.name for food in environment.get_entities_of_type(Food)]
    names += [storage.food_name for storage in environment.get_entities_of_type(Storage)]
    names += list(state.visible_orders)
    for recipe in state.cook_recipes:
        names += [recipe.ingredient, recipe.output]
    for recipe in state.combine_recipes:
        names += [*recipe.ingredients, recipe.output]
    for name in names:
        if name:
            kitchen.kinds.setdefault(kind_symbol(name), name)

    equipment_names: list[str] = []
    for equipment in environment.get_entities_of_type(Equipment):
        equipment_names.append(equipment.name)
        if equipment.requires:
            equipment_names.append(equipment.requires)
    for name in equipment_names:
        if name:
            kitchen.equipment.setdefault(equipment_symbol(name), name)


# --- Tiles --------------------------------------------------------------------


def _read_slots(environment: Environment, kitchen: Kitchen) -> None:
    def slot(location: Location) -> Slot:
        symbol = tile_symbol(location)
        if symbol not in kitchen.slots:
            kitchen.slots[symbol] = Slot(symbol=symbol, location=location)
        return kitchen.slots[symbol]

    for counter in environment.get_entities_of_type(Counter):
        if counter.x is not None and counter.y is not None:
            slot((counter.x, counter.y)).counter = True
    for storage in environment.get_entities_of_type(Storage):
        if storage.x is not None and storage.y is not None and storage.food_name:
            slot((storage.x, storage.y)).storage = storage.food_name
    for delivery in environment.get_entities_of_type(Delivery):
        if delivery.x is not None and delivery.y is not None:
            slot((delivery.x, delivery.y)).delivery = True
    for bin_ in environment.get_entities_of_type(Bin):
        if bin_.x is not None and bin_.y is not None:
            slot((bin_.x, bin_.y)).bin = True
    for equipment in environment.get_entities_of_type(Equipment):
        if equipment.portable or equipment.x is None or equipment.y is None:
            continue
        tile = slot((equipment.x, equipment.y))
        # A counter under a station is not a bare counter: Place refuses it.
        tile.counter = False
        if tile.equipment is None:
            tile.equipment = equipment.name
        else:
            # Two fixed stations on one tile: the model can only use one.
            tile.blocked = True


# --- Items --------------------------------------------------------------------


def _read_items(environment: Environment, kitchen: Kitchen) -> None:
    agents = environment.get_entities_of_type(Agent)
    holder_of: dict[str, Agent] = {
        agent.held_item_id: agent for agent in agents if agent.held_item_id
    }
    equipment_by_id = {
        equipment.id: equipment
        for equipment in environment.get_entities_of_type(Equipment)
    }
    counters_by_item = {
        counter.held_item_id: counter
        for counter in environment.get_entities_of_type(Counter)
        if counter.held_item_id
    }
    container_of: dict[str, Equipment] = {
        equipment.held_item_id: equipment
        for equipment in equipment_by_id.values()
        if equipment.held_item_id
    }

    # Vessels first, so food can refer to them.
    for equipment in equipment_by_id.values():
        if not equipment.portable:
            continue
        if isinstance(equipment, Plate) and equipment.dirty:
            # Dirty plates need a sink; not modelled yet.
            _block_tile_of(kitchen, equipment)
            continue
        symbol = "v_" + pddl_name(equipment.id)
        kitchen.vessels[symbol] = Vessel(
            symbol=symbol,
            entity_id=equipment.id,
            name=equipment.name,
            plate=isinstance(equipment, Plate),
        )
        if equipment.id in holder_of:
            kitchen.held_by[symbol] = "a_" + pddl_name(holder_of[equipment.id].id)
        elif equipment.x is not None and equipment.y is not None:
            tile = kitchen.slot_at((equipment.x, equipment.y))
            if tile is None or tile.item is not None:
                del kitchen.vessels[symbol]
                continue
            tile.item = symbol
            kitchen.on_slot[symbol] = tile.symbol

    for food in environment.get_entities_of_type(Food):
        symbol = "f_" + pddl_name(food.id)
        if food.id in holder_of:
            kitchen.foods[symbol] = food.name
            kitchen.held_by[symbol] = "a_" + pddl_name(holder_of[food.id].id)
            continue
        container = container_of.get(food.id)
        if container is not None and container.portable:
            vessel = "v_" + pddl_name(container.id)
            if vessel in kitchen.vessels:
                kitchen.foods[symbol] = food.name
                kitchen.in_vessel[symbol] = vessel
            continue
        location: Location | None = None
        if container is not None and container.x is not None and container.y is not None:
            location = (container.x, container.y)
        elif food.id in counters_by_item:
            counter = counters_by_item[food.id]
            if counter.x is not None and counter.y is not None:
                location = (counter.x, counter.y)
        if location is None:
            continue
        tile = kitchen.slot_at(location)
        if tile is None or tile.item is not None:
            continue
        kitchen.foods[symbol] = food.name
        tile.item = symbol
        kitchen.on_slot[symbol] = tile.symbol


def _block_tile_of(kitchen: Kitchen, item: GameObject) -> None:
    if item.x is None or item.y is None:
        return
    tile = kitchen.slot_at((item.x, item.y))
    if tile is not None:
        tile.blocked = True


# --- Chefs --------------------------------------------------------------------


def _read_chefs(
    environment: Environment,
    controlled_agent_ids: Collection[str] | None,
    kitchen: Kitchen,
) -> None:
    blocked = _blocked_tiles(environment)
    inside = _inside(environment)
    for agent in sorted(environment.get_entities_of_type(Agent), key=lambda a: a.id):
        if controlled_agent_ids is not None and agent.id not in controlled_agent_ids:
            continue
        if agent.x is None or agent.y is None:
            continue
        symbol = "a_" + pddl_name(agent.id)
        kitchen.chefs[symbol] = agent.id
        floor = _reachable_floor(blocked, inside, (agent.x, agent.y))
        kitchen.reach[symbol] = {
            tile.symbol
            for tile in kitchen.slots.values()
            if any(
                (tile.location[0] + dx, tile.location[1] + dy) in floor
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        }


def _blocked_tiles(environment: Environment) -> set[Location]:
    """Return tiles that block walking, ignoring chefs (they move)."""
    return {
        (entity.x, entity.y)
        for entity in environment.get_entities_of_type(GameObject)
        if entity.x is not None
        and entity.y is not None
        and entity.blocks_movement
        and not isinstance(entity, Agent)
    }


def _inside(environment: Environment) -> Callable[[int, int], bool]:
    """Return a test for tiles inside the kitchen.

    Use the kitchen's ``Bounds``; without one, the box around every object.
    """
    bounds = environment.get_first_entity_of_type(Bounds)
    if bounds is not None:
        return bounds.contains
    placed = [
        (entity.x, entity.y)
        for entity in environment.get_entities_of_type(GameObject)
        if entity.x is not None and entity.y is not None
    ]
    if not placed:
        return lambda x, y: False
    low_x = min(x for x, _ in placed)
    high_x = max(x for x, _ in placed)
    low_y = min(y for _, y in placed)
    high_y = max(y for _, y in placed)
    return lambda x, y: low_x <= x <= high_x and low_y <= y <= high_y


def _reachable_floor(
    blocked: set[Location], inside: Callable[[int, int], bool], start: Location
) -> set[Location]:
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for step in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if step in seen or step in blocked or not inside(*step):
                continue
            seen.add(step)
            queue.append(step)
    return seen


# --- Orders and storage tokens -----------------------------------------------


def _read_orders(state: OvercookedState, kitchen: Kitchen) -> None:
    for index, dish in enumerate(state.visible_orders):
        kitchen.orders[f"o_{index}"] = dish


def _read_tokens(state: OvercookedState, kitchen: Kitchen) -> None:
    """Give each storage kind enough not-yet-existing food for the orders."""
    stored = {tile.storage for tile in kitchen.slots.values() if tile.storage}
    if not stored:
        return
    memo: dict[str, Tally[str]] = {}
    need: Tally[str] = Tally()
    for dish in kitchen.orders.values():
        need.update(_leaf_counts(state, dish, stored, memo, frozenset()))
    for food_name in sorted(stored):
        for index in range(need.get(food_name, 0)):
            kitchen.tokens[f"t_{pddl_name(food_name)}_{index}"] = food_name


def _leaf_counts(
    state: OvercookedState,
    dish: str,
    stored: set[str],
    memo: dict[str, Tally[str]],
    visiting: frozenset[str],
) -> Tally[str]:
    """Count the storage pulls of each kind that making ``dish`` can take.

    Alternative recipes for the same food are combined with an elementwise
    maximum, so whichever route the planner picks has enough tokens.
    """
    if dish in memo:
        return memo[dish]
    if dish in visiting or len(visiting) > _MAX_RECIPE_DEPTH:
        return Tally()
    visiting = visiting | {dish}
    routes: list[Tally[str]] = []
    if dish in stored:
        routes.append(Tally({dish: 1}))
    for recipe in state.cook_recipes:
        if recipe.output == dish:
            routes.append(_leaf_counts(state, recipe.ingredient, stored, memo, visiting))
    for recipe in state.combine_recipes:
        if recipe.output == dish:
            ingredients = list(recipe.ingredients)
            if len(ingredients) == 1:
                ingredients *= 2
            total: Tally[str] = Tally()
            for ingredient in ingredients:
                total.update(_leaf_counts(state, ingredient, stored, memo, visiting))
            routes.append(total)
    result: Tally[str] = Tally()
    for route in routes:
        for name, count in route.items():
            result[name] = max(result[name], count)
    memo[dish] = result
    return result
