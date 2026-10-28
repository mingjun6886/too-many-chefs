"""Read the kitchen into the objects the PDDL problem talks about.

The problem generator and the plan translator both build a ``Kitchen`` from
the same ``Environment`` snapshot, so they agree on every symbol: the
generator writes facts about ``Kitchen`` objects, and the translator turns
the symbols in a plan back into chef ids, tiles and catalog names.

The model is instance-level: each chef, each tile a chef can act on, each
piece of food, each plate or pan and each visible order is one PDDL object.
Food keeps its identity while it is cooked or combined; its current kind is
a fluent. Storages and plate dispensers hand out new things, which the
problem describes as "tokens" that do not exist yet (see ``service.py``).

Names:

- chefs ``a_<id>``; tiles ``loc_<x>_<y>``; food ``f_<id>``; food tokens
  ``t_<job>_<kind>_<n>``; vessels (plates and other portable equipment)
  ``v_<id>``; plates still in a dispenser ``v_d<x>_<y>_<n>``; food kinds
  ``k_<name>``; equipment kinds ``e_<name>``; orders ``o_<n>``; jobs
  ``j_<n>``.
"""

from __future__ import annotations

import re
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
    PlateDispenser,
    Sink,
    Storage,
)
from simulator.entities.plate import PLATE_NAME
from simulator.environment import Environment
from simulator.types import Location

# How many plates of one dispenser the model can hand out in one plan.
DISPENSER_TOKENS = 4
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
    """A tile a chef acts on: counter, station, storage, delivery, bin, sink
    or plate dispenser."""

    symbol: str
    location: Location
    # Bare counter: takes raw food or a vessel.
    counter: bool = False
    # Name of the fixed equipment on the tile (cutboard, stove, ...), if any.
    equipment: str | None = None
    storage: str | None = None
    delivery: bool = False
    bin: bool = False
    sink: bool = False
    dispenser: bool = False
    # The symbol of the item on the tile, if the model can describe it.
    item: str | None = None
    # Whether something the model cannot describe occupies the tile.
    blocked: bool = False


@dataclass
class Vessel:
    """A portable piece of equipment: a plate, pan, pot or mixing bowl."""

    symbol: str
    entity_id: str | None
    name: str
    plate: bool
    dirty: bool = False
    # The base it works on (a stove for a pan), if any.
    requires: str | None = None
    # For a plate still in a dispenser: the dispenser's slot and its place
    # in the stack (0 comes out first).
    dispenser: str | None = None
    position: int = 0


@dataclass
class Order:
    symbol: str
    dish: str
    index: int
    reward: int
    # Last timestep it can be served, or None when it never expires.
    deadline: int | None
    # When a dish of this kind must be served for no order of it to expire.
    # A delivery serves the first order in the queue with that dish, not the
    # one a plan meant, so this is the earliest deadline among this order and
    # the later ones for the same dish.
    due: int | None = None


@dataclass
class Kitchen:
    chefs: dict[str, str] = field(default_factory=dict)  # symbol -> agent id
    chef_tiles: dict[str, Location] = field(default_factory=dict)
    slots: dict[str, Slot] = field(default_factory=dict)
    vessels: dict[str, Vessel] = field(default_factory=dict)
    foods: dict[str, str] = field(default_factory=dict)  # symbol -> food name
    kinds: dict[str, str] = field(default_factory=dict)  # symbol -> food name
    equipment: dict[str, str] = field(default_factory=dict)  # symbol -> name
    orders: list[Order] = field(default_factory=list)
    # Where each described item is: chef holding it, slot it sits on, or the
    # vessel it is in. Values are symbols.
    held_by: dict[str, str] = field(default_factory=dict)
    on_slot: dict[str, str] = field(default_factory=dict)
    in_vessel: dict[str, str] = field(default_factory=dict)
    # Static reachability: chef symbol -> slot symbols it can act on.
    reach: dict[str, set[str]] = field(default_factory=dict)
    # Walking distance from each chef to each slot it reaches.
    distance: dict[str, dict[str, int]] = field(default_factory=dict)
    # Order rules.
    strict: bool = False
    sequential: bool = False
    timestep: int = 0

    def slot_at(self, location: Location) -> Slot | None:
        return self.slots.get(tile_symbol(location))

    def place_of(self, item: str) -> str | None:
        """Return the slot an item (or the vessel holding it) stands on."""
        if item in self.in_vessel:
            item = self.in_vessel[item]
        return self.on_slot.get(item)

    def holder_of(self, item: str) -> str | None:
        """Return the chef holding an item, or the vessel holding it."""
        if item in self.in_vessel:
            item = self.in_vessel[item]
        return self.held_by.get(item)


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
    kitchen = Kitchen(timestep=environment.timestep)
    _read_kinds(environment, state, kitchen)
    _read_slots(environment, kitchen)
    _read_items(environment, kitchen)
    _read_dispensers(environment, kitchen)
    _read_chefs(environment, controlled_agent_ids, kitchen)
    _read_orders(state, kitchen)
    return kitchen


# --- Kinds --------------------------------------------------------------------


def _read_kinds(
    environment: Environment, state: OvercookedState, kitchen: Kitchen
) -> None:
    names: list[str] = list(state.food_definitions)
    names += [food.name for food in environment.get_entities_of_type(Food)]
    names += [
        storage.food_name for storage in environment.get_entities_of_type(Storage)
    ]
    names += list(state.visible_orders)
    for recipe in state.cook_recipes:
        names += [recipe.ingredient, recipe.output]
    for recipe in state.combine_recipes:
        names += [*recipe.ingredients, recipe.output]
    for name in names:
        if name:
            kitchen.kinds.setdefault(kind_symbol(name), name)

    equipment_names: list[str] = [PLATE_NAME]
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
    for sink in environment.get_entities_of_type(Sink):
        if sink.x is not None and sink.y is not None:
            slot((sink.x, sink.y)).sink = True
    for dispenser in environment.get_entities_of_type(PlateDispenser):
        if dispenser.x is not None and dispenser.y is not None:
            slot((dispenser.x, dispenser.y)).dispenser = True
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
        symbol = "v_" + pddl_name(equipment.id)
        vessel = Vessel(
            symbol=symbol,
            entity_id=equipment.id,
            name=equipment.name,
            plate=isinstance(equipment, Plate),
            dirty=isinstance(equipment, Plate) and equipment.dirty,
            requires=equipment.requires,
        )
        if equipment.id in holder_of:
            kitchen.vessels[symbol] = vessel
            kitchen.held_by[symbol] = "a_" + pddl_name(holder_of[equipment.id].id)
        elif equipment.x is not None and equipment.y is not None:
            tile = kitchen.slot_at((equipment.x, equipment.y))
            if tile is None or tile.item is not None:
                if tile is not None:
                    tile.blocked = True
                continue
            kitchen.vessels[symbol] = vessel
            tile.item = symbol
            kitchen.on_slot[symbol] = tile.symbol

    for food in environment.get_entities_of_type(Food):
        symbol = "f_" + pddl_name(food.id)
        kitchen.kinds.setdefault(kind_symbol(food.name), food.name)
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
        if (
            container is not None
            and container.x is not None
            and container.y is not None
        ):
            location = (container.x, container.y)
        elif food.id in counters_by_item:
            counter = counters_by_item[food.id]
            if counter.x is not None and counter.y is not None:
                location = (counter.x, counter.y)
        if location is None:
            continue
        tile = kitchen.slot_at(location)
        if tile is None or tile.item is not None:
            if tile is not None:
                tile.blocked = True
            continue
        kitchen.foods[symbol] = food.name
        tile.item = symbol
        kitchen.on_slot[symbol] = tile.symbol


def _read_dispensers(environment: Environment, kitchen: Kitchen) -> None:
    """Describe the plates still stacked in each dispenser, clean ones first."""
    for dispenser in environment.get_entities_of_type(PlateDispenser):
        if dispenser.x is None or dispenser.y is None:
            continue
        tile = kitchen.slot_at((dispenser.x, dispenser.y))
        if tile is None:
            continue
        if dispenser.infinite_supply:
            stack = [False] * DISPENSER_TOKENS
        else:
            stack = [False] * dispenser.plate_count + [
                True
            ] * dispenser.dirty_plate_count
        for position, dirty in enumerate(stack[:DISPENSER_TOKENS]):
            symbol = f"v_d{dispenser.x}_{dispenser.y}_{position}"
            kitchen.vessels[symbol] = Vessel(
                symbol=symbol,
                entity_id=None,
                name=PLATE_NAME,
                plate=True,
                dirty=dirty,
                dispenser=tile.symbol,
                position=position,
            )


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
        kitchen.chef_tiles[symbol] = (agent.x, agent.y)
        floor = _floor_distances(blocked, inside, (agent.x, agent.y))
        reach: set[str] = set()
        distance: dict[str, int] = {}
        for tile in kitchen.slots.values():
            steps = [
                floor[(tile.location[0] + dx, tile.location[1] + dy)]
                for dx, dy in _STEPS
                if (tile.location[0] + dx, tile.location[1] + dy) in floor
            ]
            if steps:
                reach.add(tile.symbol)
                distance[tile.symbol] = min(steps)
        kitchen.reach[symbol] = reach
        kitchen.distance[symbol] = distance


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


def _floor_distances(
    blocked: set[Location], inside: Callable[[int, int], bool], start: Location
) -> dict[Location, int]:
    """Return the walking distance to every floor tile reachable from start."""
    seen = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _STEPS:
            step = (x + dx, y + dy)
            if step in seen or step in blocked or not inside(*step):
                continue
            seen[step] = seen[(x, y)] + 1
            queue.append(step)
    return seen


# --- Orders ---------------------------------------------------------------------


def _read_orders(state: OvercookedState, kitchen: Kitchen) -> None:
    queue = state.order_queue
    kitchen.strict = bool(queue.strict_ordering)
    kitchen.sequential = queue.order_reveal == "sequential"
    for index, entry in enumerate(queue.visible):
        deadline = (
            entry.revealed_at + entry.time_limit
            if entry.time_limit is not None
            else None
        )
        kitchen.orders.append(
            Order(
                symbol=f"o_{index}",
                dish=entry.name,
                index=index,
                reward=entry.reward,
                deadline=deadline,
            )
        )
    # The k-th delivery of a dish serves the k-th order for it in the queue,
    # so it must arrive by the earliest deadline from that order on.
    earliest: dict[str, int] = {}
    for order in reversed(kitchen.orders):
        due = earliest.get(order.dish)
        if order.deadline is not None and (due is None or order.deadline < due):
            due = order.deadline
        if due is not None:
            earliest[order.dish] = due
        order.due = due
