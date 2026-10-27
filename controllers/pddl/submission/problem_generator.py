"""Describe the kitchen as a PDDL problem for the instance-level domain.

``DefaultPDDLController`` calls ``to_pddl`` in a worker process whenever a
plan is needed. ``kitchen.read_kitchen`` turns the environment into the
objects the domain talks about (chefs, slots, food, vessels, kinds, orders),
and the ``describe_*`` functions below write their objects, facts and goals.
``from_pddl.py`` reads the same ``Kitchen`` back to translate the plan.

``should_replan`` runs in the simulator process. It asks for a new plan
when an order is served or expires (``order_revision`` changes) and when the
kitchen has stood still for a while, which means a chef is waiting on a step
the kitchen no longer allows.
"""

from __future__ import annotations

import time
from collections.abc import Collection
from pathlib import Path

from controllers.pddl.interfaces import PDDLProblem, ProblemGenerator
from controllers.pddl.problem_builder import ProblemBuilder
from controllers.pddl.submission.kitchen import (
    Kitchen,
    equipment_symbol,
    kind_symbol,
    read_kitchen,
)
from simulator.entities import Equipment, OvercookedState
from simulator.environment import Environment

DOMAIN_PATH = Path(__file__).with_name("overcooked.pddl")

# A kitchen whose timestep has not moved for this long, over at least this
# many checks, is stuck: replan from the kitchen as it is.
STALL_SECONDS = 1.0
STALL_CHECKS = 20
# A gap this long between checks means a plan was being computed meanwhile.
PLANNING_GAP_SECONDS = 0.25


class DefaultProblemGenerator(ProblemGenerator):
    """Generate problems and decide when to replan."""

    def __init__(self) -> None:
        self._revision: int | None = None
        self._timestep: int | None = None
        self._still_since = 0.0
        self._still_checks = 0
        self._last_check = 0.0

    def to_pddl(
        self,
        environment: Environment,
        controlled_agent_ids: Collection[str] | None = None,
    ) -> PDDLProblem:
        """Return the domain and a problem for ``environment``."""
        return PDDLProblem(
            domain=build_domain(),
            problem=build_problem(environment, controlled_agent_ids),
        )

    def should_replan(self, environment: Environment) -> bool:
        state = environment.get_first_entity_of_type(OvercookedState)
        now = time.monotonic()
        gap = now - self._last_check
        self._last_check = now
        revision = state.order_revision if state is not None else 0
        if self._revision is None or revision != self._revision:
            first = self._revision is None
            self._revision = revision
            self._watch(environment.timestep, now)
            return not first
        if environment.timestep != self._timestep or gap > PLANNING_GAP_SECONDS:
            self._watch(environment.timestep, now)
            return False
        self._still_checks += 1
        if (
            self._still_checks >= STALL_CHECKS
            and now - self._still_since >= STALL_SECONDS
        ):
            self._watch(environment.timestep, now)
            return True
        return False

    def _watch(self, timestep: int, now: float) -> None:
        self._timestep = timestep
        self._still_since = now
        self._still_checks = 0


def build_domain() -> str:
    """Return the text of overcooked.pddl."""
    return DOMAIN_PATH.read_text(encoding="utf-8")


def build_problem(
    environment: Environment,
    controlled_agent_ids: Collection[str] | None = None,
) -> str:
    """Describe the kitchen as a problem for the domain in overcooked.pddl."""
    kitchen = read_kitchen(environment, controlled_agent_ids)
    state = environment.get_first_entity_of_type(OvercookedState)
    if state is None:
        raise ValueError("The environment has no OvercookedState")
    problem = ProblemBuilder(domain_name="overcooked", problem_name="kitchen")
    describe_kinds(kitchen, state, problem)
    describe_equipment(environment, kitchen, state, problem)
    describe_slots(kitchen, problem)
    describe_vessels(kitchen, problem)
    describe_food(kitchen, problem)
    describe_chefs(kitchen, problem)
    describe_orders(kitchen, problem)
    return problem.to_pddl()


# --- Kitchen description ----------------------------------------------------


def describe_kinds(
    kitchen: Kitchen, state: OvercookedState, problem: ProblemBuilder
) -> None:
    """Declare each food kind, which kinds are raw, and the recipes."""
    for symbol, name in kitchen.kinds.items():
        problem.add_object(symbol, "fkind")
        if state.is_raw_food(name):
            problem.add_fact(f"(raw {symbol})")
    for recipe in state.cook_recipes:
        equipment = equipment_symbol(recipe.equipment)
        if equipment not in kitchen.equipment:
            continue
        problem.add_fact(
            f"(cooks {equipment} {kind_symbol(recipe.ingredient)}"
            f" {kind_symbol(recipe.output)})"
        )
    for recipe in state.combine_recipes:
        ingredients = sorted(recipe.ingredients)
        if len(ingredients) == 1:
            ingredients *= 2
        if len(ingredients) != 2:
            continue
        first, second = (kind_symbol(name) for name in ingredients)
        output = kind_symbol(recipe.output)
        problem.add_fact(f"(combines {first} {second} {output})")
        problem.add_fact(f"(combines {second} {first} {output})")


def describe_equipment(
    environment: Environment,
    kitchen: Kitchen,
    state: OvercookedState,
    problem: ProblemBuilder,
) -> None:
    """Declare equipment kinds: what each accepts, needs and cooks by hand."""
    entities = environment.get_entities_of_type(Equipment)
    plate_names = {
        vessel.name for vessel in kitchen.vessels.values() if vessel.plate
    }
    for symbol, name in kitchen.equipment.items():
        problem.add_object(symbol, "ekind")
        for kind, food_name in kitchen.kinds.items():
            if name in plate_names or state.equipment_accepts(food_name, name):
                problem.add_fact(f"(accepts {symbol} {kind})")
        same = [entity for entity in entities if entity.name == name]
        if not any(entity.cooks_by_itself for entity in same):
            problem.add_fact(f"(manual {symbol})")
        for entity in same:
            if entity.requires:
                problem.add_fact(
                    f"(requires {symbol} {equipment_symbol(entity.requires)})"
                )


def describe_slots(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare each tile a chef acts on and what it is."""
    for slot in kitchen.slots.values():
        if slot.blocked:
            continue
        symbol = problem.add_object(slot.symbol, "slot")
        if slot.counter:
            problem.add_fact(f"(counter {symbol})")
        if slot.equipment is not None:
            problem.add_fact(f"(station {symbol} {equipment_symbol(slot.equipment)})")
        if slot.storage is not None:
            problem.add_fact(f"(storage {symbol} {kind_symbol(slot.storage)})")
        if slot.delivery:
            problem.add_fact(f"(delivery {symbol})")
        if slot.bin:
            problem.add_fact(f"(bin {symbol})")
        if slot.item is None and (slot.counter or slot.equipment is not None):
            problem.add_fact(f"(clear {symbol})")


def describe_vessels(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare plates and other portable equipment, and where they are."""
    holds_food = set(kitchen.in_vessel.values())
    for vessel in kitchen.vessels.values():
        symbol = problem.add_object(vessel.symbol, "vessel")
        problem.add_fact(f"(vkind {symbol} {equipment_symbol(vessel.name)})")
        if vessel.plate:
            problem.add_fact(f"(servable {symbol})")
        if symbol not in holds_food:
            problem.add_fact(f"(empty {symbol})")
        _describe_place(kitchen, symbol, problem)


def describe_food(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare each piece of food, its kind and place, and storage tokens."""
    for symbol, name in kitchen.foods.items():
        problem.add_object(symbol, "food")
        problem.add_fact(f"(kind {symbol} {kind_symbol(name)})")
        if symbol in kitchen.in_vessel:
            problem.add_fact(f"(in {symbol} {kitchen.in_vessel[symbol]})")
        _describe_place(kitchen, symbol, problem)
    for symbol, name in kitchen.tokens.items():
        problem.add_object(symbol, "food")
        problem.add_fact(f"(spawn {symbol} {kind_symbol(name)})")


def describe_chefs(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Declare each chef, whether its hands are free, and what it can reach."""
    holding = set(kitchen.held_by.values())
    for symbol in kitchen.chefs:
        problem.add_object(symbol, "chef")
        if symbol not in holding:
            problem.add_fact(f"(free {symbol})")
        for slot in sorted(kitchen.reach.get(symbol, ())):
            if not kitchen.slots[slot].blocked:
                problem.add_fact(f"(reach {symbol} {slot})")


def describe_orders(kitchen: Kitchen, problem: ProblemBuilder) -> None:
    """Add one goal per visible order, so repeated dishes are served twice."""
    for symbol, dish in kitchen.orders.items():
        problem.add_object(symbol, "order")
        problem.add_fact(f"(dish {symbol} {kind_symbol(dish)})")
        problem.add_goal(f"(served {symbol})")


def _describe_place(kitchen: Kitchen, item: str, problem: ProblemBuilder) -> None:
    if item in kitchen.held_by:
        chef = kitchen.held_by[item]
        if chef in kitchen.chefs:
            problem.add_fact(f"(holding {chef} {item})")
    elif item in kitchen.on_slot:
        slot = kitchen.on_slot[item]
        if not kitchen.slots[slot].blocked:
            problem.add_fact(f"(at {item} {slot})")
