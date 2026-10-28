"""Describe the next part of the service as a PDDL problem.

``DefaultPDDLController`` calls ``to_pddl`` in a worker process whenever a
plan is needed. ``kitchen.read_kitchen`` turns the environment into objects
(chefs, slots, food, vessels, kinds, orders); ``service.plan_service``
decides which orders to plan now, who cooks each one and what each may
use; the ``describe_*`` functions below write only those objects, facts
and goals. ``from_pddl.py`` reads the same ``Kitchen`` back to translate
the plan.

``should_replan`` runs in the simulator process. It asks for a new plan
when an order is served or expires (``order_revision`` changes), and when
the kitchen stands still: no chef acted and nothing cooked for a few
checks in a row. That happens when a plan is finished (every plan ends
with a step that never becomes legal, so the run does not end while orders
remain), when the planner found no plan, or when a chef waits for
something that will not happen. ``attempt`` counts the replans in a row
that found the kitchen exactly where the last plan started; the generator
object is copied to the worker with every request, so ``to_pddl`` sees it
and plans something smaller and different.
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
from controllers.pddl.submission.service import Service, burns, plan_service
from simulator.entities import Agent, Equipment, OvercookedState
from simulator.entities.plate import PLATE_NAME
from simulator.environment import Environment

DOMAIN_PATH = Path(__file__).with_name("overcooked.pddl")

# The kitchen counts as standing still after this many checks in a row with
# the same timestep, none of them right after a plan was computed.
STALL_CHECKS = 3
# A gap this long between checks means a plan was being computed meanwhile.
PLANNING_GAP_SECONDS = 0.25
# Chefs that move for this many timesteps without anything changing hands or
# scoring are going round in circles (two chefs swapping tiles, say).
IDLE_STEPS = 40


class DefaultProblemGenerator(ProblemGenerator):
    """Generate problems and decide when to replan."""

    def __init__(self) -> None:
        self._revision: int | None = None
        self._timestep: int | None = None
        self._still_checks = 0
        self._last_check = 0.0
        # Timestep at which the current plan started, and how many plans in
        # a row did not move the kitchen at all.
        self._started_at: int | None = None
        self.attempt = 0
        # What the chefs hold and the score, and when that last changed.
        self._signature: tuple[object, ...] | None = None
        self._changed_at = 0

    def to_pddl(
        self,
        environment: Environment,
        controlled_agent_ids: Collection[str] | None = None,
    ) -> PDDLProblem:
        """Return the domain and a problem for ``environment``."""
        return PDDLProblem(
            domain=build_domain(),
            problem=build_problem(environment, controlled_agent_ids, self.attempt),
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
            self._watch(environment.timestep)
            if first:
                self._started_at = environment.timestep
                return False
            self.attempt = 0
            self._started_at = environment.timestep
            return True
        if environment.timestep != self._timestep or gap > PLANNING_GAP_SECONDS:
            self._watch(environment.timestep)
            if self._going_round(environment, state):
                self.attempt += 1
                self._started_at = environment.timestep
                return True
            return False
        self._still_checks += 1
        if self._still_checks < STALL_CHECKS:
            return False
        # Standing still: the plan is done or stuck.
        if environment.timestep == self._started_at:
            self.attempt += 1
        else:
            self.attempt = 0
        self._started_at = environment.timestep
        self._watch(environment.timestep)
        return True

    def _going_round(
        self, environment: Environment, state: OvercookedState | None
    ) -> bool:
        """Return whether time passes but nothing changes hands or scores."""
        signature = (
            state.score if state is not None else 0,
            *sorted(
                (agent.id, agent.held_item_id)
                for agent in environment.get_entities_of_type(Agent)
            ),
        )
        if signature != self._signature:
            self._signature = signature
            self._changed_at = environment.timestep
            return False
        if environment.timestep - self._changed_at >= IDLE_STEPS:
            self._changed_at = environment.timestep
            return True
        return False

    def _watch(self, timestep: int) -> None:
        self._timestep = timestep
        self._still_checks = 0


def build_domain() -> str:
    """Return the text of overcooked.pddl."""
    return DOMAIN_PATH.read_text(encoding="utf-8")


def build_problem(
    environment: Environment,
    controlled_agent_ids: Collection[str] | None = None,
    attempt: int = 0,
) -> str:
    """Describe the next part of the service for the domain in overcooked.pddl."""
    kitchen = read_kitchen(environment, controlled_agent_ids)
    state = environment.get_first_entity_of_type(OvercookedState)
    if state is None:
        raise ValueError("The environment has no OvercookedState")
    service = plan_service(kitchen, state, attempt)
    problem = ProblemBuilder(domain_name="overcooked", problem_name="kitchen")
    if service.rest is not None:
        describe_rest(kitchen, service.rest, problem)
        return problem.to_pddl()
    describe_kinds(environment, kitchen, service, state, problem, attempt)
    describe_equipment(environment, kitchen, service, state, problem)
    describe_slots(kitchen, service, problem)
    describe_vessels(kitchen, service, problem)
    describe_food(kitchen, service, problem)
    describe_chefs(kitchen, service, problem)
    describe_jobs(kitchen, service, problem)
    return problem.to_pddl()


# --- Kitchen description ----------------------------------------------------


def describe_rest(kitchen: Kitchen, rest: str, problem: ProblemBuilder) -> None:
    """With nothing to cook, let the first chef wait, or stop the service."""
    if not kitchen.chefs or not kitchen.slots:
        return
    chef = next(iter(kitchen.chefs))
    slot = min(
        kitchen.slots,
        key=lambda s: (kitchen.distance[chef].get(s, 10**6), s),
    )
    problem.add_object(chef, "chef")
    problem.add_object(slot, "slot")
    problem.add_fact(f"(reach {chef} {slot})")
    problem.add_fact(f"({'resting' if rest == 'wait' else 'closing'} {chef})")
    problem.add_goal("(rested)")


def describe_kinds(
    environment: Environment,
    kitchen: Kitchen,
    service: Service,
    state: OvercookedState,
    problem: ProblemBuilder,
    attempt: int = 0,
) -> None:
    """Declare the food kinds the jobs need, which are raw, and the recipes.

    Food no visible order needs is waste, which chefs may bin. After two
    plans in a row went nowhere, anything may be binned. Food that a station
    cooking by itself would turn into waste burns there if left too long.
    """
    names = service.kinds
    wanted: set[str] = set()
    for order in kitchen.orders:
        wanted |= _ancestors(state, order.dish)
    for name in sorted(names):
        symbol = problem.add_object(kind_symbol(name), "fkind")
        if state.is_raw_food(name):
            problem.add_fact(f"(raw {symbol})")
        if name not in wanted or attempt >= 2:
            problem.add_fact(f"(waste {symbol})")
    burning = burns(state, kitchen, wanted)
    for recipe in state.cook_recipes:
        equipment = equipment_symbol(recipe.equipment)
        if equipment not in kitchen.equipment:
            continue
        if (
            recipe.equipment,
            recipe.ingredient,
        ) in burning and recipe.ingredient in names:
            problem.add_fact(f"(burns {equipment} {kind_symbol(recipe.ingredient)})")
        if recipe.ingredient not in names or recipe.output not in names:
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
        if not set(ingredients) <= names or recipe.output not in names:
            continue
        first, second = (kind_symbol(name) for name in ingredients)
        output = kind_symbol(recipe.output)
        problem.add_fact(f"(combines {first} {second} {output})")
        problem.add_fact(f"(combines {second} {first} {output})")


def describe_equipment(
    environment: Environment,
    kitchen: Kitchen,
    service: Service,
    state: OvercookedState,
    problem: ProblemBuilder,
) -> None:
    """Declare equipment kinds: what each accepts and what base it needs.

    A plate takes any food on the way to an order. Other equipment takes
    what the recipes let it take; illegal recipes are never planned.
    """
    entities = environment.get_entities_of_type(Equipment)
    dishes = set()
    for job in service.jobs:
        dishes |= _ancestors(state, job.target)
    for symbol, name in kitchen.equipment.items():
        problem.add_object(symbol, "ekind")
        for food_name in sorted(service.kinds):
            if name == PLATE_NAME:
                if food_name in dishes:
                    problem.add_fact(f"(accepts {symbol} {kind_symbol(food_name)})")
            elif _accepts(state, food_name, name, set()):
                problem.add_fact(f"(accepts {symbol} {kind_symbol(food_name)})")
        for entity in entities:
            if entity.name == name and entity.requires:
                problem.add_fact(
                    f"(requires {symbol} {equipment_symbol(entity.requires)})"
                )


def describe_slots(kitchen: Kitchen, service: Service, problem: ProblemBuilder) -> None:
    """Declare the tiles worth describing and what each one is."""
    for symbol in sorted(service.slots):
        slot = kitchen.slots[symbol]
        problem.add_object(symbol, "slot")
        if slot.counter:
            problem.add_fact(f"(counter {symbol})")
        if slot.equipment is not None:
            problem.add_fact(f"(station {symbol} {equipment_symbol(slot.equipment)})")
        if slot.storage is not None and slot.storage in service.kinds:
            problem.add_fact(f"(storage {symbol} {kind_symbol(slot.storage)})")
        if slot.delivery:
            problem.add_fact(f"(delivery {symbol})")
        if slot.bin:
            problem.add_fact(f"(bin {symbol})")
        if slot.sink:
            problem.add_fact(f"(sink {symbol})")
        if slot.dispenser:
            problem.add_fact(f"(dispenser {symbol})")
        if slot.item is None and (slot.counter or slot.equipment is not None):
            problem.add_fact(f"(clear {symbol})")


def describe_vessels(
    kitchen: Kitchen, service: Service, problem: ProblemBuilder
) -> None:
    """Declare the plates and cookware the jobs use, and where they are.

    Plates still in a dispenser are chained in the order they come out.
    """
    used = _described(service)
    holds_food = {vessel for food, vessel in kitchen.in_vessel.items() if food in used}
    stacks: dict[str, list[str]] = {}
    for vessel in kitchen.vessels.values():
        if vessel.symbol not in used:
            continue
        symbol = problem.add_object(vessel.symbol, "vessel")
        problem.add_fact(f"(vkind {symbol} {equipment_symbol(vessel.name)})")
        if vessel.plate:
            problem.add_fact(f"(servable {symbol})")
        if symbol not in holds_food:
            problem.add_fact(f"(empty {symbol})")
        problem.add_fact(f"({'dirty' if vessel.dirty else 'clean'} {symbol})")
        if vessel.dispenser is not None:
            stacks.setdefault(vessel.dispenser, []).append(symbol)
        else:
            _describe_place(kitchen, service, symbol, problem)
    for dispenser, plates in stacks.items():
        plates.sort(key=lambda symbol: kitchen.vessels[symbol].position)
        end = problem.add_object(f"v_end_{dispenser}", "vessel")
        problem.add_fact(f"(top {dispenser} {plates[0]})")
        for first, second in zip(plates, [*plates[1:], end], strict=True):
            problem.add_fact(f"(next {first} {second})")


def describe_food(kitchen: Kitchen, service: Service, problem: ProblemBuilder) -> None:
    """Declare the food the jobs use, its kind and place, and storage tokens."""
    used = _described(service)
    for symbol, name in kitchen.foods.items():
        if symbol not in used:
            continue
        problem.add_object(symbol, "food")
        problem.add_fact(f"(kind {symbol} {kind_symbol(name)})")
        if symbol in kitchen.in_vessel:
            problem.add_fact(f"(in {symbol} {kitchen.in_vessel[symbol]})")
        _describe_place(kitchen, service, symbol, problem)
    for job in service.jobs:
        for symbol, name in job.tokens.items():
            problem.add_object(symbol, "food")
            problem.add_fact(f"(spawn {symbol} {kind_symbol(name)})")
        if job.prep is not None:
            problem.add_object(job.prep.token, "food")
            problem.add_fact(f"(spawn {job.prep.token} {kind_symbol(job.prep.raw)})")


def describe_chefs(kitchen: Kitchen, service: Service, problem: ProblemBuilder) -> None:
    """Declare the chefs in a crew, whether their hands are free, and reach."""
    holding = set(kitchen.held_by.values())
    crews = {chef for job in service.jobs for chef in job.crew}
    for symbol in kitchen.chefs:
        if symbol not in crews:
            continue
        problem.add_object(symbol, "chef")
        if symbol not in holding:
            problem.add_fact(f"(free {symbol})")
        for slot in sorted(kitchen.reach.get(symbol, ()) & service.slots):
            problem.add_fact(f"(reach {symbol} {slot})")


def describe_jobs(kitchen: Kitchen, service: Service, problem: ProblemBuilder) -> None:
    """Declare each job: its crew, what it owns, and its order as a goal.

    A prep job has no order: its goal is the portion, cooked, in its
    cookware. No chef may end a plan holding cookware: the controller would
    never replan after it.
    """
    for job in service.jobs:
        symbol = problem.add_object(job.symbol, "job")
        if job.order is not None:
            order = problem.add_object(job.order.symbol, "order")
            problem.add_fact(f"(aim {symbol} {order})")
            problem.add_fact(f"(dish {order} {kind_symbol(job.order.dish)})")
            problem.add_goal(f"(served {order})")
        for chef in job.crew:
            problem.add_fact(f"(crew {symbol} {chef})")
        for slot in sorted(job.slots & service.slots):
            problem.add_fact(f"(uses {symbol} {slot})")
        owned = job.items | service.shared | set(job.tokens)
        if job.prep is not None:
            owned.add(job.prep.token)
            problem.add_goal(f"(kind {job.prep.token} {kind_symbol(job.prep.food)})")
            problem.add_goal(f"(in {job.prep.token} {job.prep.vessel})")
        for item in sorted(owned):
            problem.add_fact(f"(owns {symbol} {item})")
    crews = {chef for job in service.jobs for chef in job.crew}
    for vessel in kitchen.vessels.values():
        if vessel.plate or vessel.symbol not in _described(service):
            continue
        for chef in sorted(crews):
            problem.add_goal(f"(not (holding {chef} {vessel.symbol}))")


def _described(service: Service) -> set[str]:
    items = set(service.shared)
    for job in service.jobs:
        items |= job.items
    return items


def _describe_place(
    kitchen: Kitchen, service: Service, item: str, problem: ProblemBuilder
) -> None:
    if item in kitchen.held_by:
        problem.add_fact(f"(holding {kitchen.held_by[item]} {item})")
    elif item in kitchen.on_slot:
        problem.add_fact(f"(at {item} {kitchen.on_slot[item]})")


def _ancestors(state: OvercookedState, dish: str) -> set[str]:
    seen: set[str] = set()
    stack = [dish]
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


def _accepts(
    state: OvercookedState, food_name: str, equipment_name: str, seen: set[str]
) -> bool:
    """Return whether empty equipment takes this food under the legal recipes.

    It does if it cooks the food, or the food combines into something it
    takes (a mixing bowl takes egg because batter ingredients go in it).
    """
    if food_name in seen:
        return False
    seen.add(food_name)
    if state.can_cook_with(food_name, equipment_name):
        return True
    return any(
        _accepts(state, recipe.output, equipment_name, seen)
        for recipe in state.combine_recipes
        if food_name in recipe.ingredients
    )
