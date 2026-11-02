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

The escape: the watch rules for food that burns (food the pan would burn
goes in only while a chef watches it) once froze a hidden kitchen after two
tickets: every later plan went nowhere, and a kitchen where nobody acts
stops the clock, so it sat until the time limit. In a kitchen where those
rules restrict something, when the kitchen still
stands still after ``ESCAPE_ATTEMPTS`` replans, or right after a solve slow
enough to have hit the planner's limit, ``escape`` turns on and problems
are written without those rules, as main did before them. If even that
goes nowhere ``WAIT_STALLS`` times, one plan only lets time pass
(``resting``). The escape ends at the next order change.
"""

from __future__ import annotations

import time
from collections.abc import Collection
from pathlib import Path

from controllers.pddl.interfaces import PDDLProblem, ProblemGenerator
from controllers.pddl.problem_builder import ProblemBuilder
from controllers.pddl.solvers import DEFAULT_SOLVER_TIME_LIMIT_SECONDS
from controllers.pddl.submission.kitchen import (
    Kitchen,
    equipment_symbol,
    kind_symbol,
    read_kitchen,
)
from controllers.pddl.submission.service import (
    Service,
    burns,
    needed,
    plan_service,
    watched_foods,
)
from simulator.entities import Agent, Equipment, Food, OvercookedState
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
# An order change waits up to this many timesteps while a station that cooks
# by itself holds food that burns there (see ``_watching``).
WATCH_STEPS = 8
# Replans in a row that find the kitchen where the last plan started before
# the escape turns on. No run of a fully scored level reached this.
ESCAPE_ATTEMPTS = 2
# A solve this long (wall seconds) that leaves the kitchen where it was also
# turns the escape on: just under the time after which the planner gives up.
# The longest plan seen locally took 12.9 s (slow cores, under load) and
# moved the kitchen.
SLOW_SOLVE = DEFAULT_SOLVER_TIME_LIMIT_SECONDS - 1
# Stalls in a row after the escape before one plan only lets time pass.
WAIT_STALLS = 2


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
        # Timestep at which an order change was first held back.
        self._held_since: int | None = None
        # The escape from the watch rules (see the module docstring): whether
        # it is on, whether the next plan only waits, the longest planning
        # gap since the last replan, and stalls in a row since it turned on.
        self.escape = False
        self.resting = False
        self._solve_seconds = 0.0
        self._escape_stalls = 0

    def to_pddl(
        self,
        environment: Environment,
        controlled_agent_ids: Collection[str] | None = None,
    ) -> PDDLProblem:
        """Return the domain and a problem for ``environment``."""
        return PDDLProblem(
            domain=build_domain(),
            problem=build_problem(
                environment,
                controlled_agent_ids,
                self.attempt,
                watch=not self.escape,
                wait=self.resting,
            ),
        )

    def should_replan(self, environment: Environment) -> bool:
        state = environment.get_first_entity_of_type(OvercookedState)
        now = time.monotonic()
        gap = now - self._last_check
        self._last_check = now
        revision = state.order_revision if state is not None else 0
        if self._revision is None or revision != self._revision:
            first = self._revision is None
            if not first and not self.escape and self._watching(environment, state):
                if self._held_since is None:
                    self._held_since = environment.timestep
                if environment.timestep - self._held_since < WATCH_STEPS:
                    return False
            self._held_since = None
            self._revision = revision
            self._watch(environment.timestep)
            self.escape = False
            self._escape_stalls = 0
            if first:
                self._started_at = environment.timestep
                return False
            self.attempt = 0
            self._started_at = environment.timestep
            return self._replan()
        if environment.timestep != self._timestep or gap > PLANNING_GAP_SECONDS:
            if gap > PLANNING_GAP_SECONDS:
                self._solve_seconds = max(self._solve_seconds, gap)
            self._watch(environment.timestep)
            if self._going_round(environment, state):
                self.attempt += 1
                self._started_at = environment.timestep
                return self._replan()
            return False
        self._still_checks += 1
        if self._still_checks < STALL_CHECKS:
            return False
        # Standing still: the plan is done or stuck.
        resting = False
        if environment.timestep == self._started_at:
            self.attempt += 1
            if self.escape:
                self._escape_stalls += 1
                if self._escape_stalls >= WAIT_STALLS:
                    resting = True
                    self._escape_stalls = 0
            elif (
                self.attempt >= ESCAPE_ATTEMPTS or self._solve_seconds >= SLOW_SOLVE
            ) and self._has_watch_rules(environment, state):
                self.escape = True
        else:
            self.attempt = 0
            self._escape_stalls = 0
        self._started_at = environment.timestep
        self._watch(environment.timestep)
        return self._replan(resting)

    def _replan(self, resting: bool = False) -> bool:
        """Ask for a plan; ``resting`` makes it one that only lets time pass."""
        self.resting = resting
        self._solve_seconds = 0.0
        return True

    def _watching(
        self, environment: Environment, state: OvercookedState | None
    ) -> bool:
        """Return whether a chef may be watching food that burns if left.

        A replan then would hand the scoop to whichever chef the planner
        likes, often one too far away to save the patty, so the plan that
        put it in finishes first.
        """
        if state is None:
            return False
        held: list[tuple[str, str]] = []
        for equipment in environment.get_entities_of_type(Equipment):
            if equipment.cooks_by_itself and equipment.held_item_id:
                food = environment.get_entity_as(equipment.held_item_id, Food)
                if food is not None:
                    held.append((equipment.name, food.name))
        if not held:
            return False
        self_cooking = {name for name, _ in held}
        watched = watched_foods(
            state, self_cooking, needed(state, state.visible_orders)
        )
        return any(pair in watched for pair in held)

    def _has_watch_rules(
        self, environment: Environment, state: OvercookedState | None
    ) -> bool:
        """Return whether the watch rules restrict anything in this kitchen.

        Only then is there something to escape from; elsewhere the escape
        would only add waits and restart the order rotation of ``attempt``.
        """
        if state is None:
            return False
        self_cooking = {
            equipment.name
            for equipment in environment.get_entities_of_type(Equipment)
            if equipment.cooks_by_itself
        }
        if not self_cooking:
            return False
        wanted = needed(state, state.visible_orders)
        return bool(watched_foods(state, self_cooking, wanted))

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
    watch: bool = True,
    wait: bool = False,
) -> str:
    """Describe the next part of the service for the domain in overcooked.pddl.

    ``watch=False`` leaves out the watch rules for food that burns, as main
    did before them; ``wait=True`` only lets time pass. Both come from the
    escape in ``DefaultProblemGenerator.should_replan``.
    """
    kitchen = read_kitchen(environment, controlled_agent_ids)
    state = environment.get_first_entity_of_type(OvercookedState)
    if state is None:
        raise ValueError("The environment has no OvercookedState")
    problem = ProblemBuilder(domain_name="overcooked", problem_name="kitchen")
    if wait:
        describe_rest(kitchen, "wait", problem)
        return problem.to_pddl()
    service = plan_service(kitchen, state, attempt, watch)
    if service.rest is not None:
        describe_rest(kitchen, service.rest, problem)
        return problem.to_pddl()
    describe_kinds(environment, kitchen, service, state, problem, attempt, watch)
    describe_equipment(environment, kitchen, service, state, problem, watch)
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
    watch: bool = True,
) -> None:
    """Declare the food kinds the jobs need, which are raw, and the recipes.

    Food no visible order needs is waste, which chefs may bin. After two
    plans in a row went nowhere, anything may be binned. Food that a station
    cooking by itself would turn into waste burns there if left too long;
    without the watch rules, only what a burn recipe ruins counts.
    """
    names = service.kinds
    wanted = needed(state, [order.dish for order in kitchen.orders])
    for name in sorted(names):
        symbol = problem.add_object(kind_symbol(name), "fkind")
        if state.is_raw_food(name):
            problem.add_fact(f"(raw {symbol})")
        if name not in wanted or attempt >= 2:
            problem.add_fact(f"(waste {symbol})")
    burning = burns(state, kitchen.self_cooking, wanted, illegal=watch)
    for equipment_name, food_name in sorted(burning):
        equipment = equipment_symbol(equipment_name)
        if equipment in kitchen.equipment and food_name in names:
            problem.add_fact(f"(burns {equipment} {kind_symbol(food_name)})")
    for recipe in state.cook_recipes:
        equipment = equipment_symbol(recipe.equipment)
        if equipment not in kitchen.equipment:
            continue
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
    watch: bool = True,
) -> None:
    """Declare equipment kinds: what each accepts and what base it needs.

    A plate takes any food on the way to an order. Other equipment takes
    what the recipes let it take; illegal recipes are never planned. Food
    that burns in a station, or cooks there into food that does, is not
    accepted there: only ``cook_watch`` puts it in, with the chef waiting by
    it. Otherwise a plan could park a patty, raw or grilled, in the pan while
    the chef slices cheese, and the pan would cook it on and burn it.
    Without the watch rules (``watch=False``) the station accepts it again.
    """
    entities = environment.get_entities_of_type(Equipment)
    dishes = needed(state, [job.order.dish for job in service.jobs])
    watched = (
        watched_foods(
            state,
            kitchen.self_cooking,
            needed(state, [order.dish for order in kitchen.orders]),
        )
        if watch
        else set()
    )
    for symbol, name in kitchen.equipment.items():
        problem.add_object(symbol, "ekind")
        for food_name in sorted(service.kinds):
            if name == PLATE_NAME:
                if food_name in dishes:
                    problem.add_fact(f"(accepts {symbol} {kind_symbol(food_name)})")
            elif (name, food_name) not in watched and _accepts(
                state, food_name, name, set()
            ):
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

    No chef may end a plan holding cookware: the controller would never
    replan after it.
    """
    for job in service.jobs:
        symbol = problem.add_object(job.symbol, "job")
        order = problem.add_object(job.order.symbol, "order")
        problem.add_fact(f"(aim {symbol} {order})")
        problem.add_fact(f"(dish {order} {kind_symbol(job.order.dish)})")
        problem.add_goal(f"(served {order})")
        for chef in job.crew:
            problem.add_fact(f"(crew {symbol} {chef})")
        for slot in sorted(job.slots & service.slots):
            problem.add_fact(f"(uses {symbol} {slot})")
        for item in sorted(job.items | service.shared | set(job.tokens)):
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
    """State where an item is: in a chef's hands or on a described tile.

    An item on a tile the problem leaves out gets no place, so it cannot be
    used, rather than a fact about an undeclared object (Fast Downward
    rejects the whole problem for that).
    """
    if item in kitchen.held_by:
        problem.add_fact(f"(holding {kitchen.held_by[item]} {item})")
    elif item in kitchen.on_slot and kitchen.on_slot[item] in service.slots:
        problem.add_fact(f"(at {item} {kitchen.on_slot[item]})")


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
