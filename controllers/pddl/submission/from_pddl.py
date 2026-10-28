"""Translate plans for the instance-level domain into chef actions.

``DefaultFromPDDL`` is built with the environment the plan was made for. It
reads the same ``Kitchen`` as the problem generator, so every symbol in the
plan maps back to a chef id, a tile, or a catalog name.

Every action in overcooked.pddl names the chef first, the slot it acts on
second and (except ``wait`` and ``stop``) the job last; the slot's tile is
where the chef acts. A step becomes zero or more mutations with their
``expected_*`` fields filled in from the step's own arguments, so a chef
waits rather than acting on the wrong thing:

- ``cook`` at a station that needs pressing becomes one Cook per step of its
  cook time (fewer for food already part-way done); at a station that cooks
  by itself it becomes nothing, and the next step that takes the food out
  names the cooked food, so the chef waits there until it is ready. The
  presses go to the chef who put the food on the station earlier in the
  plan: the planner does not model walking, so it may hand them to a chef
  on the far side of the kitchen while the one at the station stands idle.
- ``wait`` becomes a few turns on the spot, so time passes; ``stop`` becomes
  nothing at all.

Every plan except a ``stop`` ends with a step that never becomes legal. The
chef stands still on it, the kitchen stops moving, and ``should_replan``
asks for the next plan. Without it, an empty queue would end the run while
orders are still open.
"""

from __future__ import annotations

from collections.abc import Callable

from controllers.pddl.interfaces import FromPDDL, PDDLPlanStep
from controllers.pddl.submission.kitchen import Kitchen, Slot, read_kitchen
from simulator.entities import Agent, Equipment, Food, get_game_objects_of_type_at
from simulator.entities.plate import PLATE_NAME
from simulator.environment import Environment
from simulator.mutations import (
    Combine,
    Cook,
    Deliver,
    Discard,
    MutationWithLocation,
    PickUp,
    Place,
    TakeFromStorage,
    TurnAgent,
    Wash,
)
from simulator.mutations.agent_mutation import AgentMutation
from simulator.mutations.cooking import cook_presses_left

# Steps a ``wait`` lets pass.
WAIT_STEPS = 3
# Names no plate has, so the closing step never becomes legal.
FINISHED = "(plan finished)"
# Actions without a job parameter.
_NO_JOB = {"wait", "stop"}


class DefaultFromPDDL(FromPDDL):
    """Translate a symbolic plan into simulator mutations."""

    def __init__(self, environment: Environment) -> None:
        self._environment = environment
        self._kitchen: Kitchen | None = None
        # Stations and vessels food was put into earlier in this plan.
        self._filled: set[str] = set()
        # The chef who last put food on each station in this plan.
        self._loaders: dict[str, str] = {}

    @property
    def kitchen(self) -> Kitchen:
        if self._kitchen is None:
            self._kitchen = read_kitchen(self._environment)
        return self._kitchen

    def from_pddl(self, steps: list[PDDLPlanStep]) -> list[MutationWithLocation]:
        """Translate all steps in plan order, then close the plan."""
        self._filled = set()
        self._loaders = {}
        mutations: list[MutationWithLocation] = []
        for step in steps:
            mutations += self.translate(step)
        if any(step.name == "stop" for step in steps):
            return mutations
        return [*mutations, self._closing_step(mutations)]

    def translate(self, step: PDDLPlanStep) -> list[MutationWithLocation]:
        handler = _HANDLERS.get(step.name)
        if handler is None:
            raise ValueError(f"No translation for this action yet: {step}")
        if len(step.parameters) < 2:
            raise ValueError(f"Step names no chef and slot: {step}")
        chef_id = self.chef_id(step.parameters[0])
        slot = self.slot(step.parameters[1])
        if step.name in {"load", "load_mix", "pour_station"}:
            self._loaders[slot.symbol] = chef_id
        elif step.name == "cook":
            chef_id = self._loaders.get(slot.symbol, chef_id)
        args = step.parameters[2:]
        if step.name not in _NO_JOB:
            args = args[:-1]
        try:
            result = handler(self, chef_id, slot, tuple(args))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot translate {step}: {error}") from error
        if step.name not in {"cook", "cook_vessel", "cook_watch", "cook_watch_vessel"}:
            self._filled.add(slot.symbol)
            self._filled.update(arg for arg in args if arg.startswith("v_"))
        if isinstance(result, AgentMutation):
            result = [result]
        return [
            MutationWithLocation(mutation=mutation, location=slot.location)
            for mutation in result
        ]

    def _closing_step(
        self, mutations: list[MutationWithLocation]
    ) -> MutationWithLocation:
        """Return a step that keeps a chef standing still until the replan."""
        if mutations:
            last = mutations[-1]
            agent_id = getattr(last.mutation, "agent_id", None)
            if agent_id is not None and last.location is not None:
                return MutationWithLocation(
                    mutation=Wash(
                        agent_id=agent_id, expected_held_equipment_name=FINISHED
                    ),
                    location=last.location,
                )
        if not self.kitchen.chefs or not self.kitchen.slots:
            raise ValueError("The kitchen has no chef or no tile to wait at")
        chef = next(iter(self.kitchen.chefs))
        distance = self.kitchen.distance.get(chef, {})
        slot = min(self.kitchen.slots, key=lambda s: (distance.get(s, 10**6), s))
        return MutationWithLocation(
            mutation=Wash(
                agent_id=self.kitchen.chefs[chef],
                expected_held_equipment_name=FINISHED,
            ),
            location=self.kitchen.slots[slot].location,
        )

    # --- Resolving symbols -----------------------------------------------------

    def chef_id(self, symbol: str) -> str:
        if symbol not in self.kitchen.chefs:
            raise ValueError(f"Unknown chef in plan: {symbol}")
        return self.kitchen.chefs[symbol]

    def slot(self, symbol: str) -> Slot:
        if symbol not in self.kitchen.slots:
            raise ValueError(f"Unknown slot in plan: {symbol}")
        return self.kitchen.slots[symbol]

    def food(self, kind: str) -> str:
        if kind not in self.kitchen.kinds:
            raise ValueError(f"Unknown food kind in plan: {kind}")
        return self.kitchen.kinds[kind]

    def equipment(self, symbol: str) -> str:
        if symbol not in self.kitchen.equipment:
            raise ValueError(f"Unknown equipment kind in plan: {symbol}")
        return self.kitchen.equipment[symbol]

    def vessel(self, symbol: str) -> str:
        if symbol not in self.kitchen.vessels:
            raise ValueError(f"Unknown vessel in plan: {symbol}")
        return self.kitchen.vessels[symbol].name

    def _presses(self, station: Equipment | None, key: str, kind_in: str) -> int:
        """Return how many Cook presses a station needs for ``kind_in``.

        Zero for a station that cooks by itself; otherwise its cook time, or
        what is left of it for food already part-way done.
        """
        if station is None:
            return 1
        if station.cooks_by_itself:
            return 0
        if key not in self._filled and station.held_item_id is not None:
            held = self._environment.get_entity_as(station.held_item_id, Food)
            if held is not None and held.name == kind_in:
                return cook_presses_left(self._environment, station)
        return station.cook_time

    # --- One method per action, given the arguments after chef and slot ----------

    def take(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _food, kind = args
        name = self.food(kind)
        return TakeFromStorage(
            agent_id=chef, expected_input_name=name, expected_output_name=name
        )

    def take_into(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ekind, _food, kind = args
        name = self.food(kind)
        return TakeFromStorage(
            agent_id=chef,
            expected_input_name=name,
            expected_held_equipment_name=self.vessel(vessel),
            expected_output_name=name,
        )

    def take_mix(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ekind, _food, _kg, kind, out = args
        return TakeFromStorage(
            agent_id=chef,
            expected_input_name=self.food(kind),
            expected_held_equipment_name=self.vessel(vessel),
            expected_output_name=self.food(out),
        )

    def take_plate(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _next = args
        return TakeFromStorage(
            agent_id=chef,
            expected_input_name=PLATE_NAME,
            expected_output_name=self.vessel(vessel),
        )

    def pick_up(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _food, kind = args
        name = self.food(kind)
        return PickUp(
            agent_id=chef, expected_input_name=name, expected_output_name=name
        )

    def pick_up_vessel(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> AgentMutation:
        (vessel,) = args
        return PickUp(agent_id=chef, expected_held_equipment_name=self.vessel(vessel))

    def pick_up_full(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> AgentMutation:
        vessel, _food, kind = args
        return PickUp(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_equipment_contents_name=self.food(kind),
        )

    def put_food(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _food, kind = args
        return Place(agent_id=chef, expected_held_food=self.food(kind))

    def put_vessel(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        (vessel,) = args
        return Place(agent_id=chef, expected_held_equipment_name=self.vessel(vessel))

    def mount(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ekind, base = args
        return Place(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_target_equipment_name=self.equipment(base),
        )

    def load(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        ekind, _food, kind = args
        name = self.food(kind)
        return Combine(
            agent_id=chef,
            expected_held_food=name,
            expected_target_equipment_name=self.equipment(ekind),
            expected_output_name=name,
        )

    def load_mix(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        ekind, _f, kf, _g, kg, out = args
        return Combine(
            agent_id=chef,
            expected_held_food=self.food(kf),
            expected_target_equipment_name=self.equipment(ekind),
            expected_target_equipment_contents_name=self.food(kg),
            expected_output_name=self.food(out),
        )

    grab_mix = load_mix

    def fill(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ekind, _food, kind = args
        name = self.food(kind)
        return Combine(
            agent_id=chef,
            expected_held_food=name,
            expected_target_equipment_name=self.vessel(vessel),
            expected_output_name=name,
        )

    def fill_mix(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ekind, _f, kf, _g, kg, out = args
        return Combine(
            agent_id=chef,
            expected_held_food=self.food(kf),
            expected_target_equipment_name=self.vessel(vessel),
            expected_target_equipment_contents_name=self.food(kg),
            expected_output_name=self.food(out),
        )

    grab_from_vessel = fill_mix

    def merge(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _f, kf, _g, kg, out = args
        return Combine(
            agent_id=chef,
            expected_held_food=self.food(kf),
            expected_target_food_name=self.food(kg),
            expected_output_name=self.food(out),
        )

    def collect(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ekind, _food, kind = args
        name = self.food(kind)
        return Combine(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_output_name=name,
            **_target_food(slot, name),
        )

    def collect_mix(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> AgentMutation:
        vessel, _ekind, _g, kg, _f, kf, out = args
        return Combine(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_equipment_contents_name=self.food(kg),
            expected_output_name=self.food(out),
            **_target_food(slot, self.food(kf)),
        )

    def collect_mix_station(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> AgentMutation:
        _se, *rest = args
        return self.collect_mix(chef, slot, tuple(rest))

    def pour(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _food, kind, target, _ew = args
        name = self.food(kind)
        return Combine(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_equipment_contents_name=name,
            expected_target_equipment_name=self.vessel(target),
            expected_output_name=name,
        )

    def pour_mix(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _f, kf, target, _ew, _g, kg, out = args
        return Combine(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_equipment_contents_name=self.food(kf),
            expected_target_equipment_name=self.vessel(target),
            expected_target_equipment_contents_name=self.food(kg),
            expected_output_name=self.food(out),
        )

    def pour_station(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> AgentMutation:
        ekind, vessel, _food, kind = args
        name = self.food(kind)
        return Combine(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_equipment_contents_name=name,
            expected_target_equipment_name=self.equipment(ekind),
            expected_output_name=name,
        )

    def scoop(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ev, target, _food, kind = args
        name = self.food(kind)
        return Combine(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_target_equipment_name=self.vessel(target),
            expected_target_equipment_contents_name=name,
            expected_output_name=name,
        )

    def scoop_mix(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _ev, _g, kg, target, _ew, _f, kf, out = args
        return Combine(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_equipment_contents_name=self.food(kg),
            expected_target_equipment_name=self.vessel(target),
            expected_target_equipment_contents_name=self.food(kf),
            expected_output_name=self.food(out),
        )

    def cook(self, chef: str, slot: Slot, args: tuple[str, ...]) -> list[AgentMutation]:
        ekind, _food, kind_in, kind_out = args
        name = self.equipment(ekind)
        station = next(
            (
                equipment
                for equipment in get_game_objects_of_type_at(
                    self._environment, *slot.location, Equipment
                )
                if not equipment.portable and equipment.name == name
            ),
            None,
        )
        presses = self._presses(station, slot.symbol, self.food(kind_in))
        press = Cook(
            agent_id=chef,
            expected_held_food=self.food(kind_in),
            expected_input_name=name,
            expected_output_name=self.food(kind_out),
        )
        return [press] * presses

    def cook_vessel(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> list[AgentMutation]:
        vessel, _ekind, _base, _food, kind_in, kind_out = args
        entity_id = (
            self.kitchen.vessels[vessel].entity_id
            if vessel in self.kitchen.vessels
            else None
        )
        station = (
            self._environment.get_entity_as(entity_id, Equipment)
            if entity_id is not None
            else None
        )
        presses = self._presses(station, vessel, self.food(kind_in))
        press = Cook(
            agent_id=chef,
            expected_held_food=self.food(kind_in),
            expected_input_name=self.vessel(vessel),
            expected_output_name=self.food(kind_out),
        )
        return [press] * presses

    def cook_watch(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> list[AgentMutation]:
        """Pour into the station, then wait there and take the food back."""
        ekind, plate, _food, kind_in, kind_out = args
        station = self.equipment(ekind)
        return [
            Combine(
                agent_id=chef,
                expected_held_equipment_name=self.vessel(plate),
                expected_held_equipment_contents_name=self.food(kind_in),
                expected_target_equipment_name=station,
                expected_output_name=self.food(kind_in),
            ),
            Combine(
                agent_id=chef,
                expected_held_equipment_name=self.vessel(plate),
                expected_target_equipment_name=station,
                expected_target_equipment_contents_name=self.food(kind_out),
                expected_output_name=self.food(kind_out),
            ),
        ]

    def cook_watch_vessel(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> list[AgentMutation]:
        """Pour into the pan or pot, then wait there and take the food back."""
        plate, vessel, _ekind, _base, _food, kind_in, kind_out = args
        return [
            Combine(
                agent_id=chef,
                expected_held_equipment_name=self.vessel(plate),
                expected_held_equipment_contents_name=self.food(kind_in),
                expected_target_equipment_name=self.vessel(vessel),
                expected_output_name=self.food(kind_in),
            ),
            Combine(
                agent_id=chef,
                expected_held_equipment_name=self.vessel(plate),
                expected_target_equipment_name=self.vessel(vessel),
                expected_target_equipment_contents_name=self.food(kind_out),
                expected_output_name=self.food(kind_out),
            ),
        ]

    def deliver(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        vessel, _food, kind, _order = args
        name = self.food(kind)
        return Deliver(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_food=name,
            expected_held_equipment_contents_name=name,
            expected_output_name=name,
        )

    def discard(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _food, kind = args
        return Discard(agent_id=chef, expected_held_food=self.food(kind))

    def discard_from(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> AgentMutation:
        vessel, _food, kind = args
        return Discard(
            agent_id=chef,
            expected_held_equipment_name=self.vessel(vessel),
            expected_held_equipment_contents_name=self.food(kind),
        )

    def wash(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        (vessel,) = args
        return Wash(agent_id=chef, expected_held_equipment_name=self.vessel(vessel))

    def wait(self, chef: str, slot: Slot, args: tuple[str, ...]) -> list[AgentMutation]:
        agent = self._environment.get_entity_as(chef, Agent)
        if agent is None:
            raise ValueError(f"Unknown chef {chef}")
        return [
            TurnAgent(agent_id=chef, orientation=agent.orientation)
            for _ in range(WAIT_STEPS)
        ]

    def stop(self, chef: str, slot: Slot, args: tuple[str, ...]) -> list[AgentMutation]:
        return []


def _target_food(slot: Slot, food_name: str) -> dict[str, str]:
    """Return the expectations for food the chef faces on ``slot``.

    Food in a station is checked as the station's contents; food on a bare
    counter as the target food.
    """
    if slot.equipment is not None:
        return {
            "expected_target_equipment_name": slot.equipment,
            "expected_target_equipment_contents_name": food_name,
        }
    return {"expected_target_food_name": food_name}


Handler = Callable[
    [DefaultFromPDDL, str, Slot, tuple[str, ...]],
    AgentMutation | list[AgentMutation],
]

_HANDLERS: dict[str, Handler] = {
    name: getattr(DefaultFromPDDL, name)
    for name in (
        "take",
        "take_into",
        "take_mix",
        "take_plate",
        "pick_up",
        "pick_up_vessel",
        "pick_up_full",
        "put_food",
        "put_vessel",
        "mount",
        "load",
        "load_mix",
        "grab_mix",
        "fill",
        "fill_mix",
        "grab_from_vessel",
        "merge",
        "collect",
        "collect_mix",
        "collect_mix_station",
        "pour",
        "pour_mix",
        "pour_station",
        "scoop",
        "scoop_mix",
        "cook",
        "cook_vessel",
        "cook_watch",
        "cook_watch_vessel",
        "deliver",
        "discard",
        "discard_from",
        "wash",
        "wait",
        "stop",
    )
}
