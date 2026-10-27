"""Translate plans for the instance-level domain into chef actions.

``DefaultFromPDDL`` is built with the environment the plan was made for. It
reads the same ``Kitchen`` as the problem generator, so every symbol in the
plan maps back to a chef id, a tile, or a catalog name.

Every action in overcooked.pddl names the chef first and the slot it acts on
second; the slot's tile is where the chef acts. Each step becomes one
mutation with its ``expected_*`` fields filled in from the step's own
arguments, so a chef waits rather than acting on the wrong thing.
"""

from __future__ import annotations

from collections.abc import Callable

from controllers.pddl.interfaces import FromPDDL, PDDLPlanStep
from controllers.pddl.submission.kitchen import Kitchen, Slot, read_kitchen
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
)
from simulator.mutations.agent_mutation import AgentMutation


class DefaultFromPDDL(FromPDDL):
    """Translate a symbolic plan into simulator mutations."""

    def __init__(self, environment: Environment) -> None:
        self._environment = environment
        self._kitchen: Kitchen | None = None

    @property
    def kitchen(self) -> Kitchen:
        if self._kitchen is None:
            self._kitchen = read_kitchen(self._environment)
        return self._kitchen

    def from_pddl(self, steps: list[PDDLPlanStep]) -> list[MutationWithLocation]:
        """Translate all steps in plan order."""
        return [self.translate(step) for step in steps]

    def translate(self, step: PDDLPlanStep) -> MutationWithLocation:
        handler = _HANDLERS.get(step.name)
        if handler is None:
            raise ValueError(f"No translation for this action yet: {step}")
        if len(step.parameters) < 2:
            raise ValueError(f"Step names no chef and slot: {step}")
        chef_id = self.chef_id(step.parameters[0])
        slot = self.slot(step.parameters[1])
        try:
            mutation = handler(self, chef_id, slot, step.parameters[2:])
        except (ValueError, KeyError) as error:
            raise ValueError(f"Cannot translate {step}: {error}") from error
        return MutationWithLocation(mutation=mutation, location=slot.location)

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

    # --- One method per action, given the arguments after chef and slot ----------

    def take(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _food, kind = args
        name = self.food(kind)
        return TakeFromStorage(
            agent_id=chef, expected_input_name=name, expected_output_name=name
        )

    def take_into(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _vessel, _ekind, _food, kind = args
        name = self.food(kind)
        return TakeFromStorage(
            agent_id=chef, expected_input_name=name, expected_output_name=name
        )

    def take_mix(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _vessel, _ekind, _food, _kg, kind, out = args
        return TakeFromStorage(
            agent_id=chef,
            expected_input_name=self.food(kind),
            expected_output_name=self.food(out),
        )

    def pick_up(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        _food, kind = args
        name = self.food(kind)
        return PickUp(agent_id=chef, expected_input_name=name, expected_output_name=name)

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
        vessel, _ekind, _base = args
        return Place(agent_id=chef, expected_held_equipment_name=self.vessel(vessel))

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

    def cook(self, chef: str, slot: Slot, args: tuple[str, ...]) -> AgentMutation:
        ekind, _food, kind_in, kind_out = args
        return Cook(
            agent_id=chef,
            expected_held_food=self.food(kind_in),
            expected_input_name=self.equipment(ekind),
            expected_output_name=self.food(kind_out),
        )

    def cook_vessel(
        self, chef: str, slot: Slot, args: tuple[str, ...]
    ) -> AgentMutation:
        vessel, _ekind, _base, _food, kind_in, kind_out = args
        return Cook(
            agent_id=chef,
            expected_held_food=self.food(kind_in),
            expected_input_name=self.vessel(vessel),
            expected_output_name=self.food(kind_out),
        )

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


Handler = Callable[[DefaultFromPDDL, str, Slot, tuple[str, ...]], AgentMutation]

_HANDLERS: dict[str, Handler] = {
    name: getattr(DefaultFromPDDL, name)
    for name in (
        "take",
        "take_into",
        "take_mix",
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
        "deliver",
        "discard",
        "discard_from",
    )
}
