"""Test throwing food away at a bin."""

from __future__ import annotations

import pytest

from simulator.configuration.configuration import Configuration
from simulator.configuration.load import load
from simulator.entities import Agent, Equipment, Food
from simulator.mutations import Discard, Interact
from simulator.mutations.mutation import IllegalMutationError

TOMATO = "catalog/food/tomato"
PAN = "catalog/equipment/pan"


def _kitchen(layout: str):
    """Return a kitchen whose chef faces east."""
    return load(
        Configuration.from_dict(
            {
                "layout": layout,
                "state": {"orders": [TOMATO]},
                "legend": {
                    "agents": [{"kind": "agent", "symbol": "A", "orientation": "e"}],
                    "counters": [{"kind": "counter", "symbol": "-"}],
                    "bins": [{"kind": "bin", "symbol": "X"}],
                    "foods": [{"kind": "food", "name": TOMATO}],
                },
            }
        )
    )


def _agent(environment) -> Agent:
    agent = environment.get_first_entity_of_type(Agent)
    assert agent is not None
    return agent


def _holding_tomato(layout: str):
    """Return a kitchen whose chef holds a tomato, and the tomato."""
    environment = _kitchen(layout)
    tomato = Food(name=TOMATO, x=None, y=None)
    environment = environment.with_entity(tomato).replace_entity(
        _agent(environment).copy_with(held_item_id=tomato.id)
    )
    return environment, tomato


def _holding_pan_of_tomato(layout: str):
    """Return a kitchen whose chef holds a pan with a tomato in it."""
    environment = _kitchen(layout)
    tomato = Food(name=TOMATO, x=None, y=None)
    pan = Equipment(name=PAN, can_pick_up=True, held_item_id=tomato.id, x=None, y=None)
    environment = (
        environment.with_entity(tomato)
        .with_entity(pan)
        .replace_entity(_agent(environment).copy_with(held_item_id=pan.id))
    )
    return environment, pan, tomato


def test_discarding_at_a_bin_throws_the_held_food_away() -> None:
    environment, tomato = _holding_tomato("| A | X |")

    result = Discard(agent_id=_agent(environment).id).run(environment)

    assert _agent(result).held_item_id is None
    assert result.get_entity(tomato.id) is None


def test_discarding_at_a_bin_empties_the_held_cookware() -> None:
    environment, pan, tomato = _holding_pan_of_tomato("| A | X |")

    result = Discard(agent_id=_agent(environment).id).run(environment)

    assert _agent(result).held_item_id == pan.id
    emptied = result.get_entity_as(pan.id, Equipment)
    assert emptied is not None
    assert emptied.held_item_id is None
    assert result.get_entity(tomato.id) is None


def test_interact_at_a_bin_discards() -> None:
    environment, tomato = _holding_tomato("| A | X |")

    result = Interact(agent_id=_agent(environment).id).run(environment)

    assert _agent(result).held_item_id is None
    assert result.get_entity(tomato.id) is None


def test_discarding_away_from_a_bin_is_illegal() -> None:
    environment, _ = _holding_tomato("| A | - |")

    with pytest.raises(IllegalMutationError):
        Discard(agent_id=_agent(environment).id).run(environment)


def test_discarding_with_a_bin_behind_the_chef_is_illegal() -> None:
    environment, _ = _holding_tomato("| X | A | - |")

    with pytest.raises(IllegalMutationError):
        Discard(agent_id=_agent(environment).id).run(environment)


def test_discarding_cookware_contents_away_from_a_bin_is_illegal() -> None:
    environment, _, _ = _holding_pan_of_tomato("| A | - |")

    with pytest.raises(IllegalMutationError):
        Discard(agent_id=_agent(environment).id).run(environment)


def test_discarding_empty_handed_is_illegal() -> None:
    environment = _kitchen("| A | X |")

    with pytest.raises(IllegalMutationError):
        Discard(agent_id=_agent(environment).id).run(environment)
