import numpy as np
from GameEngine import GameEngine


def test_machine_stores_stone_then_hero_steals_it():
    rows = [
        "X+.",
        "F.M",
    ]
    elevation = np.ones((2, 3), dtype=np.int8)
    e = GameEngine(rows, elevation, 20)
    machine = e.machines[0]
    machine["facing"] = (1, 0)
    machine["move_pending"] = True
    e.step({"F": "WAIT", "M": "WAIT"})
    assert machine["stones"] == 1
    assert (1, 0) not in e.stones
    e.pos["F"] = (machine["x"], machine["y"])
    e._transfer_machine_stones(0)
    assert machine["stones"] == 0
    assert e.stones_collected == 1


def test_excavator_does_not_dig_bush():
    rows = [
        "F.M",
        ".X@",
    ]
    elevation = np.ones((2, 3), dtype=np.int8)
    e = GameEngine(rows, elevation, 20)
    m = e.machines[0]
    m["x"], m["y"] = 2, 1
    m["steps"] = 5
    e.step({"F": "WAIT", "M": "WAIT"})
    assert e.terrain[1][2] == "@"


def test_invalid_move_executes_implicit_wait_and_recovers():
    rows = [
        "F#M",
    ]
    elevation = np.array([[1, 0, 1]], dtype=np.int8)
    e = GameEngine(rows, elevation, 20)
    e.stamina["F"] = 99.0
    e.regen["F"] = 1.0
    ev = e.step({"F": "RIGHT", "M": "WAIT"})
    assert ev["F"]["kind"] == "implicit_wait"
    assert e.stamina["F"] == 100.0


def test_hacked_hero_can_move_out_and_unhack():
    rows = [
        "FXM",
        "...",
    ]
    elevation = np.ones((2, 3), dtype=np.int8)
    e = GameEngine(rows, elevation, 20)
    e.pos["F"] = (1, 0)
    e.machines[0]["x"], e.machines[0]["y"] = 1, 0
    e.machines[0]["hacked_by"] = "F"
    e.on_engine["F"] = 0
    ev = e.step({"F": "DOWN", "M": "WAIT"})
    assert ev["F"]["kind"] == "unhack_move"
    assert e.on_engine["F"] is None
    assert e.machines[0]["hacked_by"] is None
    assert e.pos["F"] == (1, 1)
