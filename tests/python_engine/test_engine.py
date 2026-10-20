import json
import hashlib
from pathlib import Path

from training.arena.engine import Agent, State, choose_action, observe, step
from training.arena.seeds import PCG32, derive_seed

ROOT = Path(__file__).resolve().parents[2]


def read_fixture(name):
    return json.loads((ROOT / "shared" / "fixtures" / name).read_text())


def fixture_state(case):
    positions = [tuple(p) for p in case["positions"]]
    trails = [[p] for p in positions]
    for x, y in case.get("occupied_cells", []):
        trails[0].insert(0, (x, y))
    agents = tuple(Agent(positions[i], ("UP", "RIGHT", "DOWN", "LEFT").index(case["headings"][i]), tuple(trails[i])) for i in range(2))
    return State(case["width"], case["height"], agents)


def test_collision_golden_vectors():
    vectors = read_fixture("collision-transitions-1.json")
    assert vectors["action_order"] == ["UP", "RIGHT", "DOWN", "LEFT"]
    for case in vectors["cases"]:
        transition = step(fixture_state(case), case["actions"])
        assert [list(causes) for causes in transition.causes] == case["expected_causes"], case["id"]
        assert transition.state.result == case["expected_result"], case["id"]


def test_seed_derivation_and_pcg_golden_vectors():
    for vector in read_fixture("seed-vectors-1.json")["vectors"]:
        seed = derive_seed(int(vector["master_seed"]), vector["label"], *vector["indices"])
        assert seed.preimage_hex == vector["preimage_hex"]
        assert seed.digest_hex == vector["sha256"]
        assert str(seed.initstate) == vector["initstate"]
        assert str(seed.initseq) == vector["initseq"]
        rng = PCG32.from_seed(seed)
        assert [rng.next_u32() for _ in vector["pcg32"]] == vector["pcg32"]


def test_observation_shape_orientation_and_action_tie_break():
    state = State(4, 3, (Agent((1, 1), 1, ((1, 1),)), Agent((2, 1), 3, ((2, 1),))))
    obs = observe(state, 0)
    assert (len(obs), len(obs[0]), len(obs[0][0])) == (3, 4, 12)
    assert obs[1][1][0] == obs[1][1][2] == obs[1][1][5] == 1
    assert obs[1][2][1] == obs[1][2][3] == obs[1][2][11] == 1
    assert choose_action([0.25, 0.25, -2, 0.25]) == 0


def test_head_swap_collects_both_causes_and_commits_simultaneously():
    initial = State(5, 5, (Agent((1, 2), 1, ((1, 2),)), Agent((2, 2), 3, ((2, 2),))))
    outcome = step(initial, (1, 3))
    assert outcome.causes == (("occupied_head", "head_swap"), ("occupied_head", "head_swap"))
    assert outcome.state.result == "draw"
    assert outcome.state.agents == tuple(Agent(a.position, a.heading, a.trail, False) for a in initial.agents)
    assert step(outcome.state, (0, 0)).state == outcome.state


def test_swapping_agent_indices_cannot_change_collision_priority():
    initial = State(5, 5, (Agent((0, 0), 0, ((0, 0),)), Agent((4, 4), 3, ((4, 4),))))
    original = step(initial, (0, 0))
    swapped = State(5, 5, tuple(reversed(initial.agents)))
    reversed_outcome = step(swapped, (0, 0))
    assert reversed_outcome.causes == tuple(reversed(original.causes))
    assert reversed_outcome.state.result == "agent_0_win"  # swapped agent 1 now survives
    assert original.state.result == "agent_1_win"


def test_replay_fixture_framing_and_collision_contract():
    replay = read_fixture("lc-replay-1.valid.json")
    assert replay["format"] == "lc-replay/1"
    assert replay["initial_state"]["width"] > 0 and replay["initial_state"]["height"] > 0
    assert replay["tick_count"] == len(replay["ticks"])
    for index, frame in enumerate(replay["ticks"]):
        assert frame["tick"] == index
        assert len(frame["joint_actions"]) == len(frame["collision_causes"]) == 2
        assert all(isinstance(causes, list) for causes in frame["collision_causes"])
    assert replay["result"] in {"agent_0_win", "agent_1_win", "draw"}
    initial = replay["initial_state"]
    agents = tuple(
        Agent((position["x"], position["y"]), ("UP", "RIGHT", "DOWN", "LEFT").index(initial["headings"][i]),
              tuple((cell["x"], cell["y"]) for cell in initial["trails"][i]))
        for i, position in enumerate(initial["positions"])
    )
    state = State(initial["width"], initial["height"], agents)
    for frame in replay["ticks"]:
        transition = step(state, frame["joint_actions"])
        assert [list(causes) for causes in transition.causes] == frame["collision_causes"]
        expected_terminal = [f"agent_{i}_dead" for i, agent in enumerate(transition.state.agents) if not agent.alive]
        assert expected_terminal == frame["terminal_causes"]
        state = transition.state
    assert state.result == replay["result"]
    assert state.tick == replay["tick_count"]
    digest_doc = dict(replay)
    digest = digest_doc.pop("replay_sha256")
    canonical = json.dumps(digest_doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == digest
