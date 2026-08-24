"""integration_test.py — Validation de bout en bout de la strategie
"Visite recente + WAIT modere" (anti-balancier WAIT <-> oscillation).

Verifie sur une carte de test miniature :
  - l'env reset/step sans erreur
  - has_productive_action retourne True quand le hero peut bouger
  - un aller-retour declenche la penalite revisited_position (-2.0)
  - un aller-retour coute PLUS cher que 2 WAIT consecutifs
    (c'est le coeur de la strategie anti-see-saw)
"""
import sys, os, tempfile, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from AlgoEnv import AlgoEnv
from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

# Carte 6x5 : 1 F, 1 M, 1 coffre *, 1 pierre +, 1 buisson @
rows = [
    "F...M.",
    "......",
    "...*..",
    "...+..",
    "...@..",
]
H, W = 5, 6
elev = np.array([[1]*W for _ in range(H)], dtype=np.int64)

tmp = tempfile.mkdtemp()
map_path = os.path.join(tmp, "map.txt")
elev_path = os.path.join(tmp, "elevation.txt")
with open(map_path, "w") as f:
    f.write(f"{H} {W} 50\n")
    for r in rows:
        f.write(r + "\n")
with open(elev_path, "w") as f:
    for row in elev:
        f.write(" ".join(map(str, row)) + "\n")


def _make_env():
    """Env avec ressources non pleines pour declencher wait_productive_action_available
    (-0.75) au lieu de wait_full_resources (-1.00)."""
    e = AlgoEnv(map_path, elev_path, hero="F",
                engine_config=DEFAULT_ENGINE_CONFIG,
                reward_config=DEFAULT_REWARD_CONFIG, seed=42)
    e.reset()
    e.engine.stamina["F"] = 50.0
    e.engine.battery["F"] = 50.0
    e._prev_potential = e._potential()
    return e


# Test 1: reset/step + has_productive_action
env = _make_env()
print("[1] reset OK, hero pos=", env.engine.pos["F"])
print("[1] has_productive_action au reset =", env._has_productive_action())
assert env._has_productive_action() is True, "F doit avoir une action productive au reset"

# Test 2: 10 steps aleatoires legaux — verifier qu'on ne reste pas coince
random.seed(0)
total_reward = 0.0
term = trunc = False
for i in range(10):
    mask = env.engine.legal_action_mask("F", env.action_names)
    legal_indices = [i for i, v in enumerate(mask) if v]
    a = random.choice(legal_indices)
    obs, reward, term, trunc, info = env.step(a)
    total_reward += reward
    print(f"[2] step {i}: action={env.action_names[a]:<8} reward={reward:+.3f} kind={info['action_result']:<18} pos={env.engine.pos['F']}")
    if term or trunc:
        print(f"[2] term={term} trunc={trunc} reason={info.get('termination_reason')}")
        break
print(f"[2] Total reward sur 10 steps: {total_reward:+.3f}")

# Test 3: aller-retour coute PLUS cher que 2 WAIT (coeur de la strategie)
env_a = _make_env()
right_idx = env_a.action_names.index("RIGHT")
left_idx = env_a.action_names.index("LEFT")
_, r_right, _, _, _ = env_a.step(right_idx)
_, r_left, _, _, _ = env_a.step(left_idx)
osc_cost = r_right + r_left
print(f"[3] Aller-retour RIGHT->LEFT: r_right={r_right:+.3f} r_left={r_left:+.3f} total={osc_cost:+.3f}")
# Le retour doit contenir la penalite de revisite (-2.0).
assert r_left < -2.0, f"Le retour doit declencher la penalite de revisite, r_left={r_left:+.3f}"

env_b = _make_env()
wait_idx = env_b.action_names.index("WAIT")
_, w1, _, _, _ = env_b.step(wait_idx)
_, w2, _, _, _ = env_b.step(wait_idx)
wait_cost = w1 + w2
print(f"[3] 2x WAIT (ressources non pleines): w1={w1:+.3f} w2={w2:+.3f} total={wait_cost:+.3f}")

# L'oscillation doit etre PLUS CHERE que 2 WAIT.
print(f"[3] Verdict: osc_cost={osc_cost:+.3f} vs wait_cost={wait_cost:+.3f}")
assert osc_cost < wait_cost, (
    f"L'oscillation ({osc_cost:+.3f}) doit etre plus chere que 2 WAIT "
    f"({wait_cost:+.3f}) — c'est le coeur de la strategie anti-see-saw."
)

# Test 4: un pas vers un objectif est recompense positivement
env_c = _make_env()
right_idx = env_c.action_names.index("RIGHT")
# F en (0,0), pierre en (3,4). RIGHT rapproche (Manhattan 7 -> 6).
_, r, _, _, info = env_c.step(right_idx)
print(f"[4] RIGHT vers objectif: r={r:+.3f}")
assert r > 0.3, f"Avancer vers un objectif doit etre recompense, r={r:+.3f}"

print("\nTous les tests d'integration OK — la strategie anti-see-saw fonctionne.")
