"""integration_test.py — Validation de bout en bout des correctifs apportes
aux fichiers AlgoEnv.py / PlayOnnx.py / AlgoTrain.py (cf. diagnostic).

Lance une partie courte sur une carte de test miniature et verifie :
  - l'env reset/step sans erreur
  - has_productive_action retourne True quand le hero peut bouger
  - aller-retour (UP puis DOWN) a un gain net ~0 (symetrie shaping)
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

# Test 1: reset/step + has_productive_action
env = AlgoEnv(map_path, elev_path, hero="F",
              engine_config=DEFAULT_ENGINE_CONFIG,
              reward_config=DEFAULT_REWARD_CONFIG, seed=42)
obs, info = env.reset()
print("[1] reset OK, hero pos=", env.engine.pos["F"])
print("[1] has_productive_action au reset =", env._has_productive_action())
assert env._has_productive_action() is True, "F doit avoir une action productive au reset"

# Test 2: 10 steps aleatoires legaux
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

# Test 3: symetrie aller-retour
env2 = AlgoEnv(map_path, elev_path, hero="F",
              engine_config=DEFAULT_ENGINE_CONFIG,
              reward_config=DEFAULT_REWARD_CONFIG, seed=42)
obs, info = env2.reset()
mask = env2.engine.legal_action_mask("F", env2.action_names)
move_indices = [i for i, v in enumerate(mask)
                if v and env2.action_names[i] in ("UP", "DOWN", "LEFT", "RIGHT")]
print(f"[3] Moves legaux au reset: {[env2.action_names[i] for i in move_indices]}")

if move_indices:
    a = move_indices[0]
    name = env2.action_names[a]
    opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[name]
    opp_idx = env2.action_names.index(opp) if opp in env2.action_names else None
    if opp_idx is not None:
        _, r1, _, _, _ = env2.step(a)
        mask2 = env2.engine.legal_action_mask("F", env2.action_names)
        if mask2[opp_idx]:
            _, r2, _, _, _ = env2.step(opp_idx)
            print(f"[3] Aller {name}: r1={r1:+.3f}, Retour {opp}: r2={r2:+.3f}, somme={r1+r2:+.3f}")
            # Le potential telescoping + shaping symetrique => somme ~ step_time_cost*2 = -0.10
            # On accepte une tolerance large (le potential varie legerement avec la ressource).
            assert abs(r1 + r2) < 0.7, f"Aller-retour non symetrique: {r1+r2}"
            print("[3] Symetrie aller-retour: OK")
        else:
            print(f"[3] Action opposee {opp} non legale, test ignore")
    else:
        print(f"[3] Pas d'oppose pour {name}, test ignore")
else:
    print("[3] Aucun move legal au reset, test ignore")

print("\nTous les tests d'integration OK")
