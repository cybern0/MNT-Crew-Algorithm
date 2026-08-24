"""GameEngine.py — simulateur deterministe GDD + Twist. 1 step() = 1 tick.

Autorite unique des regles : mouvement F/M (hop trou / climb arbre), push de
coffres (blocage montee, destruction falaise >= 5 ou trou, dissimulation dans
un buisson), hacking, autopilote X/G, regeneration de stamina, resolution
look-ahead des collisions simultanees.

Deux changements par rapport a la version precedente :
  1. Attribution par hero (stones_by / chests_hidden_by / chests_destroyed_by).
     Sans elle, la recompense d'un hero bougeait a cause de l'autre et
     l'invariant "aucun cycle positif" tombait.
  2. Un seul masque : legal_action_mask(). Le masque "structurel" laissait
     passer des actions converties en implicit_wait par le moteur, c'est-a-dire
     des WAIT gratuits que rien n'incitait a eviter.
"""
from __future__ import annotations

import random

import numpy as np

DIR = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
PUSH_DIR = {"PUSH_UP": (0, -1), "PUSH_DOWN": (0, 1),
            "PUSH_LEFT": (-1, 0), "PUSH_RIGHT": (1, 0)}
CW = {(0, -1): (1, 0), (1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1)}
CCW = {v: k for k, v in CW.items()}
MOVE_STAMINA = 1.0

_DEFAULTS = dict(
    hero_uphill_block=3, hero_downhill_block=5,
    machine_height_block=5, machine_chest_uphill_block=3,
    uphill_stamina_cost=3.0, downhill_stamina_cost=0.0,
    forbid_chest_uphill=True, destroy_chest_drop=5,
    chest_push_stamina_cost=5.0, hole_hop_stamina_cost=10.0,
    tree_climb_stamina_cost=10.0,
    excavator_initial_facing="LEFT", grappler_initial_facing="RIGHT",
    hack_battery_cost=1.0, stamina_recovery_rate=0.5,
    stamina_recovery_interval=2.0, machine_dig_every=5,
    avoid_competing_pushers=True,
)


class GameEngine:
    """hero in {'F', 'M'} partout ci-dessous."""

    def __init__(self, ascii_rows, elevation, max_time, engine_config=None, seed=0):
        self.cfg = {**_DEFAULTS, **(engine_config or {})}
        self.H, self.W = len(ascii_rows), len(ascii_rows[0])
        self.elevation = elevation
        self.max_time = max_time
        self.rng = random.Random(seed)
        self._load(ascii_rows)

    def _load(self, ascii_rows):
        self.terrain = [list(row) for row in ascii_rows]
        self.stones, self.chests, self.machines = set(), [], []
        self.pos = {}
        for y in range(self.H):
            for x in range(self.W):
                c = self.terrain[y][x]
                if c == "+":
                    self.stones.add((x, y)); self.terrain[y][x] = "."
                elif c == "*":
                    self.chests.append([x, y]); self.terrain[y][x] = "."
                elif c in ("F", "M"):
                    self.pos[c] = (x, y); self.terrain[y][x] = "."
                elif c in ("X", "G"):
                    key = "excavator_initial_facing" if c == "X" else "grappler_initial_facing"
                    self.machines.append({
                        "type": c, "x": x, "y": y, "facing": DIR[self.cfg[key]],
                        "hacked_by": None, "move_pending": False,
                        "steps": 0, "stones": 0,
                    })
                    self.terrain[y][x] = "."
        self.stamina = {"F": 100.0, "M": 100.0}
        self.battery = {"F": 100.0, "M": 100.0}
        self.on_engine = {"F": None, "M": None}
        self.regen = {"F": 0.0, "M": 0.0}
        self.facing = {"F": (1, 0), "M": (1, 0)}
        self.hero_move_pending = {"F": False, "M": False}
        # Compteurs globaux (score officiel, parite GameRunner).
        self.stones_collected = 0
        self.chests_hidden = 0
        self.chests_destroyed = 0
        # Attribution par hero (credit assignment de la recompense locale).
        self.stones_by = {"F": 0, "M": 0}
        self.chests_hidden_by = {"F": 0, "M": 0}
        self.chests_destroyed_by = {"F": 0, "M": 0}
        self.tick = 0
        self.resource_low_ticks = {"F": 0, "M": 0}

    # ---- lecture d'etat ---------------------------------------------------
    def in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def elev(self, x, y):
        return int(self.elevation[y, x])

    def chest_at(self, x, y):
        for i, (cx, cy) in enumerate(self.chests):
            if cx == x and cy == y:
                return i
        return None

    def machine_at(self, x, y):
        for i, m in enumerate(self.machines):
            if m["x"] == x and m["y"] == y:
                return i
        return None

    def hero_at(self, x, y, exclude=None):
        for h, (hx, hy) in self.pos.items():
            if h != exclude and hx == x and hy == y:
                return h
        return None

    def can_stand(self, hero, x, y):
        """Case ou l'on peut poser un hero (utilise par le reverse curriculum)."""
        if not self.in_bounds(x, y):
            return False
        if self.terrain[y][x] in ("#", "o", "t"):
            return False
        if self.chest_at(x, y) is not None:
            return False
        return self.hero_at(x, y, exclude=hero) is None

    def official_score(self):
        """Formule exacte du GDD / GameRunner.calcTotalScore()."""
        s = int(self.stamina["F"] + self.stamina["M"])
        b = int(self.battery["F"] + self.battery["M"])
        t = max(0, int(self.max_time - self.tick))
        return (self.chests_hidden * 150 + self.stones_collected * 25
                + (s + b) // 4 + t // 2)

    def _implicit_wait_event(self, reason):
        return {"valid": True, "requested_valid": False,
                "kind": "implicit_wait", "reason": reason}

    # ---- masque d'actions legales ----------------------------------------
    # Chaque condition est un miroir exact de celle utilisee par step().
    # C'est un contrat structurel : il ne dit pas ou aller, seulement ce qui
    # existe. Aucune penalite d'invalidite n'est donc necessaire.
    def legal_action_mask(self, hero, action_names):
        mask = np.zeros(len(action_names), dtype=bool)
        riding = self.on_engine[hero] is not None

        if riding and self.hero_move_pending[hero]:
            mask[action_names.index("WAIT")] = True
            return mask

        m = self.machines[self.on_engine[hero]] if riding else None
        for i, name in enumerate(action_names):
            if riding:
                if name == "WAIT":
                    mask[i] = True
                elif name in ("HACK_MOVE", "HACK_CW", "HACK_CCW"):
                    mask[i] = self.battery[hero] >= self.cfg["hack_battery_cost"]
                elif name == "HACK_FILL" and hero == "F":
                    dx, dy = m["facing"]
                    fx, fy = m["x"] + dx, m["y"] + dy
                    mask[i] = (self.battery[hero] >= self.cfg["hack_battery_cost"]
                               and self.in_bounds(fx, fy)
                               and self.terrain[fy][fx] == "o")
                continue
            if name == "WAIT":
                mask[i] = True
            elif name in DIR:
                tgt = self._hero_target(hero, *DIR[name])
                if tgt is not None:
                    lx, ly, hop = tgt
                    mask[i] = self.stamina[hero] >= self._move_cost(
                        hero, *self.pos[hero], lx, ly, hop)
            elif name in PUSH_DIR:
                mask[i] = self._try_push(hero, *PUSH_DIR[name]) is not None
            elif name == "HACK":
                x, y = self.pos[hero]
                idx = self.machine_at(x, y)
                mask[i] = (idx is not None
                           and self.machines[idx]["type"] == ("X" if hero == "F" else "G")
                           and self.machines[idx]["hacked_by"] is None
                           and self.battery[hero] >= self.cfg["hack_battery_cost"])

        if not mask.any():
            mask[action_names.index("WAIT")] = True
        return mask

    # ---- geometrie / couts ------------------------------------------------
    def _elev_blocked(self, kind, x0, y0, x1, y1):
        d = self.elev(x1, y1) - self.elev(x0, y0)
        if kind in ("F", "M"):
            if d > 0:
                return d >= self.cfg["hero_uphill_block"]
            return (-d) >= self.cfg["hero_downhill_block"]
        return abs(d) >= self.cfg["machine_height_block"]

    def _hero_target(self, hero, dx, dy):
        """(x, y, hop) si le deplacement existe, sinon None."""
        x, y = self.pos[hero]
        tx, ty = x + dx, y + dy
        if not self.in_bounds(tx, ty):
            return None
        t = self.terrain[ty][tx]
        if t == "#" or (hero == "F" and t == "t") or (hero == "M" and t == "o"):
            return None
        special = t == "o" if hero == "F" else t == "t"
        if special:
            lx, ly = tx + dx, ty + dy
            if not self.in_bounds(lx, ly):
                return None
            lt = self.terrain[ly][lx]
            if lt == "#" or lt == t:      # chaine de 2 trous/arbres interdite
                return None
            if self._elev_blocked(hero, x, y, lx, ly):
                return None
            return (lx, ly, True)
        if self._elev_blocked(hero, x, y, tx, ty):
            return None
        return (tx, ty, False)

    def _move_cost(self, hero, x0, y0, x1, y1, hop):
        if hop:
            return (self.cfg["hole_hop_stamina_cost"] if hero == "F"
                    else self.cfg["tree_climb_stamina_cost"])
        d = self.elev(x1, y1) - self.elev(x0, y0)
        if d > 0:
            return self.cfg["uphill_stamina_cost"]
        if d < 0:
            return self.cfg["downhill_stamina_cost"]
        return MOVE_STAMINA

    def _try_push(self, hero, dx, dy):
        x, y = self.pos[hero]
        cx, cy = x + dx, y + dy
        idx = self.chest_at(cx, cy)
        if idx is None:
            return None
        # Le pousseur avance sur la case du coffre : meme contrainte
        # d'elevation que pour un deplacement (cf. GameRunner
        # isNextCellUnreachable, applique aussi aux PUSH_*).
        if self._elev_blocked(hero, x, y, cx, cy):
            return None
        bx, by = cx + dx, cy + dy
        if not self.in_bounds(bx, by):
            return None
        bt = self.terrain[by][bx]
        if bt in ("#", "t"):
            return None
        if self.cfg["forbid_chest_uphill"] and self.elev(bx, by) > self.elev(cx, cy):
            return None
        if self.chest_at(bx, by) is not None or self.machine_at(bx, by) is not None:
            return None
        if self.hero_at(bx, by, exclude=hero) is not None:
            return None
        if self.stamina[hero] < self.cfg["chest_push_stamina_cost"]:
            return None
        return (idx, cx, cy, bx, by, bt)

    # ---- machines ---------------------------------------------------------
    def _machine_push_plan(self, m, cx, cy):
        dx, dy = m["facing"]
        if self.chest_at(cx, cy) is None:
            return None
        bx, by = cx + dx, cy + dy
        if not self.in_bounds(bx, by):
            return None
        bt = self.terrain[by][bx]
        if bt in ("#", "t"):
            return None
        d = self.elev(bx, by) - self.elev(cx, cy)
        if d > 0 and d >= self.cfg["machine_chest_uphill_block"]:
            return None
        if self.chest_at(bx, by) is not None or self.machine_at(bx, by) is not None:
            return None
        if self.hero_at(bx, by) is not None:
            return None
        return (self.chest_at(cx, cy), cx, cy, bx, by, bt)

    def _lookahead_triangle(self, cx, cy, dx, dy):
        px, py = -dy, dx
        return (cx + px, cy + py), (cx + dx, cy + dy), (cx - px, cy - py)

    def _machine_lookahead_blocked(self, m, target):
        dx, dy = m["facing"]
        tx, ty = target
        scan = self._lookahead_triangle(tx, ty, dx, dy)

        def has_machine(p):
            return self.in_bounds(*p) and self.machine_at(*p) is not None

        def has_hero(p):
            return self.in_bounds(*p) and self.hero_at(*p) is not None

        if self.chest_at(tx, ty) is not None:
            lx, ly = scan[1]
            land = self._lookahead_triangle(lx, ly, dx, dy)
            if any(has_machine(p) for p in scan) or any(has_machine(p) for p in land):
                return True
            return any(has_hero(p) for p in scan)

        if any(has_machine(p) for p in scan):
            return True
        offsets = ((-dy, dx), (dx, dy), (dy, -dx))
        for p, off in zip(scan, offsets):
            if self.chest_at(*p) is not None and has_hero((p[0] + off[0], p[1] + off[1])):
                return True
        return False

    def _machine_target(self, m):
        dx, dy = m["facing"]
        tx, ty = m["x"] + dx, m["y"] + dy
        if not self.in_bounds(tx, ty):
            return None
        t = self.terrain[ty][tx]
        if t == "#" or (m["type"] == "X" and t == "t") or (m["type"] == "G" and t == "o"):
            return None
        other = self.machine_at(tx, ty)
        if other is not None and self.machines[other] is not m:
            return None
        if self._elev_blocked(m["type"], m["x"], m["y"], tx, ty):
            return None
        push = self._machine_push_plan(m, tx, ty)
        if self.chest_at(tx, ty) is not None and push is None:
            return None
        if self._machine_lookahead_blocked(m, (tx, ty)):
            return None
        return (tx, ty, push)

    def _rotate(self, m):
        m["facing"] = CCW[m["facing"]] if m["type"] == "X" else CW[m["facing"]]

    def _rider(self, machine_index):
        return next((h for h in ("F", "M") if self.on_engine[h] == machine_index), None)

    def _transfer_machine_stones(self, machine_index):
        machine = self.machines[machine_index]
        amount = int(machine.get("stones", 0))
        if amount <= 0:
            return
        mx, my = machine["x"], machine["y"]
        for hero in ("F", "M"):
            if self.pos[hero] == (mx, my):
                self.stones_collected += amount
                self.stones_by[hero] += amount
                machine["stones"] = 0
                return

    def preview_machine_positions(self, ticks):
        """Simule N ticks d'autopilote SANS muter l'etat reel (canaux 13/14)."""
        terrain = [row[:] for row in self.terrain]
        clones = [dict(m) for m in self.machines if m["hacked_by"] is None]
        for c in clones:
            c["_pending"] = False
        for _ in range(max(0, ticks)):
            for m in clones:
                dx, dy = m["facing"]
                tx, ty = m["x"] + dx, m["y"] + dy
                blocked = not (0 <= tx < self.W and 0 <= ty < self.H)
                if not blocked:
                    t = terrain[ty][tx]
                    blocked = (t == "#"
                               or (m["type"] == "X" and t == "t")
                               or (m["type"] == "G" and t == "o")
                               or any(o is not m and o["x"] == tx and o["y"] == ty for o in clones)
                               or self._elev_blocked(m["type"], m["x"], m["y"], tx, ty))
                if blocked:
                    m["facing"] = CCW[m["facing"]] if m["type"] == "X" else CW[m["facing"]]
                    m["_pending"] = False
                elif m["_pending"]:
                    if terrain[ty][tx] == "t":
                        terrain[ty][tx] = "."
                    m["x"], m["y"] = tx, ty
                    m["_pending"] = False
                else:
                    m["_pending"] = True
        out = {"X": [], "G": []}
        for m in clones:
            out[m["type"]].append((m["x"], m["y"]))
        return out

    # ---- heuristique du hero non controle --------------------------------
    def scripted_action(self, hero):
        if self.on_engine[hero] is not None:
            return "HACK_MOVE"
        x, y = self.pos[hero]
        targets = list(self.stones) + [(cx, cy) for cx, cy in self.chests]
        if targets:
            tx, ty = min(targets, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
            candidates = (
                ((1 if tx > x else -1, 0), "RIGHT" if tx > x else "LEFT"),
                ((0, 1 if ty > y else -1), "DOWN" if ty > y else "UP"),
            )
            for cand, name in candidates:
                if cand == (0, 0):
                    continue
                if self._hero_target(hero, *cand) is not None:
                    return name
        idx = self.machine_at(x, y)
        want = "X" if hero == "F" else "G"
        if idx is not None and self.machines[idx]["type"] == want \
                and self.machines[idx]["hacked_by"] is None:
            return "HACK"
        return "WAIT"

    # ---- un tick ----------------------------------------------------------
    def step(self, actions):
        """actions: {'F': nom, 'M': nom}. Renvoie {hero: evenement}."""
        ev = {h: {"valid": False, "kind": "invalid"} for h in ("F", "M")}
        intents, push_plan, machine_push_plan = {}, {}, {}

        for h in ("F", "M"):
            a = actions.get(h, "WAIT")
            riding = self.on_engine[h] is not None

            if riding and a in DIR:                       # descendre en bougeant
                dx, dy = DIR[a]
                self.facing[h] = (dx, dy)
                tgt = self._hero_target(h, dx, dy)
                if tgt is None:
                    ev[h] = self._implicit_wait_event("blocked_exit"); continue
                lx, ly, hop = tgt
                cost = self._move_cost(h, *self.pos[h], lx, ly, hop)
                if self.stamina[h] < cost:
                    ev[h] = self._implicit_wait_event("insufficient_stamina"); continue
                idx = self.on_engine[h]
                self.machines[idx]["hacked_by"] = None
                self.machines[idx]["move_pending"] = False
                self.on_engine[h] = None
                self.hero_move_pending[h] = False
                intents[("hero", h)] = (lx, ly)
                ev[h] = {"valid": True, "kind": "unhack_move", "cost": cost}
                continue

            if riding:
                m = self.machines[self.on_engine[h]]
                if self.hero_move_pending[h]:
                    # Tick d'execution du HACK_MOVE precedent : le hero attend,
                    # la machine termine son mouvement (look-ahead prioritaire).
                    ev[h] = {"valid": True, "kind": "ride_wait"}
                    idx = self.on_engine[h]
                    tgt = self._machine_target(m)
                    if tgt is None:
                        intents[("machine", idx)] = (m["x"], m["y"])
                    else:
                        tx, ty, push = tgt
                        intents[("machine", idx)] = (tx, ty)
                        if push is not None:
                            machine_push_plan[idx] = push
                    continue
                if a == "WAIT":
                    ev[h] = {"valid": True, "kind": "unhack"}
                elif a in ("HACK_MOVE", "HACK_FILL", "HACK_CW", "HACK_CCW"):
                    if (a == "HACK_FILL" and h != "F") or \
                            self.battery[h] < self.cfg["hack_battery_cost"]:
                        ev[h] = self._implicit_wait_event("hack_not_allowed"); continue
                    if a == "HACK_MOVE":
                        if self._machine_target(m) is None:
                            ev[h] = {"valid": True, "kind": "hack_blocked_rotate"}
                        else:
                            self.hero_move_pending[h] = True
                            ev[h] = {"valid": True, "kind": "hack_move_queued"}
                    elif a == "HACK_CW":
                        m["facing"] = CW[m["facing"]]
                        ev[h] = {"valid": True, "kind": "hack_cw"}
                    elif a == "HACK_CCW":
                        m["facing"] = CCW[m["facing"]]
                        ev[h] = {"valid": True, "kind": "hack_ccw"}
                    else:                                  # HACK_FILL
                        dx, dy = m["facing"]
                        fx, fy = m["x"] + dx, m["y"] + dy
                        if self.in_bounds(fx, fy) and self.terrain[fy][fx] == "o":
                            self.terrain[fy][fx] = "."
                            ev[h] = {"valid": True, "kind": "fill"}
                        else:
                            ev[h] = self._implicit_wait_event("no_hole_to_fill"); continue
                    self.battery[h] = max(0.0, self.battery[h] - self.cfg["hack_battery_cost"])
                else:
                    ev[h] = self._implicit_wait_event("invalid_action")
                continue

            if a == "WAIT":
                ev[h] = {"valid": True, "kind": "wait"}
            elif a in DIR:
                dx, dy = DIR[a]
                self.facing[h] = (dx, dy)
                tgt = self._hero_target(h, dx, dy)
                if tgt is None:
                    ev[h] = self._implicit_wait_event("blocked_move"); continue
                lx, ly, hop = tgt
                cost = self._move_cost(h, *self.pos[h], lx, ly, hop)
                if self.stamina[h] < cost:
                    ev[h] = self._implicit_wait_event("insufficient_stamina"); continue
                intents[("hero", h)] = (lx, ly)
                ev[h] = {"valid": True, "kind": "move", "cost": cost}
            elif a in PUSH_DIR:
                dx, dy = PUSH_DIR[a]
                self.facing[h] = (dx, dy)
                plan = self._try_push(h, dx, dy)
                if plan is None:
                    ev[h] = self._implicit_wait_event("push_blocked"); continue
                push_plan[h] = plan
                intents[("hero", h)] = (plan[1], plan[2])
                ev[h] = {"valid": True, "kind": "push"}
            elif a == "HACK":
                x, y = self.pos[h]
                idx = self.machine_at(x, y)
                want = "X" if h == "F" else "G"
                if (idx is None or self.machines[idx]["type"] != want
                        or self.machines[idx]["hacked_by"] is not None
                        or self.battery[h] < self.cfg["hack_battery_cost"]):
                    ev[h] = self._implicit_wait_event("hack_unavailable"); continue
                self.machines[idx]["hacked_by"] = h
                self.on_engine[h] = idx
                self.battery[h] = max(0.0, self.battery[h] - self.cfg["hack_battery_cost"])
                ev[h] = {"valid": True, "kind": "hack"}
            else:
                ev[h] = self._implicit_wait_event("invalid_action")

        # autopilote des machines libres
        for i, m in enumerate(self.machines):
            if m["hacked_by"] is not None:
                continue
            tgt = self._machine_target(m)
            if m["move_pending"]:
                if tgt is None:
                    m["move_pending"] = False
                    self._rotate(m)
                else:
                    tx, ty, push = tgt
                    intents[("machine", i)] = (tx, ty)
                    if push is not None:
                        machine_push_plan[i] = push
            else:
                if tgt is None:
                    self._rotate(m)
                else:
                    m["move_pending"] = True

        # resolution look-ahead des collisions simultanees
        dest_count = {}
        for d in intents.values():
            dest_count[d] = dest_count.get(d, 0) + 1
        blocked = {k for k, d in intents.items() if dest_count[d] > 1}
        if self.cfg["avoid_competing_pushers"]:
            chest_targets = {}
            for h, plan in push_plan.items():
                if ("hero", h) not in blocked:
                    chest_targets.setdefault((plan[1], plan[2]), []).append(("hero", h))
            for i, plan in machine_push_plan.items():
                if ("machine", i) not in blocked:
                    chest_targets.setdefault((plan[1], plan[2]), []).append(("machine", i))
            for keys in chest_targets.values():
                if len(keys) > 1:
                    blocked.update(keys)

        for key, dest in intents.items():
            kind, ident = key
            if key in blocked:
                if kind == "machine":
                    self.machines[ident]["move_pending"] = False
                    self._rotate(self.machines[ident])
                    rider = self._rider(ident)
                    if rider:
                        self.hero_move_pending[rider] = False
                else:
                    ev[ident] = {"valid": False, "kind": "blocked"}
                continue

            if kind == "hero":
                h = ident
                if ev[h]["kind"] == "push":
                    _, cx, cy, bx, by, bt = push_plan[h]
                    idx = self.chest_at(cx, cy)
                    self.stamina[h] -= self.cfg["chest_push_stamina_cost"]
                    self.pos[h] = (cx, cy)
                    mi = self.machine_at(cx, cy)
                    if mi is not None:
                        self._transfer_machine_stones(mi)
                    drop = self.elev(cx, cy) - self.elev(bx, by)
                    if drop >= self.cfg["destroy_chest_drop"]:
                        del self.chests[idx]
                        self.chests_destroyed += 1
                        self.chests_destroyed_by[h] += 1
                        ev[h]["kind"] = "chest_cliff"
                    elif bt == "o":
                        del self.chests[idx]
                        self.chests_destroyed += 1
                        self.chests_destroyed_by[h] += 1
                        ev[h]["kind"] = "chest_hole"
                    elif bt == "@":
                        del self.chests[idx]
                        self.chests_hidden += 1
                        self.chests_hidden_by[h] += 1
                        ev[h]["kind"] = "chest_hidden"
                    else:
                        self.chests[idx][0], self.chests[idx][1] = bx, by
                        ev[h]["kind"] = "chest_pushed"
                else:
                    lx, ly = dest
                    self.stamina[h] -= ev[h].get("cost", MOVE_STAMINA)
                    self.pos[h] = (lx, ly)
                    if (lx, ly) in self.stones:
                        self.stones.discard((lx, ly))
                        self.stones_collected += 1
                        self.stones_by[h] += 1
                        ev[h]["stone"] = True
                    mi = self.machine_at(lx, ly)
                    if mi is not None:
                        self._transfer_machine_stones(mi)
                self.regen[h] = 0.0
            else:
                m = self.machines[ident]
                nx, ny = dest
                rider = self._rider(ident)
                cut = m["type"] == "G" and self.terrain[ny][nx] == "t"
                m["x"], m["y"] = nx, ny
                m["move_pending"] = False
                if cut:
                    self.terrain[ny][nx] = "."
                    if rider:
                        ev[rider]["kind"] = "cut"
                if (nx, ny) in self.stones:
                    self.stones.discard((nx, ny))
                    m["stones"] = int(m.get("stones", 0)) + 1
                if ident in machine_push_plan:
                    _, cx, cy, bx, by, bt = machine_push_plan[ident]
                    cidx = self.chest_at(cx, cy)
                    if cidx is not None:
                        drop = self.elev(cx, cy) - self.elev(bx, by)
                        if drop >= self.cfg["destroy_chest_drop"] or bt == "o":
                            del self.chests[cidx]
                            self.chests_destroyed += 1
                            if rider:
                                self.chests_destroyed_by[rider] += 1
                        elif bt == "@":
                            del self.chests[cidx]
                            self.chests_hidden += 1
                            if rider:
                                self.chests_hidden_by[rider] += 1
                        else:
                            self.chests[cidx][0], self.chests[cidx][1] = bx, by
                if m["hacked_by"] is None:
                    m["steps"] += 1
                    if m["type"] == "X" and m["steps"] >= self.cfg["machine_dig_every"]:
                        if self.terrain[ny][nx] != "#":
                            self.terrain[ny][nx] = "o"
                        m["steps"] = 0
                if rider:
                    self.pos[rider] = (nx, ny)
                    self.hero_move_pending[rider] = False
                    self._transfer_machine_stones(ident)

        # regeneration de stamina
        idle_kinds = {"wait", "unhack", "ride_wait", "hack_cw", "hack_ccw", "fill",
                      "hack_blocked_rotate", "cut", "implicit_wait", "hack_move_queued"}
        for h in ("F", "M"):
            if ev[h]["kind"] in idle_kinds or self.on_engine[h] is not None:
                self.regen[h] += 1.0
                if self.regen[h] >= self.cfg["stamina_recovery_interval"]:
                    self.stamina[h] = min(
                        100.0,
                        self.stamina[h] + self.cfg["stamina_recovery_rate"] * self.regen[h])
                    self.regen[h] = 0.0
            if ev[h]["kind"] == "unhack":
                mi = self.on_engine[h]
                if mi is not None:
                    self.machines[mi]["hacked_by"] = None
                self.on_engine[h] = None

        for h in ("F", "M"):
            if self.stamina[h] <= 0.0 and self.battery[h] <= 0.0:
                self.resource_low_ticks[h] += 1
            else:
                self.resource_low_ticks[h] = 0

        self.tick += 1
        return ev
