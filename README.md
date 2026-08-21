# MNT-Crew-Algorithm

Entraînement, export et rejeu d'agents par apprentissage par renforcement (MaskablePPO) pour **AlgoGames 2** — deux héros, **Ikotofosa (F)** et **Imahaki (M)**, chacun piloté par sa propre politique, exportée en **ONNX** pour être consommée par un runtime externe (ex : `GameRunner.java` / moteur de jeu C#/UE).

Le moteur de simulation (`GameEngine.py`) réimplémente fidèlement les règles du GDD et du "Twist" (élévation, hacking, autopilote des machines, push de coffres...) telles que définies par le `GameRunner` officiel, afin que le score et le comportement obtenus ici soient reproductibles côté runner.

---

## Sommaire

- [Aperçu du jeu](#aperçu-du-jeu)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Formats de fichiers](#formats-de-fichiers)
- [Pipeline complet](#pipeline-complet)
  - [1. Générer une carte](#1-générer-une-carte)
  - [2. Entraîner un modèle](#2-entraîner-un-modèle)
  - [3. Recherche d'hyperparamètres (Optuna)](#3-recherche-dhyperparamètres-optuna)
  - [4. Exporter en ONNX](#4-exporter-en-onnx)
  - [5. Rejouer avec les modèles ONNX](#5-rejouer-avec-les-modèles-onnx)
- [Contrat d'observation (15 canaux + 6 scalaires)](#contrat-dobservation-15-canaux--6-scalaires)
- [Catalogue d'actions](#catalogue-dactions)
- [Formule de score](#formule-de-score)
- [Tests](#tests)

---

## Aperçu du jeu

Deux héros évoluent sur une grille (map ASCII + carte d'élévation) :

- **Ikotofosa (`F`)** — peut hacker les **excavateurs (`X`)**, seul héros à disposer de `HACK_FILL` (comblement de trou).
- **Imahaki (`M`)** — peut hacker les **grapplins (`G`)**, qui coupent automatiquement les arbres (`HACK_CUT` implicite, non émis comme action).

Objectif : ramasser les pierres précieuses (`+`) et cacher les coffres (`*`) sur les cases `@`, en gérant stamina et batterie, sous une limite de temps.

Chaque héros est contrôlé par sa **propre politique** (deux réseaux, deux modèles ONNX distincts), entraînée indépendamment via `AlgoTrain.py`.

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `GameEngine.py` | Simulateur déterministe des règles GDD + Twist (mouvement, push, hacking, autopilote machines, scoring). Aucune dépendance à gymnasium/torch. |
| `AlgoEnv.py` | Environnement Gymnasium (`AlgoEnv`) qui pilote `GameEngine.py` : encodage de l'observation (15 canaux + 6 scalaires), calcul de la récompense d'entraînement. |
| `AlgoTrain.py` | Script principal d'entraînement MaskablePPO (un modèle par héros). Contient aussi le catalogue d'actions, les constantes d'observation et les utilitaires de lecture de `map.txt`/`elevation.txt`. |
| `Optuna.py` | Recherche d'hyperparamètres MaskablePPO via Optuna, mêmes flags que `AlgoTrain.py`. |
| `Exports.py` | Exporte les modèles SB3 (`ModelStates/*.zip`) en `.onnx` (`OnnxModels/*.onnx`), avec vérification du nombre d'actions attendu. |
| `PlayOnnx.py` | Rejoue une partie avec les deux modèles `.onnx` (sans dépendance à stable-baselines3), écrit `actions.txt` au format GDD. |
| `MapGenerator.py` | Génère une paire `map.txt` / `elevation.txt` valide et conforme aux contraintes du Twist. |
| `test_play_onnx.py` | Tests de non-régression (formule de score, masque d'actions, format de sortie). |
| `requirements.txt` | Dépendances Python du projet. |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Dépendances principales : `numpy`, `gymnasium`, `stable-baselines3`, `sb3-contrib` (MaskablePPO), `torch`, `onnx`, `onnxruntime`, `optuna`, `pytest`.

## Formats de fichiers

### `map.txt`
```
H W tempsLimite
<H lignes de W caractères>
```

Symboles ASCII autorisés : `.` (vide), `#` (rocher), `t` (arbre), `o` (trou), `@` (cache), `+` (pierre précieuse), `*` (coffre), `F`/`M` (héros), `X`/`G` (machines). La carte doit contenir **exactement un `F` et un `M`**.

### `elevation.txt`
`H` lignes de `W` chiffres (0-9), alignées case à case sur `map.txt`. `0` est réservé aux rochers.

### `actions.txt` (sortie de `PlayOnnx.py`)
```
ACTION_F | ACTION_M
...
END_GAME
```
Une ligne `"ACTION_F | ACTION_M"` par tick, `END_GAME` en dernière ligne — format directement compatible avec `GameRunner.java`.

## Pipeline complet

### 1. Générer une carte

```bash
python MapGenerator.py --height 10 --width 12 --out-map map.txt --out-elevation elevation.txt
```

Options utiles : `--seed`, `--chests`, `--stones`, `--holes`, `--trees`, `--bushes`, `--rock-ratio`, `--max-time`. Le générateur garantit : un `F` et un `M` connectés, au moins un `X` et un `G` accessibles, au moins un coffre et une cache, et le respect des contraintes d'élévation du Twist.

### 2. Entraîner un modèle

Un modèle par héros (deux invocations distinctes) :

```bash
python AlgoTrain.py --hero F --map map.txt --elevation elevation.txt --output ModelStates/ikotofosa
python AlgoTrain.py --hero M --map map.txt --elevation elevation.txt --output ModelStates/imahaki
```

Options principales : `--timesteps`, `--n-envs`, `--seed`, `--device` (`auto`/`cpu`/`cuda`), `--learning-rate`, `--n-steps`, `--batch-size`, `--gamma`, `--gae-lambda`, `--ent-coef`, `--augmentation` (`identity`, `transpose`, `rotate90/180/270`, `mirror_horizontal/vertical`, ou `random`/`all`), `--resume` (reprise depuis un `.zip`), `--check-env` (validation Gymnasium).

Optimisations CPU : `--no-eval`, `--no-progress-bar`.

Les logs d'évolution de la carte sont écrits dans `--log-dir` (défaut `TrainingLogs/`), tous les `--log-episodes` épisodes (défaut 50).

### 3. Recherche d'hyperparamètres (Optuna)

Même jeu de flags que `AlgoTrain.py`, lançable en parallèle pour F et M :

```bash
CUDA_VISIBLE_DEVICES=0 python Optuna.py --hero F \
  --map map.txt --elevation elevation.txt \
  --output ModelStates/ikotofosa --device cuda --augmentation mirror_horizontal \
  --timesteps 10000 --n-envs 8 \
  > optuna_log_F.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python Optuna.py --hero M \
  --map map.txt --elevation elevation.txt \
  --output ModelStates/imahaki --device cuda --augmentation mirror_horizontal \
  --timesteps 10000 --n-envs 8 \
  > optuna_log_M.txt 2>&1 &
wait
```

Les meilleurs hyperparamètres trouvés sont écrits dans `OptunaParams/`.

### 4. Exporter en ONNX

```bash
python Exports.py --list                 # lister les archives ModelStates/*.zip disponibles
python Exports.py --hero F  --export      # exporte ikotofosa.onnx (14 actions)
python Exports.py --hero M  --export      # exporte imahaki.onnx  (13 actions)
python Exports.py --hero both --export    # exporte les deux
python Exports.py --all --export          # exporte toutes les archives trouvées
```

Chaque modèle attend un tenseur `map` de forme `[1, 15, 30, 30]` et un tenseur `stats` de forme `[1, 6]`, et renvoie des **logits bruts** (`[1, n_actions]`) — l'argmax (masqué) est à la charge de l'appelant. Un contrôle (`_verify_action_count`) empêche d'exporter un modèle dont le nombre d'actions ne correspond pas au héros ciblé.

### 5. Rejouer avec les modèles ONNX

```bash
python PlayOnnx.py --map map.txt --elevation elevation.txt \
    --onnx-f OnnxModels/ikotofosa.onnx --onnx-m OnnxModels/imahaki.onnx \
    --actions actions.txt
```

Options utiles : `--max-ticks` (surcharge le temps limite lu dans `map.txt`), `--tick-delay`, `--log-every`, `--no-render`, `--seed`, `--selftest` (lance `pytest test_play_onnx.py` avant de jouer). `PlayOnnx.py` ne dépend pas de `stable-baselines3` : c'est le même contrat d'inférence qu'un runtime externe (ONNX Runtime pur).

Le score final affiché (`score final officiel (GDD)`) utilise exactement la même formule que `GameRunner.java`.

## Contrat d'observation (15 canaux + 6 scalaires)

L'observation `grid` (`[15, 30, 30]`, toujours paddée à 30x30 même sur une carte plus petite) est **égocentrée** : elle est construite séparément pour chaque héros et ne marque que sa propre position (pas celle du coéquipier).

| Canal | Contenu |
|---|---|
| 0-8 | one-hot tuile : `.` `#` `*` `o` `t` `@` `+` `X` `G` |
| 9 | position du héros courant |
| 10 | case juste devant le héros (direction "facing"), si dans les limites |
| 11 | élévation absolue normalisée `(elev/4.5) - 1.0` |
| 12 | élévation relative normalisée `clip((elev - moyenne)/4.5, -1, 1)` |
| 13 | look-ahead position future de l'excavateur `X` (2 ticks) |
| 14 | look-ahead position future du grappin `G` (2 ticks) |

Les canaux 13/14 ne sont remplis que si un coffre se trouve dans la case juste devant le héros (`lookahead_only_if_chest_ahead`).

`stats` (`[6]`) : `stamina/100`, `batterie/100`, `(tempsLimite - tick)/tempsLimite`, `x/(W-1)`, `y/(H-1)`, `1.0 si sur machine piratée sinon 0.0` — toutes clippées `[0, 1]`.

## Catalogue d'actions

**Ikotofosa (F) — 14 actions**
```
UP, DOWN, LEFT, RIGHT, WAIT,
PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT,
HACK, HACK_MOVE, HACK_FILL, HACK_CW, HACK_CCW
```

**Imahaki (M) — 13 actions** (pas de `HACK_FILL` : coupe d'arbre automatique côté grappin)
```
UP, DOWN, LEFT, RIGHT, WAIT,
PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT,
HACK, HACK_MOVE, HACK_CW, HACK_CCW
```

**Masque d'actions légales** (`action_mask()` dans `AlgoTrain.py`, miroir de `GameEngine.legal_action_mask`) :
- hors machine piratée → `MOVE/WAIT/PUSH_*/HACK` valides, tout `HACK_*` invalide ;
- sur machine piratée → seuls `WAIT` et les `HACK_*` sont valides.

## Formule de score

```
score = (coffres_cachés × 150) + (pierres_ramassées × 25)
      + (stamina_F + stamina_M + batterie_F + batterie_M) // 4
      + (tempsLimite − tick_actuel) // 2
```
(divisions entières — voir `GameEngine.official_score()` et `test_play_onnx.py`).

## Tests

```bash
pytest test_play_onnx.py -q
```

Couvre : la formule de score officielle (exemple chiffré du GDD), le masque d'actions (hors/sur machine), et le format de sortie `actions.txt`.

---

*Ce dépôt est conçu pour rester interopérable avec `GameRunner.java` (référence officielle des règles AlgoGames 2) : mêmes formats de fichiers, même catalogue d'actions, mêmes constantes physiques et même formule de score.*
