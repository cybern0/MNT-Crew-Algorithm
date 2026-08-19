import json

import optuna
from sb3_contrib import MaskablePPO                                    # MIGRATION MASKABLEPPO
from AlgoTrain import AlgoGamesEnv, ContractEnv, ActionMasker

def objective(trial):
    # 1. Échantillonnage des hyperparamètres
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
    # lstm_size supprimé — MaskablePPO n’utilise pas de LSTM         # MIGRATION MASKABLEPPO

    # 2. Environnement et masquage d'actions (Héros F ou M)
    base_env = AlgoGamesEnv(map_path="map.txt", elevation_path="elevation.txt", hero="F")
    env = ActionMasker(ContractEnv(base_env, hero="F"), action_names=base_env.action_names)

    # 3. Modèle MaskablePPO (remplace RecurrentPPO)                    # MIGRATION MASKABLEPPO
    model = MaskablePPO(                                               # MIGRATION MASKABLEPPO
        "MultiInputPolicy",                                            # MIGRATION MASKABLEPPO
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        # policy_kwargs vide — le réseau est entièrement feedforward  # MIGRATION MASKABLEPPO
        verbose=0,
    )

    # 4. Entraînement d'évaluation (budget réduit)
    model.learn(total_timesteps=50_000)

    # 5. Calcul du score moyen
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        done, total_r = False, 0.0
        while not done:
            # predict() retourne l’action et les états (None)           # MIGRATION MASKABLEPPO
            action, _ = model.predict(obs, deterministic=True)         # MIGRATION MASKABLEPPO
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)

    return sum(rewards) / len(rewards)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Meilleurs hyperparamètres trouvés :", study.best_params)

with open("best_hyperparams.json", "w", encoding="utf-8") as f:
    json.dump(study.best_params, f, indent=4)

print("Paramètres sauvegardés avec succès dans best_hyperparams.json")