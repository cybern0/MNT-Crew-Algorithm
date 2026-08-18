from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
# Importer votre classe depuis features_extractor.py
from Model import GridScalarExtractor

# Configuration des arguments de la politique
policy_kwargs = dict(
    features_extractor_class=GridScalarExtractor,
    features_extractor_kwargs=dict(features_dim=128), # Taille de sortie du Combined
    lstm_hidden_size=128,                              # Taille de la mémoire LSTM gérée par SB3
    n_lstm_layers=1,
    net_arch=dict(pi=[64], vf=[64])                   # Couches MLP post-LSTM pour Actor (pi) et Critic (vf)
)

# Instanciation de RecurrentPPO
model = RecurrentPPO(
    policy=MultiInputLstmPolicy,
    env=env,
    device="cuda",
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,
    n_steps=128,  # Longueur de la séquence par environnement
)

# Lancement de l'entraînement
model.learn(total_timesteps=100_000)