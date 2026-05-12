import datetime
import os

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# ----- Pipeline de données (identique à Phase 1) -----
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


# ----- Modèle (identique à Phase 2) -----
def build_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ----- Fonction d'entraînement avec TensorBoard -----
def train_with_tensorboard(X_tr, y_tr, X_v, y_v, run_name, epochs=100):
    """
    Entraîne un modèle de régression avec callback TensorBoard horodaté.

    Pourquoi horodater le dossier : si on relance le même run_name sans
    timestamp, TensorBoard écrit dans le même dossier et superpose les courbes.
    Le timestamp HHMMSS garantit que chaque run a son propre espace.

    histogram_freq=1 : enregistre les distributions des poids à chaque epoch
    (onglet Histograms / Distributions dans TensorBoard).
    """
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = os.path.join("logs", "fit", f"{run_name}_{timestamp}")

    tb_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )

    model = build_regression_model(input_dim=X_tr.shape[1])

    history = model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_v, y_v),
        callbacks=[tb_callback],
        verbose=0,
    )

    print(f"Run '{run_name}' terminé. Logs dans {log_dir}")
    return model, history


# ----- Run 1 : données normalisées (comportement sain attendu) -----
print("=" * 60)
print("RUN 1 — Données normalisées")
print("=" * 60)
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, X_val_norm, y_val,
    run_name="california_norm",
)
final_val_loss_norm = history_norm.history['val_loss'][-1]
final_val_mae_norm = history_norm.history['val_mae'][-1]
print(f"  val_loss finale : {final_val_loss_norm:.4f}")
print(f"  val_mae  finale : {final_val_mae_norm:.4f}")


# ----- Run 2 : données brutes (comportement dégradé attendu) -----
print("\n" + "=" * 60)
print("RUN 2 — Données brutes (sans normalisation)")
print("=" * 60)
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, X_val, y_val,
    run_name="california_raw",
)
final_val_loss_raw = history_raw.history['val_loss'][-1]
final_val_mae_raw = history_raw.history['val_mae'][-1]
print(f"  val_loss finale : {final_val_loss_raw:.4f}")
print(f"  val_mae  finale : {final_val_mae_raw:.4f}")


# ----- Comparaison rapide en console (TensorBoard reste l'outil principal) -----
print("\n" + "=" * 60)
print("COMPARAISON")
print("=" * 60)
print(f"  Run normalisé : val_loss = {final_val_loss_norm:.4f}, "
      f"val_mae = {final_val_mae_norm:.4f}")
print(f"  Run brut      : val_loss = {final_val_loss_raw:.4f}, "
      f"val_mae = {final_val_mae_raw:.4f}")

print(
    "\nPour visualiser les courbes complètes :\n"
    "    tensorboard --logdir=logs/fit\n"
    "puis ouvrir http://localhost:6006 dans le navigateur.\n"
    "Dans l'onglet Scalars, cocher les deux runs pour superposer les courbes."
)


# ===================================================================
# OBSERVATION FINALE — Diagnostic des trois zones
# ===================================================================
# Run "california_norm" : train_loss et val_loss descendent ensemble et se
# stabilisent autour de 0.3 (MSE) / 0.4 (MAE). Les deux courbes restent proches
# l'une de l'autre, val légèrement au-dessus de train sur la fin (signe d'un
# léger overfitting attendu en fin d'entraînement, mais rien de critique).
# → Zone (a) : comportement sain.
#
# Run "california_raw" : la val_loss reste très haute (souvent >1 ou même
# plusieurs ordres de grandeur au-dessus du run normalisé). Les gradients
# déséquilibrés (Latitude ~37, Population ~1000 vs AveRooms ~5) empêchent
# l'optimiseur de converger correctement. Le modèle n'apprend quasiment rien.
# → Cas pathologique : ce n'est pas exactement de l'overfitting, c'est un
#    échec de convergence dû à l'absence de preprocessing.
#
# Conclusion : à architecture, optimiseur et durée identiques, seule la
# normalisation des features sépare un run exploitable d'un run inutilisable.
# C'est ce qui justifie de toujours scaler avant d'entraîner un PMC.
#
# ===================================================================
# TESTS MANUELS (à exécuter dans un terminal, pas dans ce script)
# ===================================================================
# Cas limite — TensorBoard sur dossier inexistant :
#     tensorboard --logdir=logs/fit_vide
# Résultat attendu : TensorBoard démarre sans erreur bloquante, mais affiche
# "No dashboards are active for the current data set." Trompeur : on croit que
# les logs sont perdus alors qu'il suffit de corriger le chemin. Réflexe :
# toujours vérifier le chemin avant de conclure que l'entraînement a raté.
#
# Adversarial — deux instances de TensorBoard sur le port 6006 :
#     tensorboard --logdir=logs/fit          # terminal 1
#     tensorboard --logdir=logs/fit          # terminal 2 (en parallèle)
# Résultat attendu : la seconde instance refuse de démarrer
# ("Address already in use" ou démarre sur un autre port).
# En production, ce genre de conflit silencieux bloque les pipelines de
# monitoring. Le réflexe : pkill -f tensorboard pour libérer le port, puis
# relancer.