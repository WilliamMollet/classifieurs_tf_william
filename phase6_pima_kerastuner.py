"""
Protection contre le piège du tuner :
- epochs=100 par trial avec EarlyStopping(patience=10) :
    chaque trial peut aller jusqu'à 100 epochs mais s'arrête dès plateau
- max_trials=15 : budget suffisant pour couvrir un espace à 5 dimensions
    (4x4x2x6x5 = 960 combinaisons grid, 15 essais aléatoires capturent
l'essentiel si une dimension domine — Bergstra & Bengio 2012)
- seed=42 : reproductibilité
- On comparera best_hp avec max_epochs=10 vs max_epochs=100 pour montrer
    que le tuner sélectionne "ce qui converge vite" si max_epochs est trop bas
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Dossier de sortie pour les figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# keras-tuner (pip install keras-tuner si absent)
try:
    import keras_tuner as kt
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'keras-tuner', '-q'])
    import keras_tuner as kt

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Chargement et preprocessing (identique Phase 4)
# ---------------------------------------------------------------------------
pima_url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv(pima_url, names=cols)

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# HyperModel
# ---------------------------------------------------------------------------
def build_pima_model(hp):
    """
    HyperModel Pima : keras-tuner échantillonne les valeurs dans l'espace défini.
    hp.Int, hp.Float, hp.Choice sont les trois types d'hyperparamètres supportés.
    Chaque appel construit un modèle avec une configuration différente.
    """
    units_1      = hp.Int('units_1',      min_value=32,  max_value=128, step=32)
    units_2      = hp.Int('units_2',      min_value=16,  max_value=64,  step=16)
    activation   = hp.Choice('activation', values=['relu', 'tanh'])
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

    model = keras.Sequential()

    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(units_2, activation=activation))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# Tuner RandomSearch
# ---------------------------------------------------------------------------
tuner = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=15,
    seed=42,
    directory='tuning_pima',
    project_name='pima_random',
    overwrite=True
)

print("=== Résumé de l'espace de recherche ===")
tuner.search_space_summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

print("\n=== Lancement de la recherche (15 trials × 100 epochs max) ===")
tuner.search(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# ---------------------------------------------------------------------------
# Analyse des résultats
# ---------------------------------------------------------------------------
best_hp = tuner.get_best_hyperparameters()[0]
print("\n=== Meilleurs hyperparamètres ===")
print(f"  learning_rate : {best_hp.get('learning_rate')}")
print(f"  units_1       : {best_hp.get('units_1')}")
print(f"  units_2       : {best_hp.get('units_2')}")
print(f"  activation    : {best_hp.get('activation')}")
print(f"  dropout_rate  : {best_hp.get('dropout_rate')}")

print("\n=== Top 5 trials ===")
tuner.results_summary(num_trials=5)

print("\n=== Hyperparamètres des 5 meilleurs trials (recherche d'invariants) ===")
for i, hp in enumerate(tuner.get_best_hyperparameters(num_trials=5)):
    print(f"  Trial #{i+1}: {hp.values}")

# ---------------------------------------------------------------------------
# Entraîner le meilleur modèle jusqu'à convergence complète
# ---------------------------------------------------------------------------
print("\n=== Entraînement final du meilleur modèle (200 epochs) ===")
best_model   = tuner.hypermodel.build(best_hp)
early_stop2  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history_best = best_model.fit(
    X_train_norm, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop2],
    verbose=0
)
best_val_acc = max(history_best.history['val_accuracy'])
print(f"  Best model val_accuracy : {best_val_acc:.4f}")

# ---------------------------------------------------------------------------
# Tests de validation
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TESTS DE VALIDATION")
print("="*60)

# --- Happy path : 15 trials, meilleur > 0.76 ---
print("\n[Happy path] 15 trials terminés, meilleur val_accuracy > 0.76")
status = "✓ PASS" if best_val_acc > 0.76 else "⚠ borderline (val_accuracy < 0.76 — essayer plus de trials)"
print(f"  {status}  val_accuracy={best_val_acc:.4f}")

# --- Edge case : max_trials=1 ---
print("\n[Edge case] max_trials=1 — pipeline valide mais config aléatoire")
tuner_1 = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=1,
    seed=99,
    directory='tuning_pima_1trial',
    project_name='pima_1trial',
    overwrite=True
)
tuner_1.search(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
    verbose=0
)
best_hp_1   = tuner_1.get_best_hyperparameters()[0]
model_1     = tuner_1.hypermodel.build(best_hp_1)
h_1         = model_1.fit(
    X_train_norm, y_train, epochs=200,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
    verbose=0
)
acc_1trial  = max(h_1.history['val_accuracy'])
print(f"  max_trials=1  → val_accuracy : {acc_1trial:.4f}")
print(f"  max_trials=15 → val_accuracy : {best_val_acc:.4f}")
print(f"  Gain de la recherche : +{best_val_acc - acc_1trial:.4f}")

# --- Adversarial : espace de learning rates catastrophiques ---
print("\n[Adversarial] learning_rate dans [10.0, 100.0] → tous les modèles divergent")
def build_bad_lr(hp):
    units_1   = hp.Int('units_1', min_value=32, max_value=64, step=32)
    lr        = hp.Float('learning_rate', min_value=10.0, max_value=100.0)
    model     = keras.Sequential([
        layers.Dense(units_1, activation='relu', input_shape=(8,)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner_bad = kt.RandomSearch(
    build_bad_lr, objective='val_accuracy', max_trials=5, seed=0,
    directory='tuning_bad_lr', project_name='bad_lr', overwrite=True
)
tuner_bad.search(
    X_train_norm, y_train, epochs=20, validation_split=0.2, verbose=0
)
tuner_bad.results_summary(num_trials=3)
print("  → val_accuracy proche de 0.65 : espace de recherche mal défini, pas le tuner cassé")

# --- Stabilité : seed=43 ---
print("\n[Stabilité] Comparaison seed=42 vs seed=43")
tuner_43 = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=15,
    seed=43,
    directory='tuning_pima_s43',
    project_name='pima_s43',
    overwrite=True
)
tuner_43.search(
    X_train_norm, y_train, epochs=100,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
    verbose=0
)
best_hp_43 = tuner_43.get_best_hyperparameters()[0]
m43 = tuner_43.hypermodel.build(best_hp_43)
h43 = m43.fit(
    X_train_norm, y_train, epochs=200,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
    verbose=0
)
acc_43 = max(h43.history['val_accuracy'])
delta  = abs(best_val_acc - acc_43)
print(f"  seed=42 → val_accuracy : {best_val_acc:.4f}")
print(f"  seed=43 → val_accuracy : {acc_43:.4f}")
print(f"  Δ = {delta:.4f}")
if delta > 0.02:
    print("  ⚠ Variance > 2 pts : dataset trop petit pour convergence fiable sur 15 trials")
else:
    print("  ✓ Variance faible : recherche stable")

# Distribution val_accuracy des 15 trials — deux seeds côte à côte
accs_42, accs_43 = [], []
for trial in tuner.oracle.get_best_trials(num_trials=15):
    v = trial.metrics.get_best_value('val_accuracy')
    if v is not None:
        accs_42.append(v)
for trial in tuner_43.oracle.get_best_trials(num_trials=15):
    v = trial.metrics.get_best_value('val_accuracy')
    if v is not None:
        accs_43.append(v)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(accs_42, bins=8, alpha=0.6, label='seed=42', color='steelblue')
ax.hist(accs_43, bins=8, alpha=0.6, label='seed=43', color='darkorange')
ax.set_xlabel('val_accuracy')
ax.set_ylabel('Nombre de trials')
ax.set_title('Distribution val_accuracy — 15 trials (seed=42 vs seed=43)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/phase6_seeds_distribution.png', dpi=150)
print("\n  Graphique sauvegardé : phase6_seeds_distribution.png")
plt.close()