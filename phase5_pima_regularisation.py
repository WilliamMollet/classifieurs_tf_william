# =============================================================================
# Phase 5 : Régularisation et tuning sur Pima
# =============================================================================
# Objectif chiffré : recall classe 1 (diabétiques) supérieur de 5 points au baseline
# et val_accuracy >= 0.77
#
# Leviers dans l'ordre :
# 1. L2 (l2_lambda=0.01) : pénalise les grands poids, réduit l'overfitting
#    -> commencer par là car impact direct sur les poids, facile à isoler
# 2. Dropout(0.3) en complément de L2 : force la redondance des représentations
#    -> après L2 pour mesurer l'apport marginal du Dropout seul
# 3. class_weight si le modèle préfère toujours la classe majoritaire
# 4. Seuil de décision (0.5 -> 0.4) si le recall classe 1 reste trop bas
#
# Si on bat la cible : viser F1 macro > 0.72
# =============================================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Dossier de sortie pour les figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Distribution classes
print("=== Distribution classes ===")
counts = df['Outcome'].value_counts().sort_index()
total = len(df)
for cls, cnt in counts.items():
    print(f"  {cls}  {cnt}  ({cnt/total*100:.1f}%)")

# Zéros suspects (NaN déguisés)
print("\n=== Colonnes avec des zéros suspects ===")
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    print(f"  {col:25s}: {(df[col] == 0).sum()} zéros")

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Fonction de construction
# ---------------------------------------------------------------------------
def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
    """
    Modèle Pima avec régularisation L2 optionnelle et Dropout optionnel.
    Si use_dropout=True, insère un Dropout(0.3) après chaque couche cachée.
    """
    model = keras.Sequential()

    model.add(layers.Dense(
        64, activation='relu', input_shape=(8,),
        kernel_regularizer=regularizers.l2(l2_lambda)
    ))
    if use_dropout:
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(
        32, activation='relu',
        kernel_regularizer=regularizers.l2(l2_lambda)
    ))
    if use_dropout:
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# Callback Early Stopping commun
# ---------------------------------------------------------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

# ---------------------------------------------------------------------------
# Config 1 : Baseline non régularisé
# ---------------------------------------------------------------------------
print("\n=== Config 1 : Baseline (sans régularisation) ===")
model_baseline = build_pima_regularized(l2_lambda=0.0, use_dropout=False)
history_baseline = model_baseline.fit(
    X_train_norm, y_train,
    epochs=300,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)
epochs_baseline = len(history_baseline.history['val_loss'])
acc_baseline    = max(history_baseline.history['val_accuracy'])
print(f"  Epoch d'arrêt : {epochs_baseline}")
print(f"  Max val_accuracy : {acc_baseline:.4f}")

# ---------------------------------------------------------------------------
# Config 2 : L2 seul
# ---------------------------------------------------------------------------
print("\n=== Config 2 : L2 seul (l2_lambda=0.01) ===")
model_l2 = build_pima_regularized(l2_lambda=0.01, use_dropout=False)
history_l2 = model_l2.fit(
    X_train_norm, y_train,
    epochs=300,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)
epochs_l2 = len(history_l2.history['val_loss'])
acc_l2    = max(history_l2.history['val_accuracy'])
print(f"  Epoch d'arrêt : {epochs_l2}")
print(f"  Max val_accuracy : {acc_l2:.4f}")

# ---------------------------------------------------------------------------
# Config 3 : L2 + Dropout
# ---------------------------------------------------------------------------
print("\n=== Config 3 : L2 + Dropout ===")
model_l2_drop = build_pima_regularized(l2_lambda=0.01, use_dropout=True)
history_l2_drop = model_l2_drop.fit(
    X_train_norm, y_train,
    epochs=300,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)
epochs_l2_drop = len(history_l2_drop.history['val_loss'])
acc_l2_drop    = max(history_l2_drop.history['val_accuracy'])
print(f"  Epoch d'arrêt : {epochs_l2_drop}")
print(f"  Max val_accuracy : {acc_l2_drop:.4f}")

# ---------------------------------------------------------------------------
# Graphique : 3 courbes val_loss côte à côte
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
configs_plot = [
    ("Baseline",    history_baseline, epochs_baseline, 'steelblue'),
    ("L2 seul",     history_l2,       epochs_l2,       'darkorange'),
    ("L2 + Dropout",history_l2_drop,  epochs_l2_drop,  'seagreen'),
]
for ax, (title, hist, stop_ep, color) in zip(axes, configs_plot):
    ax.plot(hist.history['val_loss'], color=color, label='val_loss')
    ax.plot(hist.history['loss'],     color=color, linestyle='--', alpha=0.5, label='train_loss')
    ax.axvline(x=stop_ep - 1, color='red', linestyle=':', linewidth=1.5, label=f'Stop ep {stop_ep}')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/phase5_pima_3configs.png', dpi=150)
print("\n  Graphique sauvegardé : phase5_pima_3configs.png")
plt.close()

# ---------------------------------------------------------------------------
# Tests de validation
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TESTS DE VALIDATION")
print("="*60)

# --- Happy path ---
print("\n[Happy path] val_accuracy > 70% sur au moins une config")
for name, acc in [("Baseline", acc_baseline), ("L2", acc_l2), ("L2+Dropout", acc_l2_drop)]:
    status = "✓ PASS" if acc > 0.70 else "✗ FAIL"
    print(f"  {status}  {name}: {acc:.4f}")

# --- Happy path : Early Stopping s'active avant 300 epochs ---
print("\n[Happy path] Early Stopping avant 300 epochs")
for name, ep in [("Baseline", epochs_baseline), ("L2", epochs_l2), ("L2+Dropout", epochs_l2_drop)]:
    status = "✓ PASS" if ep < 300 else "✗ FAIL (patience trop grande ou modèle qui continue)"
    print(f"  {status}  {name}: stoppé à epoch {ep}")

# --- Edge case : patience=1 (underfitting par arrêt prématuré) ---
print("\n[Edge case] patience=1 → underfitting par arrêt prématuré")
es_p1 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
m_p1  = build_pima_regularized(l2_lambda=0.01, use_dropout=False)
h_p1  = m_p1.fit(
    X_train_norm, y_train, epochs=300,
    validation_split=0.2, callbacks=[es_p1], verbose=0
)
ep_p1  = len(h_p1.history['val_loss'])
acc_p1 = max(h_p1.history['val_accuracy'])
print(f"  patience=1  → epochs réels : {ep_p1}, val_accuracy : {acc_p1:.4f}")
print(f"  patience=15 → epochs réels : {epochs_l2}, val_accuracy : {acc_l2:.4f}")
print(f"  Écart epochs : {epochs_l2 - ep_p1} | Écart accuracy : {acc_l2 - acc_p1:.4f}")
status = "✓ PASS" if ep_p1 < epochs_l2 else "✗ FAIL"
print(f"  {status}  patience=1 stoppe bien avant patience=15")

# --- Adversarial : l2_lambda=10.0 → underfitting par excès de régularisation ---
print("\n[Adversarial] l2_lambda=10.0 → pénalité écrase la loss")
m_over_l2 = build_pima_regularized(l2_lambda=10.0, use_dropout=False)
h_over    = m_over_l2.fit(
    X_train_norm, y_train, epochs=100,
    validation_split=0.2, verbose=0
)
acc_over   = max(h_over.history['val_accuracy'])
poids_mean = m_over_l2.layers[0].get_weights()[0].mean()
print(f"  val_accuracy avec l2=10.0 : {acc_over:.4f}")
print(f"  Moyenne poids couche 0    : {poids_mean:.6f}")
status = "✓ PASS" if acc_over < 0.70 else "✗ (régularisation pas assez forte pour ce run)"
print(f"  {status}  Poids proches de zéro confirme l'underfitting par excès de régularisation")

# --- Adversarial : vérifier que le modèle n'est pas un prédicteur de classe majoritaire ---
print("\n[Adversarial] Le modèle prédit-il réellement des positifs ?")
pred_mean = model_l2.predict(X_test_norm, verbose=0).mean()
print(f"  model.predict(X_test).mean() = {pred_mean:.4f}  (attendu proche de 0.35)")
status = "✓ PASS" if 0.15 < pred_mean < 0.55 else "✗ FAIL — le modèle préfère la classe majoritaire"
print(f"  {status}")

print("\n=== Résumé final ===")
print(f"  Baseline   : val_accuracy={acc_baseline:.4f}, arrêt epoch {epochs_baseline}")
print(f"  L2         : val_accuracy={acc_l2:.4f}, arrêt epoch {epochs_l2}")
print(f"  L2+Dropout : val_accuracy={acc_l2_drop:.4f}, arrêt epoch {epochs_l2_drop}")