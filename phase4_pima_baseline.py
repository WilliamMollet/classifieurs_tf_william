"""
- Pima est déséquilibré : ~65% classe 0 (non-diabétique), ~35% classe 1.
- Un modèle naïf qui prédit toujours 0 affiche déjà 65% d'accuracy.
- Donc un modèle qui plafonne à 65% n'a rien appris.
- Vérification clé : model.predict(X_val).mean() doit être proche de 0.35
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# Flags
RUN_ADVERSARIAL = False
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ----- Visualisation -----
def plot_history(history, title, filename):
    """Trace loss et accuracy par epoch (train vs val) côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary crossentropy')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {filename}")


# ----- Chargement Pima -----
pima_url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
            "master/pima-indians-diabetes.data.csv")
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)


# ----- Inspection : distribution et zéros suspects -----
print("=" * 60)
print("INSPECTION — Distribution et zéros suspects")
print("=" * 60)

counts = df['Outcome'].value_counts().sort_index()
total = len(df)
print("\nDistribution classes :")
for cls, n in counts.items():
    print(f"  Classe {cls} : {n:4d}  ({n / total * 100:.1f}%)")

# Baseline naïf : prédiction systématique de la classe majoritaire
majority_class = counts.idxmax()
naive_acc = counts.max() / total
print(f"\nClassificateur naïf (prédire toujours {majority_class}) : "
      f"accuracy = {naive_acc:.4f}")
print("Notre modèle DOIT dépasser cette valeur pour être considéré utile.")

print("\nColonnes avec des zéros (potentiels NaN cachés) :")
zeros_per_col = (df[cols[:-1]] == 0).sum()
print(zeros_per_col)
suspicious = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(f"\nLes 5 colonnes physiologiquement suspectes : {suspicious}")


# ----- Split + scaling -----
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)


# ----- Modèle baseline -----
def build_pima_baseline(input_dim=8):
    """PMC binaire : Dense(64) → Dense(32) → Dense(1, sigmoid)."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ----- Entraînement -----
print("\n" + "=" * 60)
print("ENTRAÎNEMENT BASELINE")
print("=" * 60)

model = build_pima_baseline()
model.summary()

history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)


# ----- Évaluation -----
test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
val_acc_max = max(history.history['val_accuracy'])
predict_mean = float(model.predict(X_test_norm, verbose=0).mean())

print(f"\nval_accuracy max (sur tous les epochs) : {val_acc_max:.4f}")
print(f"test_accuracy                          : {test_acc:.4f}")
print(f"model.predict(X_test).mean()           : {predict_mean:.4f}")
print(f"Baseline naïf à battre                 : {naive_acc:.4f}")


# ----- Diagnostics du baseline -----
print("\n" + "=" * 60)
print("DIAGNOSTIC")
print("=" * 60)

if val_acc_max > 0.70:
    print(f"OK : val_accuracy max = {val_acc_max:.4f} > 0.70 (happy path).")
else:
    print(f"⚠ val_accuracy max = {val_acc_max:.4f} ≤ 0.70 (sous-performant).")

if test_acc <= naive_acc + 0.01:
    print(f"⚠ red flag : test_acc ({test_acc:.4f}) ≈ baseline naïf ({naive_acc:.4f}).")

if predict_mean < 0.20:
    print(
        f"⚠ predict mean = {predict_mean:.4f} < 0.20 : le modèle prédit\n"
        f"  quasi-uniquement la classe majoritaire (effondrement sur 0).\n"
        f"  → Activer RUN_ADVERSARIAL pour relancer avec class_weight."
    )
elif predict_mean > 0.55:
    print(
        f"⚠ predict mean = {predict_mean:.4f} > 0.55 : le modèle surprédit\n"
        f"  la classe positive (1) — étrange étant donné le déséquilibre."
    )
else:
    print(
        f"OK : predict mean = {predict_mean:.4f} ∈ [0.20, 0.55] — le modèle\n"
        f"  voit bien des cas positifs, pas effondré sur la classe majoritaire."
    )


# ----- Plot history -----
plot_history(
    history,
    title="Baseline Pima — binary crossentropy & accuracy (100 epochs)",
    filename=os.path.join(FIGURES_DIR, "phase4_pima_baseline_history.png"),
)


# ===================================================================
# TESTS ADVERSARIAUX (RUN_ADVERSARIAL=False par défaut)
# ===================================================================
if RUN_ADVERSARIAL:
    print("\n" + "=" * 60)
    print("EDGE CASE — Imputation par médiane des zéros suspects")
    print("=" * 60)

    df_imp = df.copy()
    for col in suspicious:
        med = df_imp.loc[df_imp[col] != 0, col].median()
        df_imp[col] = df_imp[col].replace(0, med)

    X_imp = df_imp.drop('Outcome', axis=1).values
    X_imp_train, X_imp_test, y_imp_train, y_imp_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler_imp = StandardScaler().fit(X_imp_train)
    X_imp_train_norm = scaler_imp.transform(X_imp_train)
    X_imp_test_norm = scaler_imp.transform(X_imp_test)

    model_imp = build_pima_baseline()
    hist_imp = model_imp.fit(
        X_imp_train_norm, y_imp_train,
        epochs=100, batch_size=32, validation_split=0.2, verbose=0,
    )
    _, test_acc_imp = model_imp.evaluate(X_imp_test_norm, y_imp_test, verbose=0)
    val_acc_imp = max(hist_imp.history['val_accuracy'])

    print(f"\nval_accuracy max avec imputation médiane : {val_acc_imp:.4f}")
    print(f"  baseline (sans imputation)              : {val_acc_max:.4f}")
    print(f"  écart                                   : "
          f"{val_acc_imp - val_acc_max:+.4f}")

    plot_history(
        hist_imp,
        title="Pima — avec imputation médiane des zéros suspects",
        filename=os.path.join(FIGURES_DIR, "phase4_pima_imputed_history.png"),
    )

    # ---- Class weight fallback (seulement si predict_mean trop bas) ----
    if predict_mean < 0.20:
        print("\n" + "=" * 60)
        print("ADVERSARIAL — Class weight pour rééquilibrer")
        print("=" * 60)
        # Ratio inverse de fréquence : ~1.0 / 1.9 pour 65/35
        class_weight = {0: 1.0, 1: 1.9}
        model_cw = build_pima_baseline()
        hist_cw = model_cw.fit(
            X_train_norm, y_train,
            epochs=100, batch_size=32, validation_split=0.2,
            class_weight=class_weight, verbose=0,
        )
        predict_mean_cw = float(model_cw.predict(X_test_norm, verbose=0).mean())
        val_acc_cw = max(hist_cw.history['val_accuracy'])
        print(f"\nval_accuracy max (class_weight) : {val_acc_cw:.4f}")
        print(f"predict mean (class_weight)     : {predict_mean_cw:.4f}")
    else:
        print(
            "\n[Class weight skip] predict_mean dans la plage saine, pas besoin\n"
            "de rééquilibrer manuellement."
        )