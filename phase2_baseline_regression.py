"""
Note sur l'absence de sigmoid en sortie :
    Une variante "tentante" ajouterait activation='sigmoid' à la couche de sortie.
    Le code tournerait, la loss descendrait, mais le modèle serait inutilisable :
    sigmoid borne la sortie dans [0, 1], or les prix médians de Californie vont
    jusqu'à ~5 (en centaines de milliers de $). Le réseau ne pourrait jamais
    prédire au-delà de 1 — MAE et MSE seraient artificiellement plafonnées.
    Pour une régression à cible non bornée, on laisse la couche de sortie LINÉAIRE
    (aucune activation).

Note sur le choix MSE comme loss et MAE comme métrique :
    - MSE pénalise fort les grosses erreurs (carré) → guide l'optimiseur à les éviter.
    - MAE est plus lisible : 0.6 = 60 000 $ d'erreur moyenne.
    Les deux coexistent : MSE pour optimiser, MAE pour interpréter.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# Flag pour les tests adversariaux (entraînements supplémentaires lents)
RUN_ADVERSARIAL = False

# Dossier de sortie pour les figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ----- Utilitaires de visualisation -----
def plot_history(history, title, filename):
    """Trace loss (MSE) et MAE par epoch, train vs val, côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='train')
    axes[1].plot(history.history['val_mae'], label='val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (centaines de milliers de $)')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {filename}")


def plot_compare_loss(histories, title, filename, ylabel='val_loss (MSE)',
                      key='val_loss'):
    """Trace plusieurs courbes (une par run) sur un même axe pour comparaison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        ax.plot(history.history[key], label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure sauvegardée : {filename}")


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


# ----- Modèle -----
def build_regression_model(input_dim):
    """
    PMC de régression : Dense(64, relu) → Dense(32, relu) → Dense(1) linéaire.
    Compilation : optimizer=adam, loss=mse, metric=mae.
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),  # pas d'activation : régression à cible non bornée
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ----- Entraînement principal -----
print("=" * 60)
print("ENTRAÎNEMENT BASELINE — California Housing régression")
print("=" * 60)

model = build_regression_model(input_dim=8)
model.summary()

history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_norm, y_val),
    verbose=1,
)


# ----- Évaluation sur test -----
test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\nMSE test : {test_loss:.4f}")
print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")
print(f"        ≈ {test_mae * 100_000:,.0f} $ d'erreur moyenne")


# ----- Happy path : assertions -----
assert len(model.layers) == 3, "Le modèle doit contenir 3 couches Dense."
assert model.layers[-1].units == 1, "La couche de sortie doit avoir 1 neurone."
assert test_mae < 1.0, f"MAE attendue < 1.0, obtenue : {test_mae:.4f}"
print("\nOK : architecture 3 couches, sortie 1 neurone, MAE test < 1.0.")


# ----- Graphiques d'évolution loss/MAE -----
plot_history(
    history,
    title="Baseline régression — California Housing (100 epochs)",
    filename=os.path.join(FIGURES_DIR, "phase2_baseline_history.png"),
)


# ===================================================================
# TESTS ADVERSARIAUX (désactivés par défaut — passer RUN_ADVERSARIAL=True)
# ===================================================================
if RUN_ADVERSARIAL:
    print("\n" + "=" * 60)
    print("EDGE CASE — Effet du batch_size")
    print("=" * 60)
    # batch_size=1 (SGD pur) : 13209 updates par epoch, très bruité mais converge vite
    # en nombre d'epochs.
    # batch_size=len(X_train) (Batch GD) : 1 update par epoch, courbes lisses mais lent.
    EDGE_EPOCHS = 10

    print(f"\n--- SGD pur (batch_size=1, {EDGE_EPOCHS} epochs) ---")
    model_sgd_pure = build_regression_model(input_dim=8)
    hist_sgd_pure = model_sgd_pure.fit(
        X_train_norm, y_train,
        epochs=EDGE_EPOCHS, batch_size=1,
        validation_data=(X_val_norm, y_val),
        verbose=0,
    )
    print(f"  val_loss après {EDGE_EPOCHS} epochs : "
          f"{hist_sgd_pure.history['val_loss'][-1]:.4f}")

    print(f"\n--- Batch GD complet (batch_size=full, {EDGE_EPOCHS} epochs) ---")
    model_batch_full = build_regression_model(input_dim=8)
    hist_batch_full = model_batch_full.fit(
        X_train_norm, y_train,
        epochs=EDGE_EPOCHS, batch_size=len(X_train_norm),
        validation_data=(X_val_norm, y_val),
        verbose=0,
    )
    print(f"  val_loss après {EDGE_EPOCHS} epochs : "
          f"{hist_batch_full.history['val_loss'][-1]:.4f}")

    plot_compare_loss(
        {'SGD pur (batch=1)': hist_sgd_pure,
         'Batch GD (batch=full)': hist_batch_full},
        title=f"Effet du batch_size sur la val_loss ({EDGE_EPOCHS} epochs)",
        filename=os.path.join(FIGURES_DIR, "phase2_batch_size_compare.png"),
    )

    print(
        "\nObservation : SGD pur converge en bien moins d'epochs car il fait\n"
        "~13 000 mises à jour de poids par epoch, alors que Batch GD complet n'en\n"
        "fait qu'une seule. SGD est bruité, Batch GD est lisse mais très lent."
    )

    print("\n" + "=" * 60)
    print("ADVERSARIAL — Entraînement SANS normalisation")
    print("=" * 60)
    # Sans normalisation, Latitude/Longitude (~37, -120) écrasent AveRooms (~5).
    # Les gradients explosent au début.
    model_raw = build_regression_model(input_dim=8)
    hist_raw = model_raw.fit(
        X_train, y_train,
        epochs=20, batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0,
    )
    print(f"\n  Loss initiale (epoch 1) : {hist_raw.history['loss'][0]:.2f}")
    print(f"  Loss finale   (epoch 20): {hist_raw.history['loss'][-1]:.2f}")

    plot_history(
        hist_raw,
        title="Entraînement SANS normalisation (loss qui explose)",
        filename=os.path.join(FIGURES_DIR, "phase2_no_normalization.png"),
    )

    print(
        "\nObservation : la loss démarre très haute (parfois >1e10) à cause des features\n"
        "à grande échelle. Le modèle peut éventuellement descendre, mais beaucoup plus\n"
        "lentement que sur données normalisées."
    )

    print("\n" + "=" * 60)
    print("ADVERSARIAL — Adam vs SGD(lr=0.001) sur données normalisées")
    print("=" * 60)
    COMPARE_EPOCHS = 30

    print(f"\n--- Adam ({COMPARE_EPOCHS} epochs) ---")
    model_adam = build_regression_model(input_dim=8)  # Adam par défaut
    hist_adam = model_adam.fit(
        X_train_norm, y_train,
        epochs=COMPARE_EPOCHS, batch_size=32,
        validation_data=(X_val_norm, y_val),
        verbose=0,
    )

    print(f"--- SGD(lr=0.001) ({COMPARE_EPOCHS} epochs) ---")
    model_sgd = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(8,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    model_sgd.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.001),
        loss='mse',
        metrics=['mae'],
    )
    hist_sgd = model_sgd.fit(
        X_train_norm, y_train,
        epochs=COMPARE_EPOCHS, batch_size=32,
        validation_data=(X_val_norm, y_val),
        verbose=0,
    )

    print(f"\n  Adam val_loss après {COMPARE_EPOCHS} epochs : "
          f"{hist_adam.history['val_loss'][-1]:.4f}")
    print(f"  SGD  val_loss après {COMPARE_EPOCHS} epochs : "
          f"{hist_sgd.history['val_loss'][-1]:.4f}")

    plot_compare_loss(
        {'Adam (default lr)': hist_adam,
         'SGD (lr=0.001)': hist_sgd},
        title=f"Adam vs SGD sur données normalisées ({COMPARE_EPOCHS} epochs)",
        filename=os.path.join(FIGURES_DIR, "phase2_adam_vs_sgd.png"),
    )

    print(
        "\nObservation : Adam converge plus vite grâce à son learning rate adaptatif\n"
        "(estimations des moments 1 et 2 des gradients). SGD avec lr fixe = 0.001\n"
        "descend, mais plus lentement à nombre d'epochs égal."
    )