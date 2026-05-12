"""
Décision de pipeline : SPLIT AVANT FIT du scaler.

Raison : si on fitte le StandardScaler sur X entier avant le split,
les statistiques (mean, std) du test set "fuient" dans le scaler,
ce qui fausse l'évaluation finale (data leakage). Le bon réflexe :
    1) split train/val/test
    2) fit scaler sur X_train UNIQUEMENT
    3) transform val et test avec ce scaler déjà fitté

Vérification numérique de cette décision en fin de script (bloc edge case).
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----- Chargement du dataset -----
housing = fetch_california_housing()
X, y = housing.data, housing.target


# ----- Splits -----
# Premier split : train_full (80%) / test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split : train (80% du train_full) / val (20% du train_full)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)


# ----- Scaler fitté sur X_train uniquement -----
scaler = StandardScaler()
scaler.fit(X_train)

X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


# ----- Happy path : shapes, stats, feature_names -----
print("=" * 60)
print("HAPPY PATH — Vérifications du pipeline")
print("=" * 60)

print(f"\nX_train shape : {X_train_norm.shape}")
print(f"X_val   shape : {X_val_norm.shape}")
print(f"X_test  shape : {X_test_norm.shape}")

print(f"\nX_train_norm mean (par feature) :")
print(np.round(X_train_norm.mean(axis=0), 6))
print(f"X_train_norm std  (par feature) :")
print(np.round(X_train_norm.std(axis=0), 6))

feature_names = housing.feature_names
print(f"\nFeature names ({len(feature_names)}) : {feature_names}")
assert len(feature_names) == 8, "Le dataset doit contenir 8 features."
print("OK : 8 features bien présentes.")


# ----- Edge case : démonstration du data leakage -----
print("\n" + "=" * 60)
print("EDGE CASE — Démonstration du data leakage")
print("=" * 60)

# Version incorrecte : fit du scaler sur X entier AVANT le split.
scaler_leak = StandardScaler()
scaler_leak.fit(X)  # leakage : le scaler "voit" le test set
X_test_norm_leak = scaler_leak.transform(X_test)

print("\nMoyenne de X_test_norm par feature :")
print(f"  Version correcte (fit sur X_train) : "
      f"{np.round(X_test_norm.mean(axis=0), 4)}")
print(f"  Version leakage  (fit sur X entier): "
      f"{np.round(X_test_norm_leak.mean(axis=0), 4)}")

# Différence moyenne en valeur absolue
diff = np.abs(X_test_norm.mean(axis=0) - X_test_norm_leak.mean(axis=0))
print(f"\nÉcart absolu moyen entre les deux : {diff.mean():.4f}")
print(
    "Interprétation : la version leakage a une moyenne plus proche de 0 sur le test,\n"
    "parce que le scaler connaît déjà les stats globales. Sans leakage, le test\n"
    "garde un léger biais (signe sain : le test set n'a PAS été vu par le scaler)."
)


# ----- Adversarial : comportement face aux outliers -----
print("\n" + "=" * 60)
print("ADVERSARIAL — Comportement du scaler face à un outlier")
print("=" * 60)

X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])
X_extreme_norm = scaler.transform(X_extreme)

print(f"\nValeurs normalisées (X_extreme) : {np.round(X_extreme_norm[0], 2)}")
print(f"MedInc normalisé : {X_extreme_norm[0, 0]:.2f}")
print(
    "\nInterprétation : MedInc = 99999 donne une valeur normalisée à plusieurs\n"
    "dizaines de milliers d'écarts-types de la distribution d'entraînement.\n"
    "En production, le modèle extrapolerait dans une zone JAMAIS vue à l'entraînement :\n"
    "les prédictions sur ce genre d'input ne sont absolument pas fiables. Réflexe :\n"
    "détecter les outliers en amont (clipping, validation des inputs) avant l'inférence."
)