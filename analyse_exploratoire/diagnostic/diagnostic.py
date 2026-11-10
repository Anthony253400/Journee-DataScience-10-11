import pandas as pd
from pathlib import Path

# Chargement des données
racine = Path(__file__).resolve().parents[2]
df = pd.read_csv(racine / "data" / "data_joint.csv")

# Dimensions
observations, variables = df.shape

# Types des variables
types = df.dtypes.value_counts()

# Valeurs manquantes
manquantes = df.isna().sum().sum()
pourcentage = df.isna().mean().mean() * 100
manquantes_par_variable = df.isna().sum()

# Doublons
doublons = df.duplicated().sum()

# Statistiques descriptives
statistiques = df.describe()

print("Observations :", observations)
print("Variables :", variables)
print("Types :", types.to_dict())
print("Valeurs manquantes :", manquantes)
print("Pourcentage manquant :", round(pourcentage, 2))
print("Doublons :", doublons)
print("Statistiques :", statistiques.to_dict())
