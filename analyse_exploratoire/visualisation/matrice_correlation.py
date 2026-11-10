import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

racine = Path(__file__).resolve().parents[2]
df = pd.read_csv(racine / "data" / "data_joint.csv")

# Caractéristiques physico-chimiques des sols uniquement
colonnes = [
    "Argile", "Limon", "Sable", "Conduct", "pH.eau", "Carbonates",
    "Calcaire", "C..orga", "Mat_org", "Azote", "C.N", "P2O5",
    "CEC", "K2O", "MgO", "CaO", "Na2O", "Al", "Cd", "Co",
    "Cr", "Cu", "Fe", "Mn", "Ni", "Pb", "Zn"
]
correlations = df[colonnes].corr()

plt.figure(figsize=(12, 10))
plt.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Corrélation de Pearson")
plt.xticks(range(len(colonnes)), colonnes, rotation=90, fontsize=8)
plt.yticks(range(len(colonnes)), colonnes, fontsize=8)
plt.title("Corrélations entre caractéristiques des sols")
plt.tight_layout()

plt.savefig(Path(__file__).with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.show()
