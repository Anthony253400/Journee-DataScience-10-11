import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

racine = Path(__file__).resolve().parents[2]
df = pd.read_csv(racine / "data" / "data_joint.csv")

plt.boxplot(df["Mat_org"].dropna())

plt.ylabel("Matière organique (unité du fichier)")
plt.title("Dispersion de la matière organique des sols")
plt.xticks([1], ["Matière organique"])

plt.savefig(Path(__file__).with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.show()
