import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

racine = Path(__file__).resolve().parents[2]
df = pd.read_csv(racine / "data" / "data_joint.csv")

plt.hist(df["pH.eau"].dropna(), bins=15, edgecolor="black", linewidth=0.8)

plt.xlabel("pH du sol")
plt.ylabel("Nombre de sites")
plt.title("Distribution du pH des sols")

plt.savefig(Path(__file__).with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.show()
