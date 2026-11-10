"""Calcule les indices par prélèvement et évalue la prédiction de S."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOSSIER = Path(__file__).resolve().parent
FICHIER_JOINT = DOSSIER.parent / "data" / "data_joint.csv"
SORTIES = DOSSIER / "resultats"
REPETITIONS = (1, 2, 3)
GRAINE = 42

# Liste autorisée des mesures et catégories disponibles pour un nouveau site.
QUANTITATIVES = [
    "Argile", "Limon", "Sable", "Conduct", "pH.eau", "Carbonates", "Calcaire",
    "C..orga", "Mat_org", "Azote", "C.N", "P2O5", "CEC", "K2O", "MgO", "CaO",
    "Na2O", "Al", "Cd", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Pb", "Zn",
]
QUALITATIVES = ["city", "land_use", "site_group"]
VARIABLES = QUANTITATIVES + QUALITATIVES
EXCLUSIONS = {
    "X", "total_collembola", "Acari_Gamasides", "Acari_Oribates",
    "Acari_others", "Acari_total", "Other_microarthropodes",
    "total_microarthropodes",
}


def preparer_prelevements(sites):
    """Remet les répétitions de la jointure existante sur des lignes séparées."""
    if sites["Site"].isna().any() or sites["Site"].duplicated().any():
        raise ValueError("Chaque ligne de data_joint.csv doit avoir un Site unique.")

    taxa = [
        colonne.removesuffix("_data_1")
        for colonne in sites.columns
        if colonne.endswith("_data_1")
        and colonne.removesuffix("_data_1") not in EXCLUSIONS
    ]
    if not taxa:
        raise ValueError("Aucune colonne de comptage des collemboles trouvée.")

    blocs = []
    for rep in REPETITIONS:
        suffixe = f"_data_{rep}"
        index_source = f"X{suffixe}"
        total_source = f"total_collembola{suffixe}"
        colonnes_taxa = [f"{taxon}{suffixe}" for taxon in taxa]
        present = sites[index_source].notna()
        if sites.loc[~present, colonnes_taxa + [total_source]].notna().any().any():
            raise ValueError(f"Comptages sans identifiant pour la répétition {rep}.")

        bloc = sites.loc[present, ["Site", *VARIABLES, index_source, total_source,
                                   *colonnes_taxa]].copy()
        bloc = bloc.rename(columns={
            colonne: colonne.removesuffix(suffixe)
            for colonne in [index_source, total_source, *colonnes_taxa]
        })
        bloc.insert(1, "Echantillon", bloc["Site"] + f"_{rep}")
        blocs.append(bloc)

    prelevements = pd.concat(blocs, ignore_index=True)
    if prelevements.empty:
        raise ValueError("Aucun prélèvement exploitable dans data_joint.csv.")
    if prelevements["Echantillon"].duplicated().any() or prelevements["X"].duplicated().any():
        raise ValueError("Le fichier joint contient des identifiants de prélèvement en double.")

    comptes = prelevements[taxa].to_numpy(dtype=float)
    if not np.isfinite(comptes).all():
        raise ValueError("Au moins un comptage de collemboles est manquant ou non fini.")
    if (comptes < 0).any() or (comptes % 1 != 0).any():
        raise ValueError("Les comptages de collemboles doivent être des entiers positifs ou nuls.")

    abondance = comptes.sum(axis=1).astype(int)
    richesse = (comptes > 0).sum(axis=1)
    if not np.array_equal(abondance, prelevements["total_collembola"].to_numpy()):
        raise ValueError("La somme des comptages diffère de la colonne total_collembola.")

    # Les taxons absents contribuent 0 à Shannon. Quand A vaut 0, H vaut 0.
    parts = np.divide(comptes, abondance[:, None], out=np.zeros_like(comptes),
                      where=abondance[:, None] > 0)
    log_parts = np.log(parts, out=np.zeros_like(parts), where=parts > 0)
    shannon = -(parts * log_parts).sum(axis=1)
    # H / ln(S) est indéfini pour S = 0 ou S = 1.
    pielou = np.full(len(richesse), np.nan)
    plusieurs_taxa = richesse > 1
    pielou[plusieurs_taxa] = shannon[plusieurs_taxa] / np.log(richesse[plusieurs_taxa])

    print(f"Taxons retenus : {len(taxa)}")
    return prelevements[["Echantillon", "Site", *VARIABLES]].assign(
        A=abondance, S=richesse, H=shannon, J=pielou
    )


def evaluer_modeles(prelevements):
    """Compare les modèles sur les mêmes sites mis de côté."""
    x = prelevements[VARIABLES]
    y = prelevements["S"]
    groupes = prelevements["Site"]
    separation = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=GRAINE)
    entrainement, test = next(separation.split(x, y, groups=groupes))
    for nom, indices in [("Entraînement", entrainement), ("Test", test)]:
        print(f"{nom} : {len(indices)} prélèvements, {groupes.iloc[indices].nunique()} sites")

    preparation = ColumnTransformer([
        ("mesures", Pipeline([
            ("valeurs_manquantes", SimpleImputer(strategy="median")),
            ("echelle", StandardScaler()),
        ]), QUANTITATIVES),
        ("categories", Pipeline([
            ("valeurs_manquantes", SimpleImputer(strategy="most_frequent")),
            ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), QUALITATIVES),
    ])
    modeles = {
        "reference_moyenne": DummyRegressor(strategy="mean"),
        "regression_lineaire": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=GRAINE
        ),
    }

    scores = []
    mesures_test = y.iloc[test]
    predictions = prelevements.iloc[test][["Echantillon", "Site", "S"]].copy()
    for nom, modele in modeles.items():
        pipeline = Pipeline([("preparation", clone(preparation)), ("modele", modele)])
        pipeline.fit(x.iloc[entrainement], y.iloc[entrainement])
        estime = pipeline.predict(x.iloc[test])
        predictions[nom] = estime
        scores.append({
            "modele": nom,
            "MAE": mean_absolute_error(mesures_test, estime),
            "RMSE": root_mean_squared_error(mesures_test, estime),
            "R2": r2_score(mesures_test, estime),
        })

    return pd.DataFrame(scores), predictions


def main():
    if not FICHIER_JOINT.is_file():
        raise SystemExit(f"Fichier de données absent : {FICHIER_JOINT}")

    sites = pd.read_csv(FICHIER_JOINT, dtype={"Site": "string"})
    prelevements = preparer_prelevements(sites)
    print(f"Sites dans la jointure : {len(sites)}")
    print(f"Sites avec prélèvement : {prelevements['Site'].nunique()}")
    print(f"Prélèvements utilisés : {len(prelevements)}")
    manquantes = prelevements[VARIABLES].isna().sum()
    print("Valeurs explicatives manquantes par variable, parmi les prélèvements utilisés :")
    print(manquantes[manquantes > 0].to_string() if manquantes.any() else "Aucune.")
    print(f"J indéfini : {prelevements['J'].isna().sum()} prélèvements")

    scores, predictions = evaluer_modeles(prelevements)
    print(scores.to_string(index=False, float_format=lambda valeur: f"{valeur:.3f}"))
    SORTIES.mkdir(exist_ok=True)
    prelevements[["Echantillon", "Site", "A", "S", "H", "J"]].to_csv(
        SORTIES / "indices.csv", index=False
    )
    scores.to_csv(SORTIES / "scores.csv", index=False)
    predictions.to_csv(SORTIES / "predictions_test.csv", index=False)

    observe = predictions["S"]
    estime = predictions["regression_lineaire"]
    bornes = [min(observe.min(), estime.min()), max(observe.max(), estime.max())]
    fig, axes = plt.subplots(figsize=(6, 5), layout="constrained")
    axes.scatter(observe, estime, alpha=0.6, label="Prélèvements test")
    axes.plot(bornes, bornes, "--", color="gray", label="Prédiction parfaite")
    axes.set(xlabel="Richesse S observée", ylabel="Richesse S prédite",
             title="Régression linéaire sur les sites test")
    axes.legend()
    fig.savefig(SORTIES / "richesse_observee_predite.png", dpi=150)
    plt.close(fig)
    print(f"Résultats enregistrés dans {SORTIES}")


if __name__ == "__main__":
    main()
