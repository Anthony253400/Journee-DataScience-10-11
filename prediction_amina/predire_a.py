"""Sélection et validation de régressions pour l'abondance par site."""

from functools import cache
from pathlib import Path
from statistics import NormalDist

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOSSIER = Path(__file__).resolve().parent
DATA = DOSSIER.parent / "data"
SORTIES = DOSSIER / "resultats" / "abondance"
GRAINE = 42
CIBLE = "Abondance_A"
QUANTITATIVES = [
    "Argile", "Sable", "Conduct", "pH eau", "Carbonates", "Calcaire", "Mat_org",
    "Azote", "P2O5", "CEC", "K2O", "MgO", "CaO", "Na2O", "Al", "Cd", "Co",
    "Cr", "Cu", "Fe", "Mn", "Ni", "Pb", "Zn",
]
QUALITATIVES = ["city", "land_use"]
VARIABLES = QUANTITATIVES + QUALITATIVES
EXCLUSIONS = {
    "Echantillon", "total_collembola", "Acari_Gamasides", "Acari_Oribates",
    "Acari_others", "Acari_total", "Other_microarthropodes", "total_microarthropodes",
}


def preparer_sites():
    """Reproduit la logique du groupe : sommer les taxons, puis joindre par site."""
    captures = pd.read_csv(DATA / "collembola_data_year1_train.csv", sep=";", decimal=",")
    sites = pd.read_csv(DATA / "BISES_data_sites.csv", sep=";", decimal=",")
    identifiants = captures["Echantillon"]
    if identifiants.isna().any() or identifiants.duplicated().any():
        raise ValueError("Les identifiants des prélèvements doivent être présents et uniques.")
    repetitions = identifiants.str.extract(r"^(.+)_([123])$")
    if repetitions.isna().any().any():
        raise ValueError("Un identifiant de prélèvement ne finit pas par _1, _2 ou _3.")
    if sites["Site"].isna().any() or sites["Site"].duplicated().any():
        raise ValueError("Les identifiants de site doivent être présents et uniques.")

    taxa = [
        nom for nom in captures.columns
        if nom not in EXCLUSIONS and not nom.startswith("Unnamed:")
    ]
    comptes = captures[taxa].to_numpy(dtype=float)
    if not taxa or not np.isfinite(comptes).all():
        raise ValueError("Les comptages des taxons doivent être présents et finis.")
    if (comptes < 0).any() or (comptes % 1 != 0).any():
        raise ValueError("Les comptages des taxons doivent être entiers et positifs ou nuls.")
    abondances = comptes.sum(axis=1).astype(int)
    if not np.array_equal(abondances, captures["total_collembola"].to_numpy()):
        raise ValueError("La somme des taxons diffère du total de collemboles fourni.")

    par_prelevement = pd.DataFrame({"Site": repetitions[0], CIBLE: abondances})
    par_site = par_prelevement.groupby("Site", as_index=False).agg(
        Abondance_A=(CIBLE, "sum"), nb_prelevements=(CIBLE, "size")
    )
    joint = par_site.merge(sites, on="Site", how="left", validate="one_to_one", indicator=True)
    if not joint["_merge"].eq("both").all():
        raise ValueError("Au moins un site échantillonné est absent du fichier des sites.")
    joint = joint.drop(columns="_merge")
    joint["effort_complet"] = joint["nb_prelevements"].eq(3)
    joint["x_complet"] = joint[VARIABLES].notna().all(axis=1)
    joint["inclus"] = joint["effort_complet"] & joint["x_complet"]
    print(f"{len(captures)} prélèvements, {len(taxa)} taxons, {len(joint)} sites échantillonnés.")
    print(f"{joint['inclus'].sum()} sites inclus. Sites écartés de l'ajustement :")
    print(joint.loc[~joint["inclus"], ["Site", CIBLE, "nb_prelevements", "x_complet"]].to_string(index=False))
    return joint


def matrice_x(apprentissage, autres):
    """Apprend les échelles et modalités sur l'entraînement. Une modalité sert de référence."""
    moyenne = apprentissage[QUANTITATIVES].mean()
    ecart = apprentissage[QUANTITATIVES].std(ddof=0)
    if ecart.eq(0).any():
        raise ValueError("Au moins une mesure quantitative est constante dans l'entraînement.")
    modalites = {nom: sorted(apprentissage[nom].unique()) for nom in QUALITATIVES}

    groupes = {nom: [nom] for nom in QUANTITATIVES}
    groupes.update({
        nom: [f"{nom}={niveau}" for niveau in niveaux[1:]]
        for nom, niveaux in modalites.items()
    })
    matrices = []
    for table in (apprentissage, autres):
        x = pd.DataFrame({"constante": 1.0}, index=table.index)
        for nom in QUANTITATIVES:
            x[nom] = (table[nom] - moyenne[nom]) / ecart[nom]
        for nom, niveaux in modalites.items():
            if len(niveaux) < 2 or not table[nom].isin(niveaux).all():
                raise ValueError(f"Modalités insuffisantes ou inconnues pour {nom}.")
            for niveau, colonne in zip(niveaux[1:], groupes[nom]):
                x[colonne] = table[nom].eq(niveau).astype(float)
        matrices.append(x)
    return matrices[0], matrices[1], groupes


def selectionner_modeles(x, y, groupes):
    """Explore deux chemins arrière et note chaque sous-modèle selon les deux critères."""
    n = len(y)
    tous = tuple(groupes)

    @cache
    def ajuster(termes):
        colonnes = ["constante", *(c for terme in termes for c in groupes[terme])]
        design = x[colonnes].to_numpy()
        beta, _, rang, _ = np.linalg.lstsq(design, y, rcond=None)
        if rang != len(colonnes) or rang >= n:
            raise ValueError("Le modèle doit être de plein rang et avoir des degrés de liberté.")
        erreurs = y - design @ beta
        return colonnes, beta, float(erreurs @ erreurs)

    colonnes_full, _, scr_full = ajuster(tous)
    variance_full = scr_full / (n - len(colonnes_full))
    if variance_full <= 0:
        raise ValueError("Mallows Cp exige une variance résiduelle complète positive.")
    candidats = {}

    def noter(termes):
        colonnes, _, scr = ajuster(termes)
        k = len(colonnes)
        if scr <= 0:
            raise ValueError("La variance estimée doit être positive pour calculer l'AIC.")
        cp = scr / variance_full - n + 2 * k
        aic = n * (np.log(2 * np.pi * scr / n) + 1) + 2 * (k + 1)
        candidats[termes] = {
            "variables": " + ".join(termes) or "constante seule",
            "k_coefficients": k, "SCR": scr, "Cp": cp, "Cp_moins_k": cp - k, "AIC": aic,
        }
        return candidats[termes]

    noter(tous)
    for critere in ("Cp", "AIC"):
        termes = tous
        while termes:
            retraits = [termes[:i] + termes[i + 1:] for i in range(len(termes))]
            termes = min(retraits, key=lambda option: noter(option)[critere])

    choisis = {
        "lineaire_Cp": min(candidats, key=lambda option: candidats[option]["Cp"]),
        "lineaire_AIC": min(candidats, key=lambda option: candidats[option]["AIC"]),
        "lineaire_complet": tous,
    }
    modeles = {nom: ajuster(termes) for nom, termes in choisis.items()}
    bilan = pd.DataFrame([{"modele": nom, **candidats[termes]} for nom, termes in choisis.items()])
    chemin = pd.DataFrame(candidats.values()).sort_values("AIC")
    return modeles, bilan, chemin


def predire(apprentissage, autres):
    x, x_autres, groupes = matrice_x(apprentissage, autres)
    y = apprentissage[CIBLE].to_numpy()
    lineaires, bilan, chemin = selectionner_modeles(x, y, groupes)
    predictions = {
        nom: x_autres[colonnes].to_numpy() @ beta
        for nom, (colonnes, beta, _) in lineaires.items()
    }
    foret = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=GRAINE)
    foret.fit(x.drop(columns="constante"), y)
    predictions["foret_aleatoire"] = foret.predict(x_autres.drop(columns="constante"))
    predictions["reference_mediane"] = np.repeat(np.median(y), len(autres))
    return predictions, lineaires, bilan, chemin, x


def scores(y, predictions):
    return pd.DataFrame([
        {"modele": nom, "MAE": mean_absolute_error(y, estime),
         "RMSE": root_mean_squared_error(y, estime), "R2": r2_score(y, estime)}
        for nom, estime in predictions.items()
    ]).sort_values("MAE", ignore_index=True)


def diagnostics(apprentissage, x, modele, nom):
    colonnes, beta, scr = modele
    design = x[colonnes].to_numpy()
    ajustes = design @ beta
    residus = apprentissage[CIBLE].to_numpy() - ajustes
    n, k = design.shape
    q, _ = np.linalg.qr(design)
    levier = (q ** 2).sum(axis=1)
    standardises = residus / np.sqrt((scr / (n - k)) * (1 - levier))
    cook = standardises ** 2 * levier / (k * (1 - levier))
    vif = []
    for j, colonne in enumerate(colonnes[1:], start=1):
        autres = np.delete(design, j, axis=1)
        coef = np.linalg.lstsq(autres, design[:, j], rcond=None)[0]
        reste = design[:, j] - autres @ coef
        vif.append(np.sum((design[:, j] - design[:, j].mean()) ** 2) / (reste @ reste))
    pd.DataFrame({"colonne_X": colonnes[1:], "VIF": vif}).sort_values(
        "VIF", ascending=False
    ).to_csv(SORTIES / f"vif_{nom}.csv", index=False)
    pd.DataFrame({"coefficient": colonnes, "estimation": beta}).to_csv(
        SORTIES / f"coefficients_{nom}.csv", index=False
    )
    influents = cook > 1
    beta_sans = beta
    if influents.any():
        beta_sans, _, rang, _ = np.linalg.lstsq(
            design[~influents], apprentissage.loc[~influents, CIBLE], rcond=None
        )
        if rang != k:
            beta_sans = np.full(k, np.nan)
    pd.DataFrame({
        "coefficient": colonnes, "avec_tous_les_sites": beta,
        "sans_les_sites_Cook_sup_1": beta_sans, "nb_sites_ecartes": influents.sum(),
    }).to_csv(SORTIES / f"sensibilite_{nom}.csv", index=False)
    pd.DataFrame({
        "Site": apprentissage["Site"], CIBLE: apprentissage[CIBLE],
        "A_ajustee": ajustes, "residu": residus, "residu_standardise": standardises,
        "levier": levier, "Cook": cook,
    }).sort_values("Cook", ascending=False).to_csv(SORTIES / f"influence_{nom}.csv", index=False)
    print(f"{nom} : rang {k}/{k}, Cook max {cook.max():.3f}, "
          f"{(ajustes < 0).sum()} ajustements négatifs.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    axes[0, 0].scatter(ajustes, standardises, alpha=0.65)
    axes[0, 0].axhline(0, color="gray", linestyle="--")
    axes[0, 0].set(xlabel="A ajustée", ylabel="Résidu standardisé")
    quantiles = np.array([NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)])
    axes[0, 1].scatter(quantiles, np.sort(standardises), alpha=0.65)
    axes[0, 1].plot(quantiles, quantiles, "--", color="gray")
    axes[0, 1].set(xlabel="Quantiles de la loi normale", ylabel="Quantiles des résidus")
    axes[1, 0].stem(np.arange(1, n + 1), cook, markerfmt=" ")
    axes[1, 0].axhline(1, color="gray", linestyle="--")
    axes[1, 0].set(xlabel="Observation dans l'entraînement", ylabel="Distance de Cook")
    axes[1, 1].scatter(levier, standardises, alpha=0.65)
    axes[1, 1].axvline(2 * k / n, color="gray", linestyle="--")
    axes[1, 1].set(xlabel="Levier", ylabel="Résidu standardisé")
    fig.suptitle(f"Diagnostics sur l'entraînement — {nom}")
    fig.savefig(SORTIES / f"diagnostics_{nom}.png", dpi=150)
    plt.close(fig)


def explorer(apprentissage):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    axes[0, 0].hist(apprentissage[CIBLE], bins=18)
    axes[0, 0].set(xlabel="Abondance A par site", ylabel="Nombre de sites")
    axes[0, 1].scatter(apprentissage["pH eau"], apprentissage[CIBLE], alpha=0.65)
    axes[0, 1].set(xlabel="pH du sol", ylabel="Abondance A")
    for axe, nom in zip(axes[1], QUALITATIVES):
        groupes = list(apprentissage.groupby(nom, sort=True))
        axe.boxplot([table[CIBLE] for _, table in groupes])
        axe.set_xticks(range(1, len(groupes) + 1), [niveau for niveau, _ in groupes])
        axe.set(xlabel=nom, ylabel="Abondance A")
    fig.suptitle("Abondance sur les sites d'entraînement")
    fig.savefig(SORTIES / "exploration.png", dpi=150)
    plt.close(fig)


def main():
    joint = preparer_sites()
    analyses = joint.loc[joint["inclus"]]
    apprentissage, test = train_test_split(analyses, test_size=0.2, random_state=GRAINE)
    print(f"{len(apprentissage)} sites d'entraînement, {len(test)} sites test.")

    # Les deux sélections de variables sont répétées à l'intérieur de chaque pli.
    plis = KFold(n_splits=5, shuffle=True, random_state=GRAINE)
    predictions_cv, selections_cv = [], []
    for pli, (train, validation) in enumerate(plis.split(apprentissage), start=1):
        val = apprentissage.iloc[validation]
        predictions, _, bilan_pli, _, _ = predire(apprentissage.iloc[train], val)
        predictions_cv.append(pd.DataFrame({CIBLE: val[CIBLE], **predictions}, index=val.index))
        selections_cv.append(bilan_pli.loc[bilan_pli["modele"].ne("lineaire_complet")].assign(pli=pli))
    hors_pli = pd.concat(predictions_cv)
    comparaison = scores(hors_pli[CIBLE], hors_pli.drop(columns=CIBLE))
    retenu = comparaison.loc[0, "modele"]

    predictions, lineaires, bilan, chemin, x = predire(apprentissage, test)
    estimation_test = {retenu: predictions[retenu]}
    bilan_test = scores(test[CIBLE], estimation_test)
    print("\nCritères calculés sur l'entraînement :\n", bilan.to_string(index=False))
    print("\nValidation croisée, classée par MAE :\n", comparaison.round(3).to_string(index=False))
    print("\nTest final du modèle retenu :\n", bilan_test.round(3).to_string(index=False))

    SORTIES.mkdir(parents=True, exist_ok=True)
    joint.to_csv(SORTIES / "sites_et_abondance.csv", index=False)
    x.assign(Abondance_A=apprentissage[CIBLE], Site=apprentissage["Site"]).to_csv(
        SORTIES / "matrice_x_entrainement.csv", index=False
    )
    bilan.to_csv(SORTIES / "criteres_modeles.csv", index=False)
    chemin.to_csv(SORTIES / "candidats_cp_aic.csv", index=False)
    pd.concat(selections_cv).to_csv(SORTIES / "selections_par_pli.csv", index=False)
    comparaison.to_csv(SORTIES / "validation_croisee.csv", index=False)
    bilan_test.to_csv(SORTIES / "score_test.csv", index=False)
    test[["Site", CIBLE]].assign(A_predite=estimation_test[retenu]).to_csv(
        SORTIES / "predictions_test.csv", index=False
    )
    explorer(apprentissage)
    for nom in ("lineaire_Cp", "lineaire_AIC"):
        diagnostics(apprentissage, x, lineaires[nom], nom)


if __name__ == "__main__":
    main()
