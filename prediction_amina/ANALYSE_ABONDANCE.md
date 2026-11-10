# Régression linéaire de l'abondance A

**Question.** Les propriétés du sol, la ville et l'usage du sol permettent-ils de prédire combien de collemboles sont comptés sur un site en trois prélèvements ?

## 1. Exécuter dans VS Code

Ouvre la racine de `Journee-DataScience-10-11`. Si l'environnement Python du projet existe déjà, lance uniquement la dernière commande.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r prediction_amina\requirements.txt
.\.venv\Scripts\python.exe prediction_amina\predire_a.py
```

Les fichiers d'analyse sont produits dans `prediction_amina/resultats/abondance/`.

## 2. Expliquer le tableau X et la cible y

**Une ligne correspond à un site.** Le script reprend la logique du groupe en Python. Il retire le suffixe de répétition de l'identifiant du prélèvement, calcule l'abondance à partir des 98 colonnes de taxons, additionne les prélèvements du même site et joint les caractéristiques par identifiant. Par exemple, trois prélèvements de 5, 12 et 0 individus donnent **A = 17**. Les acariens, les autres microarthropodes, l'index du CSV et les colonnes de totaux sont exclus de la somme. Le total fourni sert à vérifier le calcul.

- **y** est `Abondance_A`, le nombre total d'individus de collemboles sur le site. **X** contient 23 mesures du sol et deux variables qualitatives, `city` et `land_use`. Le script code chacune des deux variables qualitatives avec une modalité de référence. La sélection ajoute ou retire toutes les colonnes d'une variable qualitative ensemble. Les mesures sont centrées et réduites sur l'entraînement. Un coefficient numérique s'interprète pour une hausse d'un écart-type de la mesure.
- Parmi les 140 sites échantillonnés, un seul n'a qu'un prélèvement. Sa somme correspond à un effort d'échantillonnage plus faible. Deux autres n'ont aucune mesure du sol. Ces trois sites restent dans le tableau de contrôle et les **137 sites restants** servent à comparer les modèles sur des lignes identiques. La conclusion porte sur les sites comparables avec trois prélèvements et des mesures du sol disponibles.
- Parmi les variables redondantes, `site_group` regroupe les modalités de `land_use`. Le limon complète les proportions d'argile et de sable. La matière organique est conservée et le carbone organique ainsi que le rapport C/N sont laissés hors du modèle pour limiter la redondance. Le tableau joint complet conserve ces colonnes. La matrice X exclut aussi l'identifiant, les comptages, l'effort de prélèvement et les autres indices de diversité.

**À commenter.** Dans `exploration.png`, l'abondance présente une longue queue à droite. La médiane d'entraînement est de 29, la moyenne de 69 et le maximum de 1 250. Compare les boîtes par ville et par usage du sol, puis décris le nuage du pH. Une différence descriptive entre groupes peut refléter plusieurs caractéristiques simultanées.

## 3. Présenter Mallows et Akaike

Les 109 sites d'entraînement servent à ajuster les modèles et à choisir les variables. Les 28 autres sites sont mis de côté pour le test final.

- **SCR** est la somme des carrés des résidus, c'est-à-dire des différences entre A observée et A ajustée.
- **Mallows** calcule `Cp = SCR / s²_complet − n + 2k`. Ici, `k` compte les coefficients de régression, constante comprise, et `s²_complet = SCR_complet / (n − k_complet)`. Le modèle complet a donc `Cp = k_complet`. On recherche un Cp faible, idéalement proche de `k`.
- **Akaike** calcule `AIC = −2 log(vraisemblance) + 2(k + 1)`. Le paramètre supplémentaire est la variance des erreurs. Dans le cadre gaussien, la log-vraisemblance utilise `SCR / n`. Parmi les modèles comparés sur les mêmes données et la même réponse, l'AIC le plus faible est préféré.

Le script part du modèle complet et retire un terme à la fois, avec un chemin guidé par Cp et un autre guidé par AIC. Les deux critères sont calculés pour tous les sous-modèles évalués sur ces chemins. La recherche est pas à pas. Les minima obtenus concernent les modèles visités.

| Modèle sur les 109 sites | Variables | k | Cp | AIC |
| --- | --- | ---: | ---: | ---: |
| Retenu par Cp et AIC | K2O, MgO, Cu, Pb | 5 | 4,42 | 1 384,84 |
| Complet | 23 mesures + ville et usage du sol | 32 | 32,00 | 1 406,69 |

**À commenter.** Les deux critères préfèrent la formule à quatre mesures sur l'entraînement. Cp est inférieur de 0,58 à k, et l'AIC est inférieur de 21,85 à celui du modèle complet. Le fait que les deux critères donnent la même formule ici est compatible avec des sélections différentes dans d'autres échantillons. Les sous-ensembles changent entre les plis dans `selections_par_pli.csv`.

## 4. Montrer les diagnostics

Dans `diagnostics_lineaire_AIC.png`, la dispersion des résidus augmente par endroits avec les valeurs ajustées. Le Q-Q plot s'écarte nettement de la diagonale dans la queue droite. La variance constante et la normalité des erreurs paraissent donc peu plausibles. Le modèle réduit est de plein rang (5 sur 5) et ses VIF vont de 2,03 à 3,66. Il produit aussi dix abondances ajustées négatives, alors que le comptage est positif ou nul.

Deux sites ont une distance de Cook supérieure à 1. Le premier compte 1 250 individus et présente un très grand résidu. Le second a un levier élevé, ce qui indique des valeurs explicatives éloignées du reste des sites. Dans `sensibilite_lineaire_AIC.csv`, retirer ces deux sites pour le seul diagnostic fait passer le coefficient standardisé de MgO de −70,98 à +4,68. Ils restent dans l'analyse principale. Présente donc les coefficients comme des associations fragiles.

## 5. Choisir la prédiction

La validation croisée découpe l'entraînement en cinq plis. Dans chaque pli, le script réapprend le codage de X et refait la sélection de variables sur la partie d'ajustement. La **MAE** mesure l'écart absolu moyen en nombre d'individus. La **RMSE** donne plus de poids aux gros écarts et **R²** compare l'erreur quadratique à celle d'une prédiction constante égale à la moyenne des observations évaluées.

| Méthode | MAE en validation croisée | RMSE | R² |
| --- | ---: | ---: | ---: |
| Médiane de l'entraînement | **54,44** | **152,16** | **−0,08** |
| Forêt aléatoire | 75,23 | 152,98 | −0,09 |
| Linéaire sélectionné par Cp | 100,66 | 190,79 | −0,69 |
| Linéaire sélectionné par AIC | 113,92 | 200,58 | −0,87 |
| Linéaire complet | 127,00 | 199,68 | −0,86 |

La médiane a la MAE la plus faible en validation croisée. Le code retient donc cette référence et l'évalue sur les 28 sites test. Elle prédit **29 individus** par site et obtient une MAE de **36,61**, une RMSE de **56,44** et un R² de **−0,12**. Le maximum est de 250 individus sur le test contre 1 250 à l'entraînement, ce qui aide à comprendre pourquoi la MAE est différente. La médiane est une référence de comparaison, avec une prédiction identique pour tous les sites. Cette expérience ne montre pas de gain prédictif des modèles qui utilisent les mesures du site.

### Formulation pour l'oral

> J'ai comparé des régressions linéaires pour l'abondance par site. Mallows et Akaike ont retenu quatre mesures sur les données d'entraînement. Les diagnostics montrent toutefois des résidus asymétriques, deux sites influents et des abondances ajustées négatives. En validation croisée, la médiane bat la régression et la forêt. Je retiens donc la formule à quatre mesures pour discuter la sélection linéaire et la médiane comme référence de prédiction sur ces données.

## Fichiers à ouvrir pour préparer la présentation

| Fichier | Usage |
| --- | --- |
| `sites_et_abondance.csv`, `matrice_x_entrainement.csv` | Vérifier une ligne par site, le calcul de y et le codage de X. |
| `exploration.png` | Décrire la distribution de A et les différences entre groupes. |
| `criteres_modeles.csv`, `candidats_cp_aic.csv` | Justifier le modèle linéaire retenu par les critères. |
| `diagnostics_lineaire_AIC.png`, `influence_lineaire_AIC.csv`, `sensibilite_lineaire_AIC.csv` | Montrer les limites de la régression. |
| `selections_par_pli.csv`, `validation_croisee.csv`, `score_test.csv` | Comparer les méthodes et justifier la décision de prédiction. |
