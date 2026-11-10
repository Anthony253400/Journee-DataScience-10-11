# Prédire la richesse des collemboles

Pour le projet actuel sur l'abondance **A** par site, voir [le guide de régression Mallows et Akaike](ANALYSE_ABONDANCE.md).

La richesse **S** est le nombre de taxons de collemboles observés dans un prélèvement. Le modèle apprend à estimer ce nombre à partir des caractéristiques du site, par exemple le pH, la teneur en argile, la ville et l'usage du sol.

## Lancer le projet dans VS Code

Ouvre le dossier `Journee-DataScience-10-11` dans VS Code. Dans **Terminal → Nouveau terminal**, lance ces commandes PowerShell depuis la racine du dépôt. Elles utilisent Python 3.10 ou plus récent et installent les bibliothèques dans un environnement propre au projet.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r prediction_amina\requirements.txt
.\.venv\Scripts\python.exe prediction_amina\predire_s.py
```

Pandas lit et transforme les tableaux, NumPy calcule les indices, scikit-learn entraîne et évalue les modèles, et Matplotlib produit le graphique. Les bibliothèques de ce projet sont listées dans `prediction_amina/requirements.txt`.

## Du fichier joint aux indices

`predire_s.py` lit `data/data_joint.csv`, la jointure déjà présente dans le dépôt. Chaque ligne de ce fichier décrit un site. Les suffixes `_data_1`, `_data_2` et `_data_3` séparent les répétitions. Le script transforme ces blocs de colonnes en lignes de prélèvements et garde le code `Site` pour les regrouper. Les colonnes `X_data_*` servent à repérer les prélèvements présents. Sur les 174 sites de la jointure, 140 ont au moins un prélèvement, pour un total de 418 prélèvements.

Le script vérifie l'unicité des identifiants, l'intégrité des comptages et l'accord de leur somme avec le total fourni. Il affiche les valeurs explicatives manquantes parmi les prélèvements utilisés. Comme dans `read_data_collembola.R`, il écarte les colonnes de totaux, les acariens et les autres microarthropodes. Il écarte aussi l'index de ligne `X`.

| Indice | Calcul par prélèvement |
| --- | --- |
| **A**, abondance | Somme des comptages des taxons de collemboles. |
| **S**, richesse | Nombre de taxons dont le comptage est strictement positif. |
| **H**, Shannon | Somme des termes `-p × ln(p)` sur les taxons présents, avec `p = comptage / A`. Le script fixe H à 0 quand A vaut 0. |
| **J**, Pielou | `H / ln(S)` quand S est supérieur à 1. La cellule reste vide pour S égal à 0 ou 1. |

## Entraîner les modèles

Le script réserve 20 % des **sites** pour le test, avec une graine fixe de 42. Tous les prélèvements d'un site vont dans le même groupe. Cela permet de mesurer l'erreur sur des sites que le modèle découvre au moment du test. Les comptages et les indices de diversité servent au calcul de la cible. Les entrées du modèle se limitent aux 27 mesures du sol et à trois catégories du site (`city`, `land_use`, `site_group`).

À partir des seules données d'entraînement, le script apprend la médiane des mesures numériques et la catégorie la plus fréquente pour compléter les valeurs manquantes. Il met les mesures numériques à une échelle comparable et transforme les catégories en colonnes indicatrices. Il applique ensuite cette même préparation aux sites test.

La référence prédit toujours la richesse moyenne des prélèvements d'entraînement. La régression linéaire est le premier modèle. Ridge utilise aussi une relation linéaire et limite la taille des coefficients, ce qui peut aider quand des mesures se ressemblent. Random Forest combine des arbres et peut apprendre des relations non linéaires. Les trois modèles sont évalués sur le même test avec des réglages fixés à l'avance.

## Lire les résultats

Le script écrit quatre fichiers dans `prediction_amina/resultats/`, un dossier local ignoré par Git.

- `indices.csv` contient A, S, H et J pour chaque prélèvement.
- `scores.csv` compare les modèles. La **MAE** est l'erreur absolue moyenne en nombre de taxons. La **RMSE** donne plus de poids aux gros écarts. Pour ces deux mesures, plus bas est préférable. Pour **R²**, plus haut est préférable. Un R² négatif indique une erreur quadratique supérieure à celle d'une prédiction constante égale à la moyenne du test. La ligne `reference_moyenne` utilise, elle, la moyenne de l'entraînement.
- `predictions_test.csv` donne S observée et S prédite pour chaque prélèvement test.
- `richesse_observee_predite.png` compare S observée et S prédite par la régression linéaire. Plus un point est proche de la diagonale, plus sa prédiction est précise.

Les prélèvements d'un même site partagent les mêmes variables explicatives. Le modèle leur attribue donc la même prédiction même si leur richesse observée varie. Les régressions peuvent produire des nombres décimaux ou négatifs. Les scores utilisent les valeurs brutes. Une seule séparation donne une première estimation qui dépend des sites tirés au sort et des villes représentées dans le jeu de données.

### Premier essai sur les données du dépôt

L'entraînement utilise 334 prélèvements de 112 sites, et le test 84 prélèvements de 28 autres sites. Six prélèvements issus de deux sites ont toutes leurs mesures numériques manquantes. Le modèle les conserve et complète leurs mesures avec les médianes apprises à l'entraînement.

| Modèle | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Référence moyenne | 1,835 | 2,255 | −0,010 |
| Régression linéaire | 2,278 | 2,956 | −0,736 |
| Ridge | 2,242 | 2,873 | −0,639 |
| Random Forest | 1,928 | 2,433 | −0,176 |

Sur ce test, la référence moyenne obtient les plus petites erreurs. La régression linéaire s'écarte de la richesse observée de 2,28 taxons en moyenne, contre 1,84 pour la référence. Random Forest réduit cet écart par rapport à la régression linéaire, mais reste derrière la référence.
