library(dplyr)
library(tidyr)
library(readr)

# 1. Charger les deux fichiers CSV
# Remplacez les noms de fichiers par vos vrais chemins
df_echantillons <- read.csv2("data/collembola_data_year1_train.csv", header= TRUE)
df_sites <- read.csv2("data/BISES_data_sites.csv",  header= TRUE)

# 2. Créer une colonne clé "Site" dans la table des échantillons
# Elle retire le suffixe "_1", "_2", "_3" de l'identifiant "Echantillon"
df_echantillons <- df_echantillons %>%
  mutate(
    Rep = sub(".*_([0-9]+)$", "data_\\1", Echantillon),       # identifie data_1, data_2, data_3
    Site = sub("_[0-9]+$", "", Echantillon)                   # retire le numéro final pour matcher le Site
  )

# 3. Pivoter les répétitions (1, 2, 3) sur une seule et même ligne par Site
# Toutes les colonnes de mesures sont réparties en suffixe (_data_1, _data_2, _data_3)
df_ech_wide <- df_echantillons %>%
  select(-Echantillon) %>%
  pivot_wider(
    names_from = Rep,
    values_from = -c(Site, Rep),
    names_sep = "_"
  )

# 4. Fusionner avec la table des sites (les données du site + les 3 réplicats sur une seule ligne)
df_final <- df_sites %>%
  left_join(df_ech_wide, by = "Site")

# 5. Exporter dans un nouveau fichier CSV
write_csv(df_final, "BISES_donnees_combinees.csv")