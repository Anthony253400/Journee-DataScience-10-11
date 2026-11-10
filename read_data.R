# Les collemboles, les acariens, les Gamasides et les oribates 
# font partie des principaux microarthropodes qui composent la mésofaune du sol 
# (mesurant généralement entre 0,1 et 2 mm)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

data1 <- read.csv2("data/BISES_collembola_data_year1_clean.csv", header= TRUE)
dim(data1)
# [1] 522 106

data2 <- read.csv2("BISES_collembola_data_year2_clean.csv", header= TRUE)
dim(data2)
# [1] 531 132

sum(colnames(data1) %in% colnames(data2))
# [1] 69
hist(data1$total_collembola, breaks= 50)

hist(data2$total_collembola, breaks= 50)

library(vegan)

data1.spe <- data1[,2:99]

richesse <- specnumber(data1.spe)
hist(richesse, breaks= 50)

#Shanon
shannon_index <- diversity(data1.spe, index = "shannon")

hist(shannon_index, breaks= 50)
#Pielou
