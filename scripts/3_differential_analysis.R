library(MMUPHin)
library(magrittr)
library(dplyr)
library(ggplot2)
library(vegan)
library(Maaslin2)

#import data
metadata <- read.csv(file = "./metadata.csv",header =  TRUE,
                     stringsAsFactors = FALSE, check.names = FALSE, row.names = 1)
data <- read.csv(file = "./data.csv",header =  TRUE,  
                    row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

#differential analysis via meta-analysis
fit_lm_meta <- lm_meta(feature_abd = data,
                       batch = "Study",
                       exposure = "group",
                       covariates = c("Gender", "Age", "BMI"),
                       data = metadata,
                       control = list(verbose = FALSE))

result = fit_lm_meta$meta_fits