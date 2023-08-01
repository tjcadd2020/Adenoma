library(pheatmap)
library(vegan)
library(dplyr)
set.seed(0)

#import data
metadata <- read.csv(file = "./metadata.csv",header =  TRUE,
                     stringsAsFactors = FALSE, check.names = FALSE, row.names = 1)
data <- read.csv(file = "./data.csv",header =  TRUE,  
                    row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

#calculate distance matrix
dist = vegdist(data, method = 'bray', correction = 'lingoes', na.rm = TRUE)

permanova = adonis(dist~Age, data = metadata, permutations=999, method = "bray")

#plot
all<-read.csv(file='./adonis_result.csv',row.names=1,header=TRUE)
bk <- c(seq(0,10,by=0.1),seq(11,35,by=1))

p<-pheatmap(all, display_numbers = TRUE, number_format = "%.2f", cellwidth=24,cluster_col = FALSE, cluster_row = FALSE, 
         cellheight=24,fontsize_row=24,fontsize_col=24, 
         color = c(colorRampPalette(colors = c("white", "deepskyblue"))(length(bk)/2),
                   colorRampPalette(colors = c("deepskyblue", "darkorchid4"))(length(bk)/2)), breaks=bk)