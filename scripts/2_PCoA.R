library("labdsv")
library("coin")
library("vegan")
library("yaml")
library("ggpubr")
library("cowplot")
library("tidyverse")

#import data
metadata <- read.csv(file = "./metadata.csv",header =  TRUE,
                     stringsAsFactors = FALSE, check.names = FALSE, row.names = 1)
data <- read.csv(file = "./data.csv",header =  TRUE,  
                    row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

#calculate
dist = vegdist(data, method = 'bray', correction = 'lingoes', na.rm = TRUE)
pco.results = pco(dist, k=2)

axis.1.title <- paste('PCoA1 ', 
                      round((pco.results$eig[1]/sum(pco.results$eig))*100,1),
                      '%', sep='')
axis.2.title <- paste('PCoA2 ', 
                      round((pco.results$eig[2]/sum(pco.results$eig))*100,1),
                      '%', sep='')

df.plot <- tibble(Axis1 = -1*pco.results$points[,1],
                  Axis2 = pco.results$points[,2],
                  Sample_ID = rownames(pco.results$points),
                  Group=metadata$Group,
                  Study=metadata$Study )

#plot
plot<-ggplot(df.plot,aes(x=Axis1,y=Axis2,colour=Group))+
        geom_point(alpha =.7, size=4)+
        scale_size_area()+scale_colour_brewer(type = "div", palette = "Dark2")+
        xlab(axis.1.title)+
        ylab(axis.2.title)+
        stat_ellipse(level=0.95,linetype=2,type="norm")+
        theme_classic()+geom_vline(xintercept = 0, color = 'black', size = 0.4, linetype = 4)+
        geom_hline(yintercept = 0, color = 'black', size = 0.4, linetype = 4)+
        theme(panel.grid = element_line(color = 'gray', linetype = 2, size = 0.1), 
              panel.background = element_rect(color = 'black', fill = 'transparent'),
                legend.title=element_blank(),aspect.ratio=1)

