### GenSmoothScatter.R
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("marray")

library(KernSmooth);
library(marray);
load("exampleData.Rd");

ageT.v <- (age.v- min(age.v))/(max(age.v)-min(age.v));
d.o <- bkde2D(x=cbind(ageT.v,beta.v),bandwidth=c(0.005,0.005),gridsize=c(500,500),range.x=list(x=c(-0.1,1.1),y=c(-0.1,1.1)));
z.m <- (d.o$fhat-min(d.o$fhat))/(max(d.o$fhat)-min(d.o$fhat));
breaks.v <- c(-0.1,0.00001,0.01,seq(0.1,1.1,0.1));
color.v <- c("white",maPalette(low="green",high="darkgreen",k=length(breaks.v)-2));

pdf("ExampleSmoothScatter.pdf",width=4,height=3);
par(mar=c(1,1,1,1));
image(x=d.o$x1,y=d.o$x2,z=z.m,axes=FALSE,col=color.v,breaks=breaks.v);
dev.off();
