# Run from the repository root using R with jsonlite, ggplot2 and ragg.
suppressPackageStartupMessages(library(ggplot2))
con <- gzfile("analysis/v7/starvation-ledger.jsonl.gz", "rt")
x <- jsonlite::stream_in(con, verbose=FALSE)
close(con)
series <- do.call(rbind, lapply(c("QQ","Master"), function(s) {
  d <- x[[s]]
  do.call(rbind,lapply(c("cash","cargo","trees"), function(m)
    data.frame(step=x$t, player=s, measure=m, value=d[[m]])))
}))
series$measure <- factor(series$measure,c("cash","cargo","trees"),c("账户资金","工人在途碳","持有树木"))
picked <- subset(series,step==164)
p <- ggplot(series,aes(step,value,colour=player)) +
  annotate("rect",xmin=48,xmax=251,ymin=-Inf,ymax=Inf,fill="#FAEBCB",alpha=.6) +
  geom_vline(xintercept=164,colour="#776749",linetype=3,linewidth=.5) +
  geom_line(linewidth=.8) + geom_point(data=picked,size=2.2) +
  facet_grid(measure~.,scales="free_y") +
  scale_colour_manual(NULL,values=c(QQ="#21608C",Master="#BA4E37"),breaks=c("QQ","Master"),
                      labels=c("yan-QQ-v6","Master7sE")) +
  scale_x_continuous(breaks=c(0,47,90,164,252,299),expand=expansion(mult=c(.008,.012))) +
  labs(title="第 164 步，v6 的碳还在工人身上",
       subtitle="阴影区内连续 204 步没有己方工人交货。虚线是截图所在步数。",
       x="对局步数",y=NULL) +
  theme_minimal(base_size=13,base_family="PingFang SC") +
  theme(panel.grid.minor=element_blank(),panel.grid.major.x=element_blank(),
        panel.grid.major.y=element_line(colour="#D8DAD7",linewidth=.3),
        strip.text=element_text(size=11,colour="#363C3C"),plot.title=element_text(size=20,face="bold"),
        plot.subtitle=element_text(size=12,colour="#545B5F"),legend.position="top",
        legend.justification="left",legend.box.margin=margin(-3,0,-3,0),
        plot.background=element_rect(fill="white",colour=NA),plot.margin=margin(14,18,6,12))
ragg::agg_png("analysis/v7/starvation.png",width=1200,height=740,res=135)
print(p)
dev.off()
