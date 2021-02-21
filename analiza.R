library(ggplot2)
library(dplyr)
library (tidyr)
library(ggstatsplot)
library(stringr)
library(tibble)
library(knitr)

# H1: Dyn settings

## Charts
df <- read.csv('settings.csv')
df$option <- str_extract(df$Algorithm, "\\([^()]+\\)")
df$Fold <- rep(1:10, each=21)
df <- df[complete.cases(df),]
df$option <- gsub("(", "", df$option, fixed=T)
df$option <- gsub(")", "", df$option, fixed=T)
df$option <- gsub("_", " ", df$option, fixed=T)
elements <-  strsplit(df$option, '=')
for (i in 1:length(elements)){
  col_name <- elements[[i]][1]
  col_val <- elements[[i]][2]
  if (!(col_name %in% colnames(df))){
    df[col_name] <- NA
  }
  df[i, col_name] <- col_val
}

df$`cut interval` <- gsub("[", "", df$`cut interval`, fixed=T)
df$`cut interval` <- gsub("]", "", df$`cut interval`, fixed=T)
df$`cut interval` <- gsub("60; 60; 60; 60; 60; 60; 60; 60; 60; 60; 60; 60; 60; 60; 60; 60", "60*16", df$`cut interval`, fixed=T)
df$`cut interval` <- gsub("120; 120; 120; 120; 120; 120; 120; 120", "120*8", df$`cut interval`, fixed=T)
df$`cut interval` <- gsub("240; 240; 240; 240", "240*4", df$`cut interval`, fixed=T)

df1 <- df %>% pivot_longer(c(Accuracy, Fscore), names_to='Metric')
g1 <- df1 %>%
  filter(!is.na(`cut type`)) %>%
  ggplot() +
  aes(x=`cut type`, y=value, fill=`cut type`) +
  geom_boxplot() +
  scale_fill_brewer(palette="Greys") +
  theme_bw() +
  facet_wrap(vars(Metric), scales="free") + 
  ggtitle('Cut type') +
  theme(legend.position="none", axis.title.y=element_blank())

g2 <- df1 %>%
  filter(!is.na(`cut perc`)) %>%
  ggplot() +
  aes(x=`cut perc`, y=value, fill=`cut perc`) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  facet_wrap(vars(Metric), scales = "free") + 
  ggtitle('Cut percentage') +
  theme(legend.position="none", axis.title.y=element_blank())

g3 <- df1 %>%
  filter(!is.na(`cut interval`)) %>%
  ggplot() +
  aes(x=`cut interval`, y=value, fill=`cut interval`) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  facet_wrap(vars(Metric), scales = "free") + 
  ggtitle('Cut interval') +
  theme(legend.position="none", axis.title.y=element_blank(), axis.text.x=element_text(angle=45, hjust=1))

g4 <- df1 %>%
  filter(!is.na(continuation)) %>%
  ggplot() +
  aes(x=continuation, y=value, fill=continuation) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  facet_wrap(vars(Metric), scales = "free") + 
  ggtitle('Continuation') +
  theme(legend.position="none", axis.title.y=element_blank())

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

multiplot(g1, g2, g3, g4, cols=1)

## Descriptive statistics
descriptive <- function(d1, c1){
  d2 <- d1 %>% pivot_longer(c(Accuracy, Fscore), names_to='Metric')
  g <- d2 %>%
    filter(!is.na(!!sym(c1))) %>%
    ggplot() +
    aes(x=!!sym(c1), y=value, fill=!!sym(c1)) +
    geom_boxplot() +
    scale_fill_brewer(palette = "Greys") +
    theme_bw() +
    facet_wrap(vars(Metric), scales = "free") + 
    theme(legend.position="none", axis.title.y=element_blank())
  
  df_w <- d1 %>% filter(!is.na(!!sym(c1))) %>% 
    select(c(c1, Accuracy, Fscore, Fold)) %>%
    pivot_wider(names_from=c(!!sym(c1)), values_from=c(Accuracy, Fscore), names_sep='')
  
  r <- df_w %>% select(-c(Fold)) %>%
    dplyr::summarise_all(list('min'=min,'mean'=mean,'median'=median,'sd'=sd,'max'=max)) %>%
    tidyr::pivot_longer(cols = everything(),
                        names_sep = "_",
                        names_to  = c("stats", ".value"))
  
  r <- r %>% column_to_rownames("stats")
  m1 <- colMeans(t(apply(-(df_w %>% select(starts_with("Accuracy"))), 1, rank)))
  m2 <- colMeans(t(apply(-(df_w %>% select(starts_with("Fscore"))), 1, rank)))
  
  print(friedman.test(as.matrix(df_w %>% select(starts_with('Accuracy')))))
  print(friedman.test(as.matrix(df_w %>% select(starts_with('Fscore')))))
  
  r <- merge(r, data.frame(rank=c(m1, m2)), by="row.names")
  r$Metric <- rep(c('Accuracy', 'F-score'), each=nrow(r)/2)
  r$Option <- c1
  colnames(r)[colnames(r) == "Row.names"] <- "Setting"
  r$Setting <- gsub("Accuracy", "", r$Setting, fixed=T)
  r$Setting <- gsub("Fscore", "", r$Setting, fixed=T)
  
  r1 <- r[,c(9,8,1,4,6,7)]
  
  #return(r1[order(r1[,1], r1[,2], r1[,3]),])
  return(list(g, r1[order(r1[,1], r1[,2], r1[,6]),]))
}

cols <- c('cut type', 'cut perc', 'cut interval', 'continuation')
charts <- c()
tab <- NULL

for (col1 in cols){
  t1 <- descriptive(df, col1)
  charts <- c(charts, t1[[1]])
  if (is.null(tab)){
    tab <- t1[[2]]
  }else{
    tab <- rbind(tab, t1[[2]])
  }
  print(t1[[2]])
}
print(tab %>% kable(format='latex', booktabs=T, digits=3))

#t1 <- descriptive(df, 'cut type')


df %>% filter(!is.na(cut_interval)) %>%
  ggwithinstats(x=cut_interval, y=Accuracy, type = "np",
                p.adjust.method = "bonferroni",
                title = "Dynamic vs. Evolutionary Feature Selection",
                results.subtitle = T,
                sample.size.label = F,
                point.path = F,
                centrality.path = F
  )


# H2: DYN vs EVO

data <- read.csv('data.csv')
data$Meta <- gsub("FS", "", data$Meta, fixed=T)
data$Meta <- gsub(" ", "", data$Meta, fixed=T)

## Charts
### Accuracy
data %>%
 filter(!(Meta %in% "")) %>%
 ggplot() +
 aes(x=Dataset, y=Accuracy, fill=Meta) +
 geom_boxplot() +
 scale_fill_brewer(palette="Greys") +
 theme_bw()

data %>%
  filter(!(Meta %in% "")) %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Meta, y = Accuracy, fill = Meta) +
  geom_boxplot() + 
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  facet_wrap(vars(Dataset), scales = "free") +
  theme(legend.position = "none")
  #theme(legend.title = element_blank(), legend.position = c(0.75, 0.22))

### Fscore
data %>%
  filter(!(Meta %in% "")) %>%
  ggplot() +
  aes(x=Dataset, y=Fscore, fill=Meta) +
  geom_boxplot() +
  scale_fill_brewer(palette="Greys") +
  theme_bw()

data %>%
  filter(!(Meta %in% "")) %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Meta, y = Fscore, fill = Meta) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  facet_wrap(vars(Dataset), scales = "free") + 
  ylab('F-score') +
  theme(legend.position = "none")

## Statistics

data_wide <- data %>% filter(!(Meta %in% "")) %>% select(c(Dataset, Fold, Meta, Nia, Accuracy, Fscore)) %>%
  pivot_wider(names_from=c(Meta), values_from=c(Accuracy, Fscore), names_sep='')


### Accuracy
data %>%
  filter(!(Meta %in% "")) %>%
  ggwithinstats(x=Meta, y=Accuracy, type = "np",
                title = "Dynamic vs. Evolutionary Feature Selection",
                results.subtitle = FALSE,
                sample.size.label = FALSE,
                point.path = FALSE
  )

data_wide$AccuracyDiff <- data_wide$AccuracyEvo - data_wide$AccuracyDyn
shapiro.test(data_wide$AccuracyDiff)
wilcox.test(data_wide$AccuracyEvo, data_wide$AccuracyDyn, paired=T)

### Fscore
data_wide$FscoreDiff <- data_wide$FscoreEvo - data_wide$FscoreDyn
shapiro.test(data_wide$FscoreDiff)
wilcox.test(data_wide$FscoreEvo, data_wide$FscoreDyn, paired=T)

### Descriptives
summary(data_wide)
s1 <- data_wide %>% select(c(AccuracyEvo, AccuracyDyn, FscoreEvo, FscoreDyn)) %>% 
  dplyr::summarise_all(c('median','max')) %>%
  tidyr::pivot_longer(cols = everything(),
                      names_sep = "_",
                      names_to  = c("stats", ".value"))

cm1 <- colMeans(t(apply(-data_wide[c('AccuracyEvo', 'AccuracyDyn')], 1, rank)))
cm2 <- colMeans(t(apply(-data_wide[c('FscoreEvo', 'FscoreDyn')], 1, rank)))
s1$Rank <- c(cm1, cm2)

print(s1 %>% kable(format='latex', booktabs=T, digits=3))

# H3: By Nia: Dyn vs Evo


## Accuracy
data %>%
  filter(!(Meta %in% "")) %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Meta, y = Accuracy, fill = Meta) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  facet_wrap(vars(Nia), scales = "free")


## Fscore
data %>%
  filter(!(Meta %in% "")) %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Meta, y = Fscore, fill = Meta) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  facet_wrap(vars(Nia), scales = "free")

## Statistics

by(data_wide$AccuracyDiff, data_wide$Nia, shapiro.test)
by(data_wide, data_wide$Nia, function(x){wilcox.test(x$AccuracyDyn, x$AccuracyEvo)})

by(data_wide$FscoreDiff, data_wide$Nia, shapiro.test)
by(data_wide, data_wide$Nia, function(x){wilcox.test(x$FscoreDyn, x$FscoreEvo)})

## Descriptives

summary(data_wide)
s2 <- data_wide %>% select(c(AccuracyEvo, AccuracyDyn, FscoreEvo, FscoreDyn, Nia)) %>% 
  group_by(Nia) %>%
  dplyr::summarise_all(c('median','max')) %>%
  tidyr::pivot_longer(cols=2:9,
                      names_sep = "_",
                      names_to  = c("stats", ".value"))

by(data_wide, data_wide$Nia, function(x){colMeans(t(apply(-x[c('AccuracyEvo', 'AccuracyDyn')], 1, rank)))})
by(data_wide, data_wide$Nia, function(x){colMeans(t(apply(-x[c('FscoreEvo', 'FscoreDyn')], 1, rank)))})
print(s1 %>% kable(format='latex', booktabs=T, digits=3))


# H4: vs Nias

## Accuracy
data %>%
  filter(Meta %in% "Dyn") %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Nia, y = Accuracy, fill = Nia) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw()

data %>%
  filter(!(Meta %in% "")) %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Nia, y = Accuracy, fill = Nia) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  theme(legend.position = "none", axis.title.x=element_blank()) +
  facet_wrap(vars(Dataset), scales = "free")

## Fscore
data %>%
  filter(Meta %in% "Dyn") %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Nia, y = Fscore, fill = Nia) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw()

data %>%
  filter(!(Meta %in% "")) %>%
  filter(!(Nia %in% "")) %>%
  ggplot() +
  aes(x = Nia, y = Fscore, fill = Nia) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Greys") +
  theme_bw() +
  ylab('F-score') +
  theme(legend.position = "none", axis.title.x=element_blank()) +
  facet_wrap(vars(Dataset), scales = "free")

data_wide1 <- data %>% filter(Meta %in% "Dyn") %>% select(c(Dataset, Fold, Meta, Nia, Accuracy, Fscore)) %>%
  pivot_wider(names_from=c(Nia), values_from=c(Accuracy, Fscore), names_sep='')
data1 <- data %>% filter(Meta %in% "Dyn") %>% select(c(Dataset, Fold, Nia, Accuracy, Fscore))
data1$DatasetFold <- paste(data1$Dataset, data1$Fold)
friedman.test(y=data1$Accuracy, groups=data1$Nia, blocks=data1$DatasetFold)
friedman.test(y=data1$Fscore, groups=data1$Nia, blocks=data1$DatasetFold)

### Descriptives
s2 <- data1 %>% select(c(Accuracy, Fscore, Nia)) %>% 
  group_by(Nia) %>%
  dplyr::summarise_all(c('median','max')) %>%
  tidyr::pivot_longer(cols = 2:5,
                      names_sep = "_",
                      names_to  = c("stats", ".value"))


colMeans(t(apply(-(data_wide1 %>% select(starts_with("Accuracy"))), 1, rank)))
colMeans(t(apply(-(data_wide1 %>% select(starts_with("Fsc"))), 1, rank)))


print(s2 %>% kable(format='latex', booktabs=T, digits=3))
