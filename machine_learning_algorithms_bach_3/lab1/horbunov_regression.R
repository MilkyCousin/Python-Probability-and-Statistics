library(corrplot)
library(car)
library(glmnet)
library(MASS)

# Много лишнего кода  

workspace_setup <- function()
{
  workspace <- dirname(sys.frame(1)$ofile)
  setwd(workspace)
  print(getwd())
}

workspace_setup()

filter.sample <- function(x)
{
  quantiles.x <- quantile(x, c(0.25, 0.5, 0.75))
  iqr.x <- quantiles.x[3] - quantiles.x[1]
  
  mask.filtered <- (quantiles.x[1] - 1.5 * iqr.x <= x) & (x <= quantiles.x[3] + 1.5 * iqr.x)
  x[mask.filtered]
}

cv <- function(x)
{
  sqrt(var(x))/mean(x)
}

skewness <- function(x)
{
  (1/((sqrt(var(x))^3)*length(x)))*sum((x-mean(x))^3)
}

r.sq.classic <- function(y.t, y.p, non.b.0 = F)
{
  sub.v <- ifelse(non.b.0, 0, mean(y.t))
  ss.err <- sum((y.t - y.p)^2)
  ss.ttl <- sum((y.t - sub.v)^2)
  1 - ss.err/ss.ttl
}

r.sq.adjusted <- function(y.t, y.p, n.par, non.b.0 = F)
{
  sub.v <- ifelse(non.b.0, 0, mean(y.t))
  ss.err <- sum((y.t - y.p)^2)/(length(y.t)-n.par-1)
  ss.ttl <- sum((y.t - sub.v)^2)/(length(y.t)-1)
  1 - ss.err/ss.ttl
}

mean.sq.err <- function(y.t, y.p)
{
  mean((y.t - y.p)^2)
}

f.test.bias.regr <- function(n, p, y.p, y.t)
{
  msm = sum((y.p - mean(y.t))^2)/(p-1)
  mse = sum((y.t - y.p)^2)/(n-p)
  msm/mse
}

#Для проведення регресiйного аналiзу потрбiно зробити наступне:
#-Побудувати ОНК, зробити висновки, щодо якостi моделi та ОНК.                                          +
#-Спробувати покращити оцiнку, шляхом використання гребеневої регресiї.                                 +
#-Спробувати зменшити розмiрнiсть простору регресорiв використовуючи метод головних компонент.          +
#-Здiйснити процедуру оптимального вiдбору регресорiв.                                                  ?

#-Напишiть функцiю, що примає число N, а також величини $\beta_i, i = 0 : 3$ як параметри. 
#Функцiя повинна згенерувати три регресора $X_i, i = 1 : 3$ зi стандартним нормальним розподiлом, 
#а також похибку з розподiлом $\mathcal{N}(0, 0.1)$ розмiрностi N. Функцiя повинна повернути дата 
#фрейм з колонкою $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \varepsilon$, а також 
#колонками $X_1; X_2; X_3$. Пiсля цього обчислiть ОНК використовуючи безпосереднью !формулу!, 
#!бiблiотечну функцiю! або за допомогою власної iмплементацiї методу !градiєнтного спуску! 
#(теж оформленого в окрему функцiю). Для рiзних значень параметра N занотуйте час який потрiбен 
#для обчислення ОНК кожним iз трьох методiв. Знайдiть таке N при якому один iз методiв буде працювати 
#вiдчутно повiльнiше. Для чистоти експерименту проведiть його 10 разiв для кожного N.

biased.calc <- function(N, b1, b2, b3)
{
  x <- replicate(3, rnorm(N))
  e <- rnorm(N, sd=.1)
  y <- b1 * x[,1] + b2 * x[,2] + b3 * x[,3] + e
  r <- data.frame(matrix(ncol = 4, nrow = N))
  c.names <- c('y', 'x1', 'x2', 'x3')
  colnames(r) <- c.names
  r[c.names[1]] <- y
  r[c.names[-1]] <- x
  r
}

#Файл з даними: NAICExpense.csv
#Вiдгук: EXPENSES
    #Регресори: RBC, STAFFWAGE, AGENTWAGE, LONGLOSS, SHORTLOSS

#Робота повинна мiстити код, та його iнтерпритацiю. Обов’язково має бути вказано:
#1. ОНК, наявнiсть кореляцiй мiж регресорами, якiсть ОНК (її дисперсiя, якiсть                +
#прогнозування, наведена дiаграма залишкiв, коефiцiєнт детермiнацiї та результати             +
#тесту Фiшера).                                                                               +
#2. Пiдбiр параметра λ, покращення (чи вiдсутнiсть покращення) якостi прогнозу,               +
#зменшення дисперсiї.                                                                         ?
#3. Пiдбiр оптимальної кiлькостi головних компонент, порiвняння результатiв зi звичайним МНК. ?
#4. Метод що використовувася для оптимального вiдбору, та досягнутi результати.               -

col.names <- c('EXPENSES', 'RBC', 'STAFFWAGE', 'AGENTWAGE', 'LONGLOSS', 'SHORTLOSS')

df <- na.omit(read.csv('NAICExpense.csv'))
df <- df[col.names]
df['RBC'] <- df['RBC'] / 10e8

p = 0.8
train.size <- floor(p * nrow(df))

seed.n <- 1
set.seed(seed.n)

print(summary(df))

df.train.i <- sample(seq_len(nrow(df)), size=train.size)

df.train <- df[df.train.i, ]
df.test <- df[-df.train.i, ]

plot(df.train[col.names[-1]], cex=0.1)

scatterplotMatrix(df.train[col.names], diagonal=list(method="histogram"), smooth=F, cex=0.25, col='black')

df.train.corr.s <- cor(df.train[col.names[-1]], method='spearman')
corrplot(df.train.corr.s, title='corr of regressors by spearman', mar=c(0,0,1,0), method = "color", addCoef.col='black')

df.train.corr.p <- cor(df.train[col.names[-1]], method='pearson')
corrplot(df.train.corr.p, title='corr of regressors by pearson', mar=c(0,0,1,0), method = "color", addCoef.col='black')

par(mar=c(4,4,1,2))
br=28

m.def <- par('mfrow')
par(mfrow=c(2,3))
for(col.name in col.names)
{
  hist(filter.sample(as.numeric(df.train[,col.name])), breaks = br, main=col.name, xlab='X')
}
par(mfrow=c(1,6))
for(col.name in col.names)
{
  boxplot(filter.sample(as.numeric(df.train[,col.name])), main=col.name)
}
par(mfrow=m.def)

# LM1: Y ~ RBC + STAFFWAGE + AGENTWAGE + LONGLOSS + SHORTLOSS

print('LM: EXPENSES ~ RBC + STAFFWAGE + AGENTWAGE + LONGLOSS + SHORTLOSS')

l.model <- lm(EXPENSES ~ RBC + STAFFWAGE + AGENTWAGE + LONGLOSS + SHORTLOSS, data=df.train)

print(summary(l.model))

# LM1: visualization

fitted.out <- boxplot.stats(l.model$fitted.values)$out

par(mfrow=c(1,2))
plot(
  l.model$fitted.values[!l.model$fitted.values %in% fitted.out], 
  l.model$residuals[!l.model$fitted.values %in% fitted.out], 
  xlab='Fitted', ylab='Residuals', cex=0.25
)
abline(c(0,0), col='red')

plot(
  l.model$fitted.values[!l.model$fitted.values %in% fitted.out], 
  df.train$EXPENSES[!l.model$fitted.values %in% fitted.out], xlab='Predicted', ylab='True', cex=0.25,
  main='Predicted-True comparison plot | LM:1'
)
abline(c(0,1), col='red')
par(mfrow=m.def)

# LM1: QQ-plotting

qqnorm(l.model$res, cex=0.25)
qqline(l.model$res, cex=0.25, col='red')

# LM1: coef distrib

l.influence <- lm.influence(l.model)

par(mfrow=c(2,3))
for(j in 2:length(col.names))
{
  qqnorm(l.influence$coef[,j], cex=0.25, main=col.names[j])
  qqline(l.influence$coef[,j], cex=0.25, col='red')
}
par(mfrow=m.def)

# LM2: unbiased (b0 = 0), Y ~ RBC + LONGLOSS + SHORTLOSS

print('LM: EXPENSES ~ RBC + LONGLOSS + SHORTLOSS')

l.model.s.0 <- lm(EXPENSES ~ RBC + LONGLOSS + SHORTLOSS - 1, data=df.train)

print(summary(l.model.s.0))

# LM2: visualization

fitted.out <- boxplot.stats(l.model.s.0$fitted.values)$out

par(mfrow=c(1,2))
plot(
  l.model.s.0$fitted.values[!l.model.s.0$fitted.values %in% fitted.out], 
  l.model.s.0$residuals[!l.model.s.0$fitted.values %in% fitted.out], 
  xlab='Fitted', ylab='Residuals', cex=0.25
)
abline(c(0,0), col='red')

plot(
  l.model.s.0$fitted.values[!l.model.s.0$fitted.values %in% fitted.out], 
  df.train$EXPENSES[!l.model.s.0$fitted.values %in% fitted.out], xlab='Predicted', ylab='True', cex=0.25,
  main='Predicted-True comparison plot | LM:2'
)
abline(c(0,1), col='red')
par(mfrow=m.def)

# LM2: QQ-plotting

qqnorm(l.model.s.0$res, cex=0.25)
qqline(l.model.s.0$res, cex=0.25, col='red')

l.s.0.influence <- lm.influence(l.model.s.0)

par(mfrow=c(3,1))
for(j in 1:length(col.names[c(2,5,6)])) #!!!
{
  qqnorm(l.s.0.influence$coef[,j], cex=0.25, main=col.names[c(2,5,6)][j])
  qqline(l.s.0.influence$coef[,j], cex=0.25, col='red')
}
par(mfrow=m.def)

# LM3: unbiased (b0 = 0), Y ~ LONGLOSS + SHORTLOSS + sqrt(RBC + SHORTLOSS)

l.model.n.0 <- lm(EXPENSES ~ LONGLOSS + SHORTLOSS + sqrt(abs(RBC + SHORTLOSS)) - 1, data=df.train)

print(summary(l.model.n.0))

# LM3: visualization
  
fitted.out <- boxplot.stats(l.model.n.0$fitted.values)$out

par(mfrow=c(1,2))
plot(
  l.model.n.0$fitted.values[!l.model.n.0$fitted.values %in% fitted.out], 
  l.model.n.0$residuals[!l.model.n.0$fitted.values %in% fitted.out], 
  xlab='Fitted', ylab='Residuals', cex=0.25
)
abline(c(0,0), col='red')

plot(
  l.model.n.0$fitted.values[!l.model.n.0$fitted.values %in% fitted.out], 
  df.train$EXPENSES[!l.model.n.0$fitted.values %in% fitted.out], xlab='Predicted', ylab='True', cex=0.25,
  main='Predicted-True comparison plot | LM:3'
)
abline(c(0,1), col='red')
par(mfrow=m.def)

# LM3: QQ-plotting

qqnorm(l.model.n.0$res, cex=0.25)
qqline(l.model.n.0$res, cex=0.25, col='red')

l.n.0.influence <- lm.influence(l.model.n.0)

names.new <- c(col.names[c(5,6)], 'sqrt(|RBS+SHORTLOSS|)')

par(mfrow=c(3,1))
for(j in 1:length(names.new)) #!!!
{
  qqnorm(l.n.0.influence$coef[,j], cex=0.25, main=names.new[j])
  qqline(l.n.0.influence$coef[,j], cex=0.25, col='red')
}
par(mfrow=m.def)

# LM4: Ridge, Y ~ RBC + STAFFWAGE + AGENTWAGE + LONGLOSS + SHORTLOSS

lambda.arr.min <- -4
lambda.arr.max <- 1
lambda.arr <- 10^seq(lambda.arr.min, lambda.arr.max, by=.1)

l.model.r <- cv.glmnet(
  x = data.matrix(df.train[col.names[-1]]), y=data.matrix(df.train[col.names[1]]),
  family = 'gaussian',
  alpha = 0, lambda = lambda.arr
)

plot(l.model.r, main='MSE - Lambda plot')

lambda.opt <- l.model.r$lambda.min
l.model.r.fit <- l.model.r$glmnet.fit

print(lambda.opt)

l.model.r.o <- glmnet(
  x = data.matrix(df.train[col.names[-1]]), y=data.matrix(df.train[col.names[1]]),
  family = 'gaussian', 
  alpha = 0, lambda = lambda.opt
)

y.prdct <- predict(l.model.r.o, s=lambda.opt, data.matrix(df.train[col.names[-1]]))
y.resid <- df.train$EXPENSES - y.prdct

print(l.model.r.o$beta)
print(mean.sq.err(df.train$EXPENSES,y.prdct))
print(r.sq.classic(df.train$EXPENSES,y.prdct, non.b.0 = T))
print(r.sq.adjusted(df.train$EXPENSES,y.prdct, 5, non.b.0 = T))

# LM4: visualization

fitted.out <- boxplot.stats(y.prdct)$out

par(mfrow=c(1,2))
plot(
  y.prdct[!y.prdct %in% fitted.out], 
  y.resid[!y.prdct %in% fitted.out], 
  xlab='Fitted', ylab='Residuals', cex=0.25
)
abline(c(0,0), col='red')

plot(
  y.prdct[!y.prdct %in% fitted.out], 
  df.train$EXPENSES[!y.prdct %in% fitted.out], xlab='Predicted', ylab='True', cex=0.25,
  main='Predicted-True comparison plot | LM:4'
)
abline(c(0,1), col='red')
par(mfrow=m.def)

# LM4: QQ-plotting

qqnorm(y.resid, cex=0.25)
qqline(y.resid, cex=0.25, col='red')

# LM5: PCA

pca.df.train <- princomp(df.train[col.names[-1]], cor=T)
pca.scores <- pca.df.train$scores

print(summary(pca.df.train))
plot(pca.df.train)

l.pca <- lm(df.train$EXPENSES ~ pca.scores[,1:3])

print(summary(l.pca))

# LM5: visualization

fitted.out <- boxplot.stats(l.pca$fitted.values)$out

par(mfrow=c(1,2))
plot(
  l.pca$fitted.values[!l.pca$fitted.values %in% fitted.out], 
  l.pca$residuals[!l.pca$fitted.values %in% fitted.out], 
  xlab='Fitted', ylab='Residuals', cex=0.25
)
abline(c(0,0), col='red')

plot(
  l.pca$fitted.values[!l.pca$fitted.values %in% fitted.out], 
  df.train$EXPENSES[!l.pca$fitted.values %in% fitted.out], xlab='Predicted', ylab='True', cex=0.25,
  main='Predicted-True comparison plot | LM:5'
)
abline(c(0,1), col='red')
par(mfrow=m.def)

# LM5: QQ-plotting

qqnorm(l.model.n.0$res, cex=0.25)
qqline(l.model.n.0$res, cex=0.25, col='red')

l.n.0.influence <- lm.influence(l.model.n.0)

names.new <- col.names[1:3]

par(mfrow=c(3,1))
for(j in 1:length(names.new)) #!!!
{
  qqnorm(l.n.0.influence$coef[,j], cex=0.25, main=names.new[j])
  qqline(l.n.0.influence$coef[,j], cex=0.25, col='red')
}
par(mfrow=m.def)