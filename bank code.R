#Output variable -> y
#y -> Whether the client has subscribed a term deposit or not 
#Binomial ("yes" or "no")
#Problem Statement for Bank Data

install.packages("readr")
library(readr)
bank<-read.csv("C://Users//Lenovo//Desktop//ExcelR//Assignments//Completed//Logistic Regression//bank-full.csv", sep = ";")
View(bank)
summary(bank)
sum(is.na(bank)) # To get the count of NA Values
attach(bank)
var(bank)
sd(bank$age)
sd(bank$balance)
sd(bank$day)
sd(bank$duration)
sd(bank$campaign)
sd(bank$pdays)
sd(bank$previous)
install.packages("moments")
library(moments)
skewness(bank$age)
skewness(bank$balance)
skewness(bank$day)
skewness(bank$duration)
skewness(bank$campaign)
skewness(bank$pdays)
skewness(bank$previous)
kurtosis(bank$age)
kurtosis(bank$balance)
kurtosis(bank$day)
kurtosis(bank$duration)
kurtosis(bank$campaign)
kurtosis(bank$pdays)
kurtosis(bank$previous)
hist(bank$age)
hist(bank$balance)
hist(bank$day)
hist(bank$duration)
hist(bank$campaign)
hist(bank$pdays)
hist(bank$previous)
barplot(bank$age)
barplot(bank$balance)
barplot(bank$day)
barplot(bank$duration)
barplot(bank$campaign)
barplot(bank$pdays)
barplot(bank$previous)
boxplot(age,balance,day,duration,campaign,pdays,previous)
str(bank)

#Model building
logit<-glm(factor(y)~.,family=binomial,data = bank)
summary(logit)


#odd's ratio'
exp(coef(logit))
table(bank$y)

# Confusion matrix table 
prob <- predict(logit,type=c("response"),bank)
prob
confusion<-table(prob>0.5,bank$y)
probo <- prob>0.5
table(probo)
confusion

# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy
Error <- 1-Accuracy
Error

# ROC Curve 
install.packages("ROCR")
library(ROCR)
rocrpred<-prediction(prob,bank$y)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained