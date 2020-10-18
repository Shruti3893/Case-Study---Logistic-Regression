#Classify whether application accepted or not using Logistic regression
#Problem Statement for Credit Card Data

install.packages("readr")
library(readr)
Credit<-read.csv("C://Users//Lenovo//Desktop//ExcelR//Assignments//Logistic Regression//creditcard.csv")
View(Credit)

sum(is.na(Credit)) # To get the count of NA Values
attach(Credit)
str(Credit)
summary(Credit)
var(Credit)
hist(Credit$reports)
hist(Credit$age)
hist(Credit$income)
hist(Credit$share)
hist(Credit$expenditure)
hist(Credit$majorcards)
boxplot(Credit$reports)
boxplot(Credit$income)
skewness(Credit$reports)
skewness(Credit$age)
skewness(Credit$income)
kurtosis(Credit$reports)
kurtosis(Credit$age)
kurtosis(Credit$income)

logit<-glm(factor(card)~.,family=binomial,data = Credit)
summary(logit)


#odd's ratio'
exp(coef(logit))
table(Credit$card)

# Confusion matrix table 
prob <- predict(logit,type=c("response"),Credit)
prob
confusion<-table(prob>0.5,Credit$card)
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
rocrpred<-prediction(prob,Credit$card)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained
