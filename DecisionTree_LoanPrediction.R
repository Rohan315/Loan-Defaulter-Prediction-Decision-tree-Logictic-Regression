# clear current workspace and console
# command/control + enter
rm(list=ls())
cat("\014")

# Use menu /Session/Set Working Directory/Choose Directory Or command below to set working directory
setwd("U:/Spring 2022/Business Analytics with R/path")

# load in the data file into loan_data data frame
loan_data <- read.csv("DT_loandata.csv",fileEncoding="UTF-8-BOM", stringsAsFactors = FALSE)
#Without file encoding, first column name was coming with some special characters. 

loan_data$not.fully.paid <- as.factor(loan_data$not.fully.paid)
loan_data$purpose <- as.factor(loan_data$purpose)
loan_data$credit.policy <- as.factor(loan_data$credit.policy)
loan_data$inq.last.6mths <- as.factor(loan_data$inq.last.6mths)
loan_data$delinq.2yrs <- as.factor(loan_data$delinq.2yrs)
loan_data$pub.rec <- as.factor(loan_data$pub.rec)
loan_data$log.annual.inc <- exp(loan_data$log.annual.inc)


##### (1) Basic Statistics #####
# structure of the data.frame
str(loan_data)
any(is.na(loan_data))
# summary of the data.frame. We have no null or NA values. We did initial data cleaning in CSV file itself.
summary(loan_data) #summary also tells no NA's present

#We can clearly see there is a negative correlation: interest rate decreases as fico (credit) score increases.
cor(loan_data$int.rate, loan_data$fico)
# p-value <0.1 => reject hypothesis <=> interest rate & fico(credit) score are significantly correlated
cor.test(loan_data$int.rate, loan_data$fico)
# correlation among multiple variables. We can see how features and strongly and weakly correlated.
cor(loan_data[, c("int.rate", "log.annual.inc", "dti", "fico", "days.with.cr.line")])
#At end we will be using this to eliminate variables and thus reduce repetition of information
#High correlation implies we cn use any one of the variable out of both

# get proportions. Most of the people(approx 80%) paid back the loan
mytab <- table(loan_data$not.fully.paid)  
prop.table(mytab)

#Histogram to to further exploratory analysis
g <- ggplot(loan_data, aes(fico))
g + geom_histogram(binwidth=10) #looks normally distributed

# side by side histograms
# facet into columns by people who paid loan or not (No stands for people who paid the loan)
#g + geom_histogram(binwidth=5) + facet_grid(.~not.fully.paid)  
# facet into rows by churn. Row wise it seems to be easy for comparison.
g + geom_histogram(binwidth=10) + facet_grid(not.fully.paid ~ .)
# People with low credit (<700) or fico score are mostly the ones who did not pay back the loan.

#credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
g + geom_histogram(binwidth=10) + facet_grid(credit.policy ~ .)
#Its is very clear that people above fico 655+ only qualify for a loan from lendingclub
#moreover it can also be seen that a score above 660 will not guarantee you a loan. There can be multiple factors deciding this.

# density plot to see overall trend
ggplot(loan_data, aes(fico)) + geom_density() #majority concentration is around score 700




# now we will split the data into testing and training data sets
# we will first randomly select 2/3 of the rows
set.seed(123) # for reproducible results
train <- sample(1:nrow(loan_data), nrow(loan_data)*(2/3)) # replace=FALSE by default

# Use the train index set to split the dataset
#  churn.train for building the model, churn.test for testing the model
loan.train <- loan_data[train,]   # 6385 rows
loan.test <- loan_data[-train,]   # the other 3193 rows

# Classification Tree with rpart
# install.packages('rpart')
library(rpart)
fit <- rpart(not.fully.paid ~ ., # formula, all predictors will be considered in splitting
             data=loan.train, # dataframe used
             #method="anova", 
             method="class", 
             control=rpart.control(xval=10, minsplit=100, cp=0), # xval: num of cross validation for gini estimation # minsplit=1000: stop splitting if node has 1000 or fewer obs
             parms=list(split="gini")) # criterial for splitting: gini default, entropy if set parms=list(split="information")fit

library(rpart.plot)
rpart.plot(fit, type = 1, extra = 1, main="Classification Tree for Loan Repayment Prediction")

loan.pred <- predict(fit,loan.test, type="class") 
loan.actual <- loan.test$not.fully.paid

length(loan.pred)
length(loan.actual)
table(loan.pred, loan.actual)

##### Model Evaluation #####
#Confusion matrix. Please use caret packageset.seed(123)
confusionMatrix(loan.pred, loan.actual)
#Accuracy is  83.4%

#ROC curve and AUC. Higher the AUC, the better the model performance.
#We will go ahead with choosing logistic regression over decision tree as it has better accuracy and auc

loan.pred <- predict(fit,loan.test, type="prob") 
r<-roc(loan.actual,loan.pred[,1])
plot.roc(r)
auc(r) #auc =0.4

#conclusion is made in Logistic regression notebook as we chose to that model.