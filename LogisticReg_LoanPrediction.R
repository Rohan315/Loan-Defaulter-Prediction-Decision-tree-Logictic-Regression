# clear current work space and console
# command/control + enter
rm(list=ls())
cat("\014")

# Use menu /Session/Set Working Directory/Choose Directory Or command below to set working directory
setwd("U:/Spring 2022/Business Analytics with R/path")

# load in the data file into loan data frame
loan<- read.csv("LR_loandata.csv",fileEncoding="UTF-8-BOM", stringsAsFactors = FALSE)
#Without file encoding, first column name was coming with some special characters. 
# We will predict if an individual would default their loan or pay the loan

#Creating factors (or say dummy variables)
loan$not.fully.paid <- as.factor(loan$not.fully.paid)
loan$purpose <- as.factor(loan$purpose)
loan$credit.policy <- as.factor(loan$credit.policy)
loan$inq.last.6mths <- as.factor(loan$inq.last.6mths)
loan$delinq.2yrs <- as.factor(loan$delinq.2yrs)
loan$pub.rec <- as.factor(loan$pub.rec)
loan$log.annual.inc <- exp(loan$log.annual.inc) 
#we can do things both ways by taking normal salary value of log of salary 
#sometimes log gives better results as it standardizes data but sometimes it doesn't give. so we have to try both things and then choose one


#It is important to scale data before building any model. 
#In our case, we will see that features are normally distributed, look stable and thus no scaling required. Otherwise PCA would have come into play.


##### Basic Statistics #####
# structure of the data.frame
str(loan)

any(is.na(loan))
# summary of the data.frame. We have no null or NA values. We did initial data cleaning in CSV file itself.
summary(loan)  #summary also tells no NA's present

#We can clearly see there is a negative correlation: interest rate decreases as fico (credit) score increases.
cor(loan$int.rate, loan$fico)
# p-value <0.1 => reject hypothesis <=> interest rate & fico(credit) score are significantly correlated
cor.test(loan$int.rate, loan$fico)
# correlation among multiple variables. We can see how features and strongly and weakly correlated.
cor(loan[, c("int.rate", "log.annual.inc", "dti", "fico", "days.with.cr.line")])
#At end we will be using this to eliminate variables and thus reduce repetition of information
#High correlation implies we cn use any one of the variable out of both

# get proportions. Most of the people(approx 80%) paid back the loan
mytab <- table(loan$not.fully.paid)  
prop.table(mytab)

##### Data Visualization (Exploratory Data Analysis) #####
#We can also visualize this to see how many loan repayment is.
ggplot(loan, aes(x = not.fully.paid)) + geom_histogram(stat = "count", aes(fill = factor(not.fully.paid)))
#Around 8000 paid back the loan

#Histogram to do basic EDA
g <- ggplot(loan, aes(fico)) #checking  fico score
g + geom_histogram(binwidth=10)#looks normally distributed

# side by side histograms
# facet into columns by people who paid loan or not (0 stands for people who paid the loan)
#g + geom_histogram(binwidth=5) + facet_grid(.~not.fully.paid)  
# facet into rows by churn. Row wise it seems to be easy for comparison.
g + geom_histogram(binwidth=10) + facet_grid(not.fully.paid ~ .)
# People with low credit (<700) or fico score are mostly the ones who did not pay back the loan.

#credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
g + geom_histogram(binwidth=10) + facet_grid(credit.policy ~ .)
#It is very clear that people above fico 655+ only qualify for a loan from lendingclub
#moreover it can also be seen that a score above 660 will not guarantee you a loan. There can be multiple factors deciding this(Example: person is is blacklisted by govt. agencies).
# density plot
ggplot(loan, aes(fico)) + geom_density() #majority concentration is around score 700

# Normal Distribution curve of interest rate #seems centered around 13% , right skewed plot
rate <- loan$int.rate
I <- hist(rate, breaks=10, col="blue", xlab="Interest rate",main="Interest rate distribution curve")



##### Splitting Data set #####
set.seed(123) # for reproducible results
train <- sample(1:nrow(loan), nrow(loan)*(2/3)) # replace=FALSE by default

# Use the train index set to split the dataset
#  train_data for building the model
#  test_data for testing the model
train_data  <- loan[train,]   # 6385 rows
test_data  <- loan[-train,]   # the other 3193 rows

##### Building logistic regression model #####
#using glm() function or "Generalized Linear Model" to perform logistic regression
#first we are using all the features  to train model and later use our knowledge to use specific features

log.reg <- glm(not.fully.paid ~ ., data = train_data, family = "binomial")
summary(log.reg)

#Tuning our model. This is where domain knowledge comes into existence.
#Clearly Credit.policy is not of much use here
#Interest rate is highly correlated to fico so eliminating it also. This is what we highlighted in our initial steps during EDA.
log.reg.rev <- glm(not.fully.paid ~ purpose + installment + log.annual.inc + fico + revol.bal +  pub.rec, data = train_data, family = "binomial")
summary(log.reg.rev)
round(data.frame(summary(log.reg.rev)$coefficients, oddSR= exp(coef(log.reg.rev)) ),2)

#making Predictions using test data. The feature to be predicted is at index 14. Note: Index starts at 1 and not 0 as in python.
predict_loan <- predict(log.reg.rev,test_data[-14], type = "response")
#Notice that we get continuous values and not just 0 or 1
head(predict_loan, n = 30)

#as we want the dependent variable (not.fully.paid) take values binary values: 0 or 1.
#we can  make it to 0 or 1 values by assigning any values below 0.5 as 0, and above 0.5 as 1 (this can be anything else other than 0.5 based on our need)
binary_predict <- as.factor(ifelse(predict_loan > 0.5, 1, 0)) # Fine tuned but 0.5 is giving highest accuracy
head(binary_predict, n = 30)


##### Model Evaluation #####
#Confusion matrix. Please use caret package
set.seed(123)
confusionMatrix(binary_predict, test_data$not.fully.paid)
#Accuracy is almost 84.5%
#Our model correctly predict off the test dataset that 2693 individuals (true positive) paid their loan while 6 people default (true negative)
#That means that given data points of 3193 observations from our test data, our model has correctly predicted 2699 outcomes. 
#But the confusion matrix indicates that our model has false negative of 4, which means that our model predicted 4 persons will default but they actually paid the loan, and 490 false positive which our model predicted not default but they actually default.

#ROC curve and AUC. Higher the AUC, the better the model performance.
#We will go ahead with choosing logistic regression over decision tree as it has better accuracy and auc.

r<-roc(test_data$not.fully.paid,predict_loan, auc=TRUE)
plot.roc(r)
#ignore waring
auc(r) #auc = 0.65

#Based on our analysis we can say that:
#People with high fico score and salary will pay back the loan and mostly these people fit into lending club's criteria of approving the loan
#At 84.53% accuracy, simple model like  logistic regression give powerful results
#We can make the model better by giving more data and applying multiple binary ML algorithms to get the best fit algorithm
#People with long history of credit line are also good at replaying the loan

