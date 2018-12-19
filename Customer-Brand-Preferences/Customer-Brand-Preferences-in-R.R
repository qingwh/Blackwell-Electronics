# Title: Classification: Predict which Brand of Products Customers Prefer

# Last update: 2018.05.18

# File:  Customer-Brand-Preference-in-R.R

###############
# Project Notes
###############

# Summarize project: We will train a model to find out which of two brands of computers our customers prefer.  
# We will use  CompleteResponses.csv to train your model and build our predictive model and pick the best classifier. We will then use the
# apply your optimized model to predict the brand preference using SurveyIncomplete.csv   

###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

getwd()
setwd("C:/Users/admin/Desktop/task2")
dir()
################
# Load packages
################
install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("doParallel")
install.packages('e1071', dependencies=TRUE)
install.packages("C50")
library(caret)
library(corrplot)
library(doMC)
library(mlbench)
library(readr)
library(e1071)
library(C50)
#####################
# Parallel processing
#####################

library(doParallel) 

# Check number of cores and workers available 
detectCores()
getDoParWorkers()
cl <- makeCluster(detectCores()-1, type='PSOCK')
registerDoParallel(cl)
###############
# Import data
##############

## Load training and test set
CompleteResponses <- read.csv("CompleteResponses.csv", stringsAsFactors = FALSE, header=T)

## Load prediction set
SurveyIncomplete <- read.csv("SurveyIncomplete.csv", stringsAsFactors = FALSE, header=T)
################
# Evaluate data
################

#--- Training and Test Set ---#
str(CompleteResponses)  # 10,000 obs. of  7 variables 
summary(CompleteResponses)
# plot
hist(CompleteResponses$brand)
hist(CompleteResponses$car)
# check for missing values 
anyNA(CompleteResponses)
is.na(CompleteResponses)
#--- Prediction Set ---#
str(SurveyIncomplete)  # 5,000 obs. of  7 variables 
summary(SurveyIncomplete)
# check for missing values 
anyNA(SurveyIncomplete)
is.na(SurveyIncomplete)

#######################
#  Feature Engineering
#######################

#--- Training and test set ---#
# change data types
CompleteResponses$brand <- as.factor(CompleteResponses$brand)
# normalize
preprocessParams <- preProcess(CompleteResponses, method = c("center", "scale"))
CompleteResponses_N <- predict(preprocessParams, CompleteResponses)

#--- Prediction set ---#

# change data types
SurveyIncomplete$brand <- as.factor(SurveyIncomplete$brand)
# normalize
SurveyIncomplete_N <- predict(preprocessParams, SurveyIncomplete)

##########################
# Sampling train/test sets
##########################

# create the training partition 75 % of total obs, chosen randomly
set.seed(123)
inTraining <- createDataPartition(CompleteResponses_N$brand, p=0.75, list=FALSE)
# create training/testing dataset
trainSet <- CompleteResponses_N[inTraining,]   
testSet <- CompleteResponses_N[-inTraining,]   

# verify number of obs 
nrow(trainSet)
nrow(testSet)

################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)


##############
# Train model
##############

## ------- Decision Tree C5.0 ------- ##

# train/fit
set.seed(123)
C5.0_Fit <- train(brand~., data=trainSet, method="C5.0", trControl=fitControl, tuneLength = 2) 
C5.0_Fit

#C5.0 

#7501 samples
#6 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 6751, 6750, 6751, 6751, 6751, 6750, ... 
#Resampling results across tuning parameters:
  
  #model  winnow  trials  Accuracy   Kappa    
  #rules  FALSE    1      0.9078815  0.8062144
  #rules  FALSE   10      0.9171832  0.8229667
  #rules   TRUE    1      0.9090278  0.8081875
  #rules   TRUE   10      0.9196361  0.8281024
  #tree   FALSE    1      0.9067215  0.8024609
  #tree   FALSE   10      0.9181434  0.8260752
  #tree    TRUE    1      0.9085077  0.8062529
  #tree    TRUE   10      0.9195434  0.8292775

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 10, model = rules and winnow = TRUE.

# eval pred vars
predictors(C5.0_Fit)
#"salary" "age"    "elevel" "credit"
# eval variable imp
varImp(C5.0_Fit)
# C5.0 variable importance

#          Overall
# salary   100.00
# age       97.28
# elevel    74.40
# credit    53.70
# zipcode    0.00
# car        0.00


#--- Save top performing model ---#

saveRDS(C5.0_Fit, "C5.0_Fit.rds")  
# load and name model
C5.0_Fit <- readRDS("C5.0_Fit.rds")

## ------- Random Forest ------- ##

# train/fit
set.seed(123)

grid <- expand.grid(mtry=c(2,3,4,5,6))
rf_Fit <- train(brand~., data=trainSet, method="rf", trControl=fitControl, tuneGrid = grid) 
rf_Fit

#Random Forest 

#7501 samples
#6 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 6751, 6750, 6751, 6751, 6751, 6750, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#   2   0.9194233  0.8290806
#   3   0.9196366  0.8292169
#   4   0.9182633  0.8262440
#   5   0.9166106  0.8227617
#   6   0.9146111  0.8185235

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 3.

# eval pred vars
predictors(rf_Fit)
#"salary"  "age"     "elevel"  "car"     "zipcode" "credit" 
# eval variable imp
varImp(rf_Fit)
#rf variable importance

#         Overall
#salary   100.000
#age      63.718
#credit    8.593
#car       3.397
#zipcode   1.524
#elevel    0.000
##--- Compare metrics ---##

ModelFitResults <- resamples(list(C5.0=C5.0_Fit,rf=rf_Fit))
# output summary metrics for tuned models 
summary(ModelFitResults)

# Call:
# summary.resamples(object = ModelFitResults)

# Models: C5.0, rf 
# Number of resamples: 100 

# Accuracy 
# #    1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5.0 0.8988016 0.9146382 0.9199466 0.9194763 0.9253333 0.9440746    0
# rf   0.8958611 0.9146667 0.9187208 0.9196366 0.9253333 0.9453333    0

# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5.0 0.7846428 0.8161293 0.8278579 0.8277469 0.8409944 0.8794248    0
# rf   0.7806307 0.8168446 0.8279826 0.8292169 0.8408850 0.8834247    0


#--- Save top performing model ---#

saveRDS(rf_Fit, "rf_Fit.rds")  
# load and name model
rf_Fit <- readRDS("rf_Fit.rds")

#################
# Predict testSet
#################
# predict with rf
rf_Pred1 <- predict(rf_Fit, testSet)
rf_Pred1 
# performance measurement
postResample(rf_Pred1, testSet$brand)
#Accuracy     Kappa 
#0.9191677 0.8267631 
# plot predicted verses observed
plot(rf_Pred1,testSet$brand)

#C5.0 is a little better than Random Forest,so we predict new data with C5.0

##################
# Predict new data
##################

# predict with C5.0
C5.0_Pred2 <- predict(C5.0_Fit, SurveyIncomplete_N)
C5.0_Pred2
summary(C5.0_Pred2)
View(C5.0_Pred2)
###############
# Save datasets 
###############
output <- SurveyIncomplete
output$brand<- C5.0_Pred2
write.csv(output, file="C2.T2output.csv", row.names = TRUE)