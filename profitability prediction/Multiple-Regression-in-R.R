# Title: Multiple Regression in R

# Last update: 2018.04.30

# File:  Multiple-Regression-in-R.R

###############
# Project Notes
###############

# Summarize project:We will train a model to Predicting sales of four different 
# product types: PC, Laptops, Netbooks and Smartphones using existingproductattributes2017.csv as the training set, 
#picking the best algorithm. We will then use the best model to predict the testset using newproductattributes2017.csv  


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

getwd()
setwd("C:/Users/admin/Desktop/task3")
dir()
################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("doParallel")
install.packages('e1071', dependencies=TRUE)
install.packages("gbm")
library(caret)
library(corrplot)
library(mlbench)
library(readr)
library(e1071)
library(gbm)

#####################
# Parallel processing
#####################

library(doParallel) 

# Check number of cores and workers available 
detectCores()
getDoParWorkers()
cl <- makeCluster(detectCores()-1, type='PSOCK')
registerDoParallel(cl)
# to stop cluster/parallel:
# stopCluster(cl) 
# to reactivate parallel:
#registerDoParallel(cl)

###############
# Import data
##############

## Load training and test set
existing_product <- read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE, header=T)

## Load prediction set
new_product<- read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE, header=T)

################
# Evaluate data
################

#--- Training and Test Set ---#
str(existing_product)  # 80obs. of  18 variables 
summary(existing_product)
head(existing_product)
tail(existing_product)
names(existing_product)
attributes(existing_product)

# plot
hist(existing_product$Volume)
plot(existing_product$Price, existing_product$Volume)
qqnorm(existing_product$Volume)  

# check the impact of customer and service reviews have on sales volume.  

corrreview <- cor(existing_product[,4:10], existing_product$Volume,method = "pearson")

corrreview
# plot correlation matrix
corrplot(corrreview, method = "circle")

# check for missing values 
anyNA(existing_product)
is.na(existing_product)

#--- Prediction Set ---#
str(new_product)  # 24 obs.of 18 variables 
summary(new_product)

# check for missing values 
anyNA(new_product)
is.na(new_product)

#################
# Feature removal
#################

# remove ID and obvious features

#--- Training and test set ---#
existing_product$ProductNum<- NULL
str(existing_product) # 80obs. of  16 variables 

#--- Prediction set ---#
new_product$ProductNum<- NULL
str(new_product) # 24obs. of  16 variables 

#############
# Preprocess
#############

#--- Training and test set ---#

#address missing values

existing_product$BestSellersRank<- NULL

# normalize

preprocessParams <- preProcess(existing_product[,1:15], method = c("center", "scale"))
print(preprocessParams)  #scaled (14)centered (14)
existing_product_N <- predict(preprocessParams, existing_product)
str(existing_product_N)# 80obs. of  16 variables 

# dummify the data 

dummies <- dummyVars(" ~ .", data = existing_product_N)

existing_product_r<- data.frame(predict(dummies, newdata = existing_product_N))
str(existing_product_r) # 80obs. of  27 variables 

#--- Prediction set ---#

new_product$BestSellersRank<- NULL

# normalize

new_product_N <- predict(preprocessParams, new_product)
str(new_product_N)# 24obs. of  16 variables 

# dummify the data

dummies <- dummyVars(" ~ .", data = new_product_N)

new_product_r <- data.frame(predict(dummies, newdata = new_product_N))
str(new_product_r) # 24obs. of  27 variables 

######################
# Feature Engineering
######################

#--- Training and test set ---#

# check for collinearity. 
corrAll  <- cor(existing_product_r)

corrAll 
# plot correlation matrix
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")


# find IVs that are highly corrected (ideally >0.80)
corr <- cor(existing_product_r[,1:26])
# summarize the correlation matrix
corr
# create object with indexes of highly corr features
corrhigh <- findCorrelation(corr, cutoff=0.8)
# print indexes of highly correlated attributes
corrhigh
# get var name of high corr IV
colnames(existing_product_r[c(16)]) # "x3StarReviews"
colnames(existing_product_r[c(15)]) # "x4StarReviews"
colnames(existing_product_r[c(17)]) #"x2StarReviews"
colnames(existing_product_r[c(20)]) #"NegativeServiceReview"
colnames(existing_product_r[c(3)]) #"ProductTypeExtendedWarranty"

#################
# Feature removal
#################

# remove based on Feature Engineering (FE)
# create 22v ds
existing_product22v<- existing_product_r
existing_product22v$x3StarReviews <- NULL
existing_product22v$x4StarReviews<- NULL
existing_product22v$x2StarReviews <- NULL
existing_product22v$NegativeServiceReview<- NULL
existing_product22v$ProductTypeExtendedWarranty<- NULL
str(existing_product22v)   # 80obs. of  22 variables 

#--- Prediction set ---#

# create 22v ds
new_product22v<- new_product_r
new_product22v$x3StarReviews <- NULL
new_product22v$x4StarReviews<- NULL
new_product22v$x2StarReviews <- NULL
new_product22v$NegativeServiceReview<- NULL
new_product22v$ProductTypeExtendedWarranty<- NULL
str(new_product22v)   # 24obs. of  22 variables 

##################
# Train/test sets
##################

# set random seed
set.seed(123) 
# create the training partition that is 75% of total obs
inTraining <- createDataPartition(existing_product22v$Volume, p=0.75, list=FALSE)
# create training/testing dataset
trainSet <- existing_product22v[inTraining,]   
testSet <- existing_product22v[-inTraining,]   
# verify number of obs 
nrow(trainSet)
nrow(testSet)

str(testSet)
################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

##############
# Train model
##############

## ------- RF ------- ##

#RF train/fit
set.seed(123)
rf_Fit <- train(Volume~., data = trainSet, method = "rf", trControl=fitControl, tuneLength=5,importance=T)

#training results
rf_Fit

#Random Forest 

#61 samples
#21 predictors

#No pre-processing
#Resampling: Cross-Validated (3 fold, repeated 10 times) 
#Summary of sample sizes: 41, 41, 40, 41, 41, 40, ...  
#Resampling results across tuning parameters:
  
# mtry    RMSE       Rsquared   MAE      
#  2    282.67138  0.9242679  219.15154
#  6    162.81051  0.9700814  113.20589
#  11   113.58098  0.9820070   72.16202
#  16   90.81423   0.9868531   53.58808
#  21   75.83534   0.9898314   43.41319

# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 21.

# eval pred vars
predictors(rf_Fit)

#[1] "ProductTypeAccessories"     "ProductTypeDisplay"         "ProductTypeGameConsole"    
#[4] "ProductTypeLaptop"          "ProductTypeNetbook"         "ProductTypePC"             
#[7] "ProductTypePrinter"         "ProductTypePrinterSupplies" "ProductTypeSmartphone"     
#[10] "ProductTypeSoftware"        "ProductTypeTablet"          "Price"                     
#[13] "x5StarReviews"              "x1StarReviews"              "PositiveServiceReview"     
#[16] "Recommendproduct"           "ShippingWeight"             "ProductDepth"              
#[19] "ProductWidth"               "ProductHeight"              "ProfitMargin" 


# eval variable imp
varImp(rf_Fit)

#only 20 most important variables shown (out of 21)

#                            Overall
#x5StarReviews              100.00000
#PositiveServiceReview       13.04276
#Recommendproduct             9.27186
#ShippingWeight               8.75551
#ProfitMargin                 8.24510
#ProductDepth                 6.91196
#x1StarReviews                6.69604
#ProductHeight                6.02111
#ProductTypeNetbook           5.40260
#ProductWidth                 4.79898
#ProductTypePC                3.73293
#ProductTypeGameConsole       3.73293
#ProductTypeAccessories       3.23165
#ProductTypeSmartphone        2.76348
#ProductTypePrinter           2.09579
#ProductTypeLaptop            2.06325
#ProductTypePrinterSupplies   2.06325
#ProductTypeSoftware          1.15937
#ProductTypeTablet            0.43928
#Price                        0.01455

#--- Save top performing model ---#

saveRDS(rf_Fit, "rfFit.rds")  
# load and name model
rf_Fit <- readRDS("rfFit.rds")

#################
# Predict testSet
#################

# predict with RF
rfPred1 <- predict(rf_Fit, testSet)
# print predictions
rfPred1

#2           3           7           9          10          19          26          30          35 
#7.196267   12.192000   41.601467   62.011467   39.301333   59.246133 1303.095200   17.998667 1233.227067 
#39          44          50          52          54          62          65          67          73 
#1232.579067  363.153600 1882.286000  202.051600  356.372400   24.862667   86.171200  993.969600 1807.817200 
#79 
#1402.309333 

summary(rfPred1)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#61.69  188.54  296.50  567.30 1018.75 1400.42 
#performace measurment
postResample(rfPred1, testSet$Volume)

#RMSE             Rsquared          MAE 
#2452.4334053    0.5782275  788.4605965 

#plot predicted verses actual
plot(rfPred1,testSet$Volume)


## ------- SVM ------- ##

# SVM train/fit
set.seed(123)
svm_Fit<- train(Volume~., data=trainSet, method="svmLinear", trControl=fitControl, tuneLength=5)
svm_Fit

#Support Vector Machines with Linear Kernel 

#61 samples
#21 predictors

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 56, 55, 54, 54, 54, 55, ... 
#Resampling results:
  
#  RMSE      Rsquared   MAE     
#211.2421  0.9235051  129.9171

#Tuning parameter 'C' was held constant at a value of 1


# eval pred vars
predictors(svm_Fit)
#[1] "ProductTypeAccessories"     "ProductTypeDisplay"         "ProductTypeGameConsole"    
#[4] "ProductTypeLaptop"          "ProductTypeNetbook"         "ProductTypePC"             
#[7] "ProductTypePrinter"         "ProductTypePrinterSupplies" "ProductTypeSmartphone"     
#[10] "ProductTypeSoftware"        "ProductTypeTablet"          "Price"                     
#[13] "x5StarReviews"              "x1StarReviews"              "PositiveServiceReview"     
#[16] "Recommendproduct"           "ShippingWeight"             "ProductDepth"              
#[19] "ProductWidth"               "ProductHeight"              "ProfitMargin"   

# eval variable imp
varImp(svm_Fit)

#only  (out of 21)

#                            Overall
#x5StarReviews              100.0000
#PositiveServiceReview       93.3798
#x1StarReviews               23.7138
#ShippingWeight              16.4410
#ProfitMargin                15.0138
#ProductHeight               12.0673
#Price                       10.8462
#ProductWidth                10.2554
#Recommendproduct             7.6845
#ProductTypePrinter           6.1761
#ProductTypeGameConsole       5.6436
#ProductTypeAccessories       5.0857
#ProductDepth                 4.6692
#ProductTypePrinterSupplies   1.7122
#ProductTypeSmartphone        1.6288
#ProductTypeNetbook           1.4036
#ProductTypePC                1.3871
#ProductTypeLaptop            1.0313
#ProductTypeSoftware          0.7356
#ProductTypeDisplay           0.2850


#--- Save top performing model ---#

saveRDS(svm_Fit, "svmFit.rds")  
# load and name model
svm_Fit<- readRDS("svmFit.rds")
#################
# Predict testSet
#################
# predict with svm
svmPred1 <- predict(svm_Fit, testSet)
# print predictions
svmPred1
#2             3           7           9          10          19          26          30          35 
#169.46790    55.11483   125.24900   123.03796   109.96594   105.15201  1215.00262    59.06762  1205.33481 
#39           44          50          52          54          62          65          67          73 
#1206.19323   374.03120 10307.56585   233.95794   445.18639    51.25651    95.88288   787.11072  6431.64786 
#79 
#1376.50489
summary(svmPred1)
#Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#51.26   107.56   233.96  1288.25  1205.76 10307.57 
#performace measurment
postResample(svmPred1, testSet$Volume)
# RMSE          Rsquared         MAE 
# 254.3886541   0.9998267 120.5268534 
#plot predicted verses actual
plot(svmPred1,testSet$Volume)

## ------- GBM ------- ##

# GBM train/fit

set.seed(123)
gbm_Fit<- train(Volume~., data=trainSet, method="gbm", trControl=fitControl,tuneLength=3)
gbm_Fit

# Stochastic Gradient Boosting 

# 61 samples
# 21 predictors

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 10 times) 
# Summary of sample sizes: 56, 55, 54, 54, 54, 55, ... 
# Resampling results across tuning parameters:
  
# interaction.depth    n.trees  RMSE      Rsquared   MAE     
  # 1                   50      205.7707  0.9171101  145.2282
  # 1                  100      200.4396  0.9210300  142.6217
  # 1                  150      193.4172  0.9259025  140.3288
  # 2                   50      206.9113  0.9199814  145.0198
  # 2                  100      201.3038  0.9234845  143.3544
  # 2                  150      194.6846  0.9271210  140.2527
  # 3                   50      211.5007  0.9152220  147.9240
  # 3                  100      203.3550  0.9187436  144.9327
  # 3                  150      198.3149  0.9227247  144.1013

  # Tuning parameter 'shrinkage' was held constant at a value of 0.1
  # Tuning parameter 'n.minobsinnode' was held
  # constant at a value of 10
  # RMSE was used to select the optimal model using the smallest value.
  # The final values used for the model were n.trees = 150, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode
  # = 10.

# eval pred vars
predictors(gbm_Fit)
# [1] "ProductTypeAccessories" "Price"                  "x5StarReviews"          "x1StarReviews"         
# [5] "PositiveServiceReview"  "Recommendproduct"       "ShippingWeight"         "ProductDepth"          
# [9] "ProductWidth"           "ProductHeight"          "ProfitMargin"     
# eval variable imp
varImp(gbm_Fit)
#gbm variable importance

#only 20 most important variables shown (out of 21)

#                            Overall
#x5StarReviews              100.0000
#PositiveServiceReview       11.8530
#ProductWidth                 4.1381
#Price                        4.0895
#ProductHeight                2.5000
#ProductDepth                 2.4326
#ShippingWeight               2.2413
#x1StarReviews                2.1540
#ProfitMargin                 1.6644
#Recommendproduct             0.8519
#ProductTypeAccessories       0.5298
#ProductTypeGameConsole       0.0000
#ProductTypeNetbook           0.0000
#ProductTypePrinter           0.0000
#ProductTypeSmartphone        0.0000
#ProductTypePC                0.0000
#ProductTypeLaptop            0.0000
#ProductTypeDisplay           0.0000
#ProductTypeTablet            0.0000
#ProductTypePrinterSupplies   0.0000
#--- Save top performing model ---#

saveRDS(gbm_Fit, "gbmFit.rds")  
# load and name model
gbm_Fit<- readRDS("gbmFit.rds")
#################
# Predict testSet
#################
# predict with GBM
gbmPred1 <- predict(gbm_Fit, testSet)
# print predictions
gbmPred1
# [1]   34.85357  -36.65330 -118.08364   57.22338  190.22338   51.30802 1437.92202  122.90395 1314.56073 1276.26554
# [11]  388.57260 1403.16707   93.10287  267.02741  112.14885  157.50489 1445.56650 1419.26835 1278.81952
summary(gbmPred1)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-118.08   75.16  190.22  573.46 1296.69 1445.57 

#performace measurment
postResample(gbmPred1, testSet$Volume)
#RMSE     Rsquared          MAE 
#2597.5618267    0.3499114  923.4390709 
#plot predicted verses actual
plot(gbmPred1,testSet$Volume)

##################
# Predict new data
##################

# predict with svm

svmPred2 <- predict(svm_Fit, new_product22v)

svmPred2

# 1            2            3            4            5            6            7            8            9 
# 431.1964812  269.0116458  306.2295371   28.7605692   -1.3088108   57.1611986 1228.1276074   70.2506345   -9.5573866 
# 10           11           12           13           14           15           16           17           18 
# 1172.6446032 3816.5962925  336.4004587  363.7962276  119.9882440  150.9100281 1877.2688071   28.2705361  137.1927492 
# 19           20           21           22           23           24 
# 72.8323359  128.0422153  398.8593685    0.5976585  -88.2543203 5950.8992571 

summary(svmPred2)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -88.25   50.06  144.05  701.91  406.94 5950.90 


# predict with rf

rfPred2 <- predict(rf_Fit, new_product22v)

rfPred2

#1          2          3          4          5          6          7          8          9         10 
#661.12253  427.64644  586.91991   77.69861  111.28845   66.82900 1168.82086  141.57491   62.08524 1179.52253 
#11         12         13         14         15         16         17         18         19         20 
#1342.17001  754.52508  931.48378  196.36135  356.98803 1359.16203  146.13675  235.40190  180.96076  251.25398 
#21         22         23         24 
#328.33268  269.01805  176.44476 1326.28714


summary(rfPred2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#62.09  168.87  298.68  514.08  798.76 1359.16 

###############
# Save datasets
###############
output <- new_product
output$Volume<- rfPred2
write.csv(output, file="C2.T3output.csv", row.names = TRUE)
