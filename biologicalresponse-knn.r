# Reference for knn and caret: 
#  http://rstudio-pubs-static.s3.amazonaws.com/16444_caf85a306d564eb490eebdbaf0072df2.html
# Objectives:
#           a. Read and process data
#           b. Use knn
#           c. Parameter tuning with caret
# Problem: Predict a biological response of molecules from 
#          their chemical properties 
# Kaggle: https://www.kaggle.com/c/bioresponse

## 1.

# 1.1
# Clear/release memory
rm(list=ls()); gc()

# 1.2 Call libraries
# Call libraries
library(class)     # for knn()
library(caret)     # For data partition, pca, confusionMatrix() and train() 
library(pROC)      # For roc(), auc
library(stringi)   # For %s+%

options(scipen = 999)  # No expoential notation 

## 2
# Read data, examine and process it
# set working directory
setwd("C:/Users/ashok/OneDrive/Documents/biological_response/")
#setwd("c:/bdata/svm")

# 2.1
# read training and test files
train <- read.csv("train.csv")
test<-read.csv("test.csv")

# 2,2 Observe data
dim(train)
dim(test)
names(train)
str(train)
head(train)

# 2.3 Is data balanced
table(train$Activity)   # Yes it is


## 3. Bind train and test data 
#      for combined processing
# 3.1 Add missing 'target' field to test
test$Activity<-0       

# 3.2 Row-wise binding 
#     and set target ('Activity') to factor
t<-rbind(train,test)
#   Class variable 'Activity' is in column 1.
t[,1]<-as.factor(t[,1])


## 4.
# Process data for pca. Values will also be centered and scaled
#  Exclude first or 'Activity column
pre<-preProcess(t[,-1],method =c("pca"))  # Model
ct<-data.frame(predict(pre,t[,-1]))       # Transformed data
dim(ct)                                   # How many columns now?

## 5.
# 5.1 Split processed data back into train and test
train<- ct[1:nrow(train),]
test<-ct[-(1:nrow(train)),]
dim(train)
dim(test)

# 5.2 To train, add back the 'Activity' column
#      It will add as the last column in the dataset
train$Activity<-t[1:nrow(train),]$Activity
train$Activity<-as.factor(train$Activity)

## 6. Remove objects not needed and release memory
rm(t) ; gc();

## 7
# Partition train data into training and validation sets
rownos<-createDataPartition(train$Activity,p=0.8,list=FALSE)
tr<-train[rownos,  ]      # training set
valid<-train[-rownos,]    # validation set

# 7.2 Check dimensions of all datasets
dim(train)
dim(tr)
dim(valid)

#
set.seed(400)
# repeats = 1 takes 588 seconds
# repeats = 3 takes 1900 seconds
ctrl <- trainControl(
                    method="repeatedcv",
                    number = 10,  # The number of folds
                    repeats = 1   # For repeated k-fold, the number of completed k-folds
                    ) 

system.time(
           knnFit <- train(Activity ~ .,
                           data = tr,
                           method = "knn",
                           trControl = ctrl,
                           tuneLength = 20  # Number of 'k's of various values that will be tried
                           )
)

#Output of kNN fit
knnFit
# Accuracy vs Neighbours
plot(knnFit)

class_knnPredict <- predict(knnFit,newdata = valid[,-588] )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(class_knnPredict, valid$Activity,
                positive = "1",
                dnn=c("predictions","actual"),  # Dimenson headings
                mode="prec_recall"              # Print precision and recall as well )
                )             


prob_knnPredict <- predict(knnFit,newdata = valid[,-588], type = "prob" )
head(prob_knnPredict)

# 10. Draw ROC graph using predicted probabilities
df_roc<-roc(valid$Activity,prob_knnPredict[,2])
plot(df_roc,main="AUC = " %s+% df_roc$auc)

############## knn ENDS #############################

