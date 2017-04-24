# Date last modified: 24/01/2017
# Reference for svm with caret
# 1. http://blog.revolutionanalytics.com/2015/10/the-5th-tribe-support-vector-machines-and-caret.html
# 2. https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
# Objectives:
#           a. Read and process data
#           b. Tune svm model through three passes using
#               random parameter selection (tuneLength) and
#                grid-search (tuneGrid)
#           c. Evaluate models
#           d. Compare models
#           d. Try Linear svm model
# Problem: Predict a biological response of molecules from 
#          their chemical properties 
# Kaggle: https://www.kaggle.com/c/bioresponse
# SVM help:
#  https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf

# Steps:

# 1.  Process data scaling, centering etc
# 2.  Randomly try a few kernels and parameters
# 2a. (Beginners) Consider the RBF kernel K(x, y) = exp (-gamma||x-y||^2)
# 3.  Use cross-validation to find the best parameter C and gamma
# 4.  Fine-tune parameters, if required
# 4.  Use the best parameter C and gamma to train the whole training set
# 5.  Test

## Some tips
# 1. SVM  requires  that  each  data  instance  is  represented
#     as  a  vector  of  real  numbers. Hence, if there are
#      categorical attributes, we  rst have to convert them into
#       numeric data. Use m numbers to represent an m category attribute.
#        Only one  of  the m numbers  is  one,  and  others  are  zero.

# 2. Scaling before applying SVM is very important. Of course we  have
#   to use same method to scale both training & testing data.

#### 1.Clear/release memory. Call libraries -------
rm(list=ls()); gc()

# 1.1 Call libraries
# Call libraries
library(kernlab)   # Functions used in caret, svm package
library(caret)     # For data partition, pca, confusionMatrix() & recall
library(dplyr)     # For data manipulation
library(pROC)      # For roc() and AUC
library(stringi)   # For %s+%
library(doMC)      # For parallel operations on Linux/Mac

options(scipen = 999)  # No expoential notation 
registerDoMC(cores = 3) # How many workers


##### 2. Read data, examine and process it----------
# set working directory
setwd("C:/Users/ashok/OneDrive/Documents/biological_response/")
#setwd("c:/bdata/svm")

# 2.1
# read training and test files
train <- read.csv("train.csv")
test<-read.csv("test.csv")

# 2.2 Observe data
dim(train)
dim(test)
names(train)
str(train)
head(train)

# 2.3 Is data balanced
table(train$Activity)   # Yes it is

# 2.4 Any NAs?
sum(is.na(train))
sum(is.na(test))


## 2.5 Bind train and test data 
#      for combined processing
# 2.5.1  Add missing 'target' field to test
test$Activity<-0       

# 2.6 Row-wise binding 
#     and set target ('Activity') to factor
t<-rbind(train,test)
#   Class variable 'Activity' is in column 1.
t[,1]<-as.factor(t[,1])

######### 3. Process data for pca. Values will also be centered and scaled---------
#  3.1 Exclude first or 'Activity column
pre<-preProcess(t[,-1],method =c("pca"))  # Model
ct<-data.frame(predict(pre,t[,-1]))       # Transformed data
dim(ct)                                   # How many columns now?

# 3.2 Split processed data back into train and test
train<- ct[1:nrow(train),]
test<-ct[-(1:nrow(train)),]
dim(train)
dim(test)

# 3.3 To train, add back the 'Activity' column
#      It will add as the last column in the dataset
train$Activity<-t[1:nrow(train),]$Activity
train$Activity<-as.factor(train$Activity)

# 3.4 Change level values from 0, 1 
#     0 and 1 are not acceptable in svm 
levels(train$Activity)[1] <-"NoResponse"
levels(train$Activity)[2] <-"YesResponse"


##### 4. Remove objects not needed and release memory--------
rm(t) ; gc();

##### 5. Partition train data into training and validation sets -------- 
rownos<-createDataPartition(train$Activity,p=0.8,list=FALSE)
tr<-train[rownos,  ]      # training set
valid<-train[-rownos,]    # validation set

# 5.1 Check dimensions of all datasets
dim(train)  # pre-partition
dim(tr)     # After partition
dim(valid)  # Validation dataset

##### 6. SUPPORT VECTOR MACHINE MODEL-----
#### 6.1 svm First pass. Determine parameters for max ROC-----
set.seed(1492)
# 6.1.1 trainControl specifies only the type of resampling: i) cross-validation (
#         once or repeated), ii) bootstrap or iii) leave-one-out 
ctrl <- trainControl(method="repeatedcv",   #  Default 10 fold cross validation
                     number = 3,      # No of k-folds
                     repeats= 1 ,		  # How many times to repeat K-folds
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     verboseIter = TRUE
                     )

system.time(
# 6.1.2 Train and Tune the SVM on train data
#       Performance metric: 'ROC'. 'Kappa' is also useful when % of samples
#         is low in one class  
svm.tune_pass1 <- train(x=tr[,-588],
                  y= tr$Activity,
                  method = "svmRadial",  # Radial (non-linear) kernel
                  tuneLength = 5,		# Pick and test parameter values randomly
                  metric="ROC",
                  trControl=ctrl,
                  verbose = TRUE
                  )
           )
# 6.1.3 Examine model
svm.tune_pass1
# 6.1.4 Note below what C and sigma were tested
#       For C = 1 it is the best
svm.tune_pass1$results 
# 6.1.5 So final results
svm.tune_pass1$finalModel


#### 7. svm Second pass ----
# Based on model parameteres above, try to further fine tune parameters
# Look at the results of svm.tune and refine the parameter space
set.seed(1492)
# 7.1 Use expand.grid() to specify search space	on train data. grid()
#      is a two-column data-frame with various permutations/combinations
#         sigma        C
#         0.0012      0.75
#         0.0012      0.9
#         0.0017      1.0
grid <- expand.grid(sigma = c(.0012, .0017, 0.0022),
                    C = c(0.75, 0.9, 1, 1.1, 1.25)
)

## 7.2 Train and Tune the SVM
svm.tune_IIndpass <- train(x=tr[,-588],
                  y= tr$Activity,
                  method = "svmRadial",
                  metric="ROC",     # Estimate of performance
                  tuneGrid = grid,  # tuneGrid dataframe replaces tuneLength
                  trControl=ctrl)

# 7.3 Examine output of svm.tune
svm.tune_IIndpass  # Final  sigma = 0.0017 and C = 1.1

# 7.3 Relationship between estimates of performance (here 'ROC')
#     and tuning parameters
plot(svm.tune_IIndpass)

###### 8. Evaluate final model


# 8.2 Make class predictions
class_predict <- predict(svm.tune_IIndpass,newdata = valid[,-588] )
class_predict

# 8.3 Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(class_predict, valid$Activity,
                positive = "YesResponse",
                dnn=c("predictions","actual"),  # Dimenson headings
                mode="prec_recall"              # Print precision and recall as well )
)             

## 8.4 Probability predictions
prob_predict <- predict(svm.tune_IIndpass,newdata = valid[,-588], type = "prob" )
head(prob_predict)

# 8.5. Draw ROC graph using predicted probabilities
df_roc<-roc(valid$Activity,prob_predict[,2])
plot(df_roc,main="AUC = " %s+% df_roc$auc)

########### 9 Full train now ####################---------
#### SVM third pass ----

# Based on model parameteres above, either try to further 
#   fine tune parameters or use the same
set.seed(1492)
# 9.1 Use the expand.grid to generate dataframe of search space	on train data
grid <- expand.grid(sigma = c(.0016, .0017, 0.0018),
                    C = c(1.05, 1.1, 1.15)
)

## 9.2 Train and Tune the SVM
svm.final.tune <- train(x=train[,-588],
                  y= train$Activity,
                  method = "svmRadial",
                  metric="ROC",
                  tuneGrid = grid,  # tuneGrid = dataframe of parameters
                  trControl=ctrl)

# 9.3 Examine output of svm.final.tune
svm.final.tune   # My final parameters: sigma = 0.0018, C = 1.15

# 9.4 Which variables were more important
varImp(svm.final.tune)   # varImp() is a caret function

########### 10. Make predictions for test ##------------
# 10.1 MAke predictions for test
# The test dataset after pca/pre-processing
dim(test)
names(test)
head(test)

# 10.2 Make class predictions for test data
prob_predict <- predict(svm.final.tune,newdata = test, type= 'prob')
head(prob_predict)

# 10.3 Write results to file now
df<-data.frame(Moleculeid=1:nrow(test))   # 1st column is id
head(df)

# 10.4 Write probabilities
df$PredictedProbability<-prob_predict[,2]    # IInd col is probability
head(df)
# 10.5 Write to a csv file. %s+% is concatenation symbol from stringi package
write.csv(df,file="result_svm" %s+% Sys.Date() %s+% ".csv", quote = FALSE, row.names = FALSE)

############ 11. Comparing models ------------
# 11.1 We have built three models. We could have built model using gbm
#  and other techniques. We can compare models here
#   Visualization of Resampling Results
visualize<-resamples(list(firstPass=svm.tune_pass1, 
                        SecondPass = svm.tune_IIndpass,
                        thirdPass = svm.final.tune
                        )
                   )
# 11.2 Summary comparisons
summary(visualize)

# 11.3 Box plot of results. Layout is one column, three rows
bwplot(visualize,layout=c(1,3))
# 11.4 Dot plot of results
dotplot(visualize,metric = "ROC")


######## 12. OR Use parameters to build model directly with kernlab ------------
## Alternatively, one can use parameters decided by caret into 'kernlab' svm
#   model building, as:
# 12.1 Model first. Assuming sigma is 0.0017 and C is 1.1
k_model <- ksvm(Activity ~ .,data=tr, kernel="rbfdot", kpar=list(sigma=0.0017),C=1.1)
k_model
# 12.2 PRedict using model
direct_predict <- predict(k_model, newdata = valid[,-588] )
# 12.3 Evaluate
confusionMatrix(direct_predict, valid$Activity,
                                   positive = "YesResponse",
                                   dnn=c("predictions","actual"),  # Dimenson headings
                                   mode="prec_recall"              # Print precision and recall as well )
                )   
################### FINISH #########-----------


####### 13. Linear Kernel ###########--------

set.seed(1492)                     
# 13.1 Train and Tune Linear SVM
svm.tune2 <- train(x=tr[,-588],
                   y= tr$Activity,
                   method = "svmLinear",
                   metric="ROC",
                   trControl=ctrl)	

# 13.2 Examine model
svm.tune2

#### 14. Evaluate Linear kernel model
# 14.1 Class predictions
class_predict <- predict(svm.tune2,newdata = valid[,-588] )
class_predict

# 14.2 Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(class_predict, valid$Activity,
                positive = "YesResponse",
                dnn=c("predictions","actual"),  # Dimenson headings
                mode="prec_recall"              # Print precision and recall as well )
)             

# 14.3 Probability values
prob_predict <- predict(svm.tune2,newdata = valid[,-588], type = "prob" )
head(prob_predict)

# 14.4 Draw ROC graph using predicted probabilities
df_roc<-roc(valid$Activity,prob_predict[,2])
plot(df_roc,main="AUC = " %s+% df_roc$auc)

############# FINISH Linear kernel ###############

############## 15. About SVM ################################
# What are cost and gamma in svm
# See an excellent answer here: https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine
#   Small C makes the cost of misclassificaiton low ("soft margin"),
#     thus allowing more of them for the sake of wider "cushion".
#   Large C makes the cost of misclassification high ('hard margin"),
#     thus forcing the algorithm to explain the input data stricter
#        and potentially overfit. The goal is to find the balance
#         between "not too strict" and "not too loose". Cross-validation
#          and resampling, along with grid search, are good ways to finding 
#            the best C.
# Gamma is the parameter of a Gaussian Kernel (to handle non-linear classification).
#   When points are not linearly separable in 2D so you want to transform them to a
#     higher dimension where they will be linearly sepparable. Imagine "raising" the central
#       points, then you can sepparate them from the surrounding points with a plane
#        (hyperplane). To "raise" the points you use the RBF kernel, gamma controls the shape
#         of the "peaks" where you raise the points. A small gamma gives you a pointed bump
#          in the higher dimensions, a large gamma gives you a softer, broader bump.
#  So a small gamma will give you low bias and high variance while a large gamma will give you
#     higher bias and low variance.


#### 16. Tips on practical use of svm----------
# =============================
# 1. Note that SVMs may be very sensible to the proper choice 
#   of parameters, so allways check a range of parameter
#    combinations, at least on a reasonable subset of your data.
#     For classifcation tasks, you will most likely use
#      C-classifcation with the RBF kernel (default), because of its
#       good general performance and the few number of parameters. 
# 2. Authors suggest to try small and large values for C like
#  1 to 1000|first, then to decide which are better for the data
#   by cross validation, and fnally to try several gammas for the
#    better C's. However, better results are obtained by using
#     a grid search over all parameters. For this, we recommend to
#      use the tune.svm() function in e1071.
# 3. Be careful with large datasets as training times may increase
#  rather fast. Scaling of the data usually drastically improves
#   the results. Therefore, svm() scales the data by default.
########################################################################

