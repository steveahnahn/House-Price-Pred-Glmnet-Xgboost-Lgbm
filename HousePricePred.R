#Predict Real Estate House Price

#Load Data
training_set <- read.csv("train.csv", stringsAsFactors = F)
test_set <- read.csv("test.csv", stringsAsFactors = F)

#Load libraries
library(glmnet)    #glmnet
library(ggplot2)   #visualize
library(caret)     #ml
library(e1071)     #statspack
library(vtreat)    #1-hot encode
library(Matrix)
library(dplyr)     #data wrangle
library(xgboost)   #xgboost
library(mice)      #na impute
library(Amelia)    #missmap fx
library(psych)     #statspack
library(corrplot)  #corr
library(reshape2)
library(timeDate)
library(data.table)
library(Metrics)
library(caTools)

#dim
cat('We have a ', dim(training_set)[1],' by ', dim(training_set)[2],' dataset')

#data struct
head(training_set)
str(training_set, list.len = 10)

#Drop ID features from both training & testing
train_ID <- training_set$Id
test_ID <- test_set$Id
training_set$Id <- NULL
test_set$Id <- NULL

#dim check
cat("Training set dimensions: ", dim(training_set)[1], ' by ', dim(training_set)[2], '\n')
cat("Test set dimensions: ", dim(test_set)[1], ' by ', dim(test_set)[2])

#Visualize target var
describe(training_set$SalePrice)
quantile(training_set$SalePrice)
ggplot(training_set, aes(x = SalePrice)) + geom_density(fill = 'dodgerblue') + ggtitle("Distribution of Sale Price") + labs(subtitle = ('Skewed Right'))
qqnorm(training_set$SalePrice)


# Noticing that the target variable, Sale Price does not follow a normal distribution
#a simple log transformation might take care of it. If not, a BoxCox transformation later on with our full model will take care of it possibly.

#log transformation
training_set$SalePrice <- log(training_set$SalePrice)

describe(training_set$SalePrice)
quantile(training_set$SalePrice)
ggplot(training_set, aes(x = SalePrice)) + geom_density(fill = 'dodgerblue') + ggtitle("Distribution of log(SalePrice") + labs(subtitle = 'Approx Normal Dist')
qqnorm(training_set$SalePrice)

# Visualize some plots and ggplots
#Overall Quality ~ Log of Sale Price
ggplot(training_set, aes(x = OverallQual,  y = SalePrice)) + geom_jitter()+ geom_smooth(method = "glm")+
ggtitle("Overall Quality - Log of Sale Price") + xlab("Overall Quality") + ylab("Log of Sale Price") + theme_minimal()

#Year Built ~ Log of Sale Price
ggplot(training_set, aes(x = YearBuilt, y = SalePrice)) + geom_jitter() +
geom_smooth(method = "loess") + ggtitle("YearBuilt - Log of Sale Price") +
xlab("Year Built") + ylab("Log of Sale Price") + theme_minimal()

#Neighborhood ~ Log of Sale Price
ggplot(training_set, aes(x = as.factor(Neighborhood), y = SalePrice)) +
geom_boxplot(position = "dodge", outlier.color = 'dodgerblue' ) +
theme_minimal()+
theme(axis.text.x = element_text(angle = 45,
size = 9, hjust = 1)) +
ggtitle("Log Sale Price - Neighborhood Boxplot") + xlab("Neighborhood") + ylab("Log of Sale Price")



# Lets join our training_set and test_set to handle all of our features and NA's all together.

#drop target variable saleprice to rbind to test_set for feature cleaning
ntrain <- dim(training_set)
ntest <- dim(test_set)
training_target <- training_set$SalePrice
train_set_feat <- training_set[,-80]

all_data <- rbind(train_set_feat, test_set)

cat("Combined data dimensions are ", dim(all_data)[1], ' by ', dim(all_data)[2])

##Correlation

#Make a correlation plot of the numeric variables
find_num_var <- array(dim = dim(training_set)[2])
for(i in 1:dim(training_set)[2]){
    find_num_var[i] <- is.numeric(training_set[,i])
}

#colnames for numeric variables of training_set
corplot_var <- colnames(training_set)[find_num_var][1:dim(training_set)[2] -1]
#drop NA's
corplot_droppedna <- na.omit(training_set[, (names(training_set) %in% corplot_var)])

#Correlation Matrix
cormat <- round(cor(corplot_droppedna),2)
get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
}
upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(cormat)

ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
geom_tile(color = "white")+
scale_fill_gradient2(low = "red", high = "blue", mid = "white",
midpoint = 0, limit = c(-1,1), space = "Lab",
name="Pearson\nCorrelation") +
theme_minimal()+
theme(axis.text.x = element_text(angle = 90, vjust = 1,
size = 9, hjust = 1))+
coord_fixed()

We can see that (GarageArea - GarageCars) and (X1stFlrSF - TotalBsmtSF)  create a highly correlated 2x2 matrix. We will deal with these features later on.


missmap(all_data, col = c('dodgerblue', 'black'), main = 'Missingness Map', y.cex = 0.3, x.cex = 0.7, legend= FALSE)


#concrete numbers
missing_data_col <- sapply(all_data, function(x) { sum(is.na(x)) })
missing_data_col <- sort(missing_data_col[missing_data_col > 0], decreasing = TRUE)
missing_data_col


#create copy for clean ver.
alldata_c1 <-all_data

#Change Numeric var that are really Categorical
char_var <- c("MSSubClass", "OverallQual")
alldata_c1[char_var] <- lapply(alldata_c1[char_var], as.character.default)


#Change NA's to "None" for categorical variables
#Grab all categorical variables
categor_var <- colnames(all_data[,which(sapply(all_data, class) == "character")])


None <- function(dataset, var){
    levels(dataset[,var]) <- c(levels(dataset[,var]), "None")
    dataset[,var][is.na(dataset[,var])] <- "None"
    return(dataset[,var])
}

#Impute NA's into Nones for Categorical variables
for (i in 1:length(categor_var)){
    alldata_c1[, categor_var[i]] <- None(all_data, categor_var[i])
}



missmap(alldata_c1, col = c('dodgerblue', 'black'), main = 'Altered Missingness Map', y.cex = 0.3, x.cex = 0.7, legend= FALSE)

#Check the missing data after changing NA's to Nones for categoricals
missing_data_after <- sapply(alldata_c1, function(x) { sum(is.na(x)) })
missing_data_after <- sort(missing_data_after[missing_data_after > 0], decreasing = TRUE)
missing_data_after


# Feature Engineering and Cleaning
# First we will deal with the highly correlated variables to
#remove dependency and we can feature engineer by combining components

#high corr
alldata_c1$GarageCars <- NULL
alldata_c1$CentralAir <- NULL
alldata_c1$MoSold <- NULL
alldata_c1$GarageYrBlt <- NULL
alldata_c1$OverallCond <- NULL

#Total Square Footage
alldata_c1$TotalSF <- alldata_c1$TotalBsmtSF + alldata_c1$X1stFlrSF + alldata_c1$X2ndFlrSF
alldata_c1$TotalBsmtSF <- NULL
alldata_c1$X1stFlrSF <- NULL
alldata_c1$X2ndFlrSF <- NULL

#Age of Home
alldata_c1$Age <- alldata_c1$YrSold - alldata_c1$YearRemodAdd
alldata_c1$YrSold <- NULL
alldata_c1$YearRemodAdd <- NULL

#Bathroom Count
alldata_c1$BathrmCnt <- alldata_c1$FullBath + alldata_c1$HalfBath + alldata_c1$BsmtFullBath + alldata_c1$BsmtHalfBath
alldata_c1$FullBath <- NULL
alldata_c1$HalfBath <- NULL
alldata_c1$BsmtFullBath <- NULL
alldata_c1$BsmtHalfBath <- NULL

#Porch SF
alldata_c1$TotalPorchSF <- alldata_c1$OpenPorchSF + alldata_c1$EnclosedPorch + alldata_c1$X3SsnPorch + alldata_c1$ScreenPorch
alldata_c1$OpenPorchSF <- NULL
alldata_c1$EnclosedPorch <- NULL
alldata_c1$X3SsnPorch <- NULL
alldata_c1$ScreenPorch <- NULL


#Save numerical variables for skewness check later
num_var<- alldata_c1[sapply(alldata_c1, is.numeric)]
num_var <- names(num_var)


#Deal with some typos
alldata_c1 <- as.data.table(alldata_c1)

alldata_c1[MSSubClass  == 150, MSSubClass:= "160"]
alldata_c1[Exterior2nd  == "Brk Cmn", Exterior2nd:= "BrkComm"] #typo
alldata_c1[Exterior2nd  == "Wd Shng", Exterior2nd:= "WdShing"] #typo
alldata_c1[RoofMatl  == "Tar&Grv", RoofMatl:= "TarGrv"] #typo '&'
alldata_c1[RoofMatl  == "WdShngl", RoofMatl:= "WdShing"] #typo 'l'

alldata_c1[Exterior1st  == "Wd Sdng", Exterior1st:= "WdSdng"] #spaces
alldata_c1[Exterior2nd  == "Wd Sdng", Exterior2nd:= "WdSdng"] #spaces

alldata_c1 <- as.data.frame(alldata_c1)


#Change all categoricals to factors to prepare for label encoding

#Label Encoding variables that are ordered variables coverted into integers
categor_var2 <- c("FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond",
"ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual", "BsmtFinType1",
"BsmtFinType2", "Functional", "Fence", "BsmtExposure", "GarageFinish",
"LandSlope", "LotShape", "PavedDrive", "Street", "Alley", "MSSubClass")


alldata_c1[categor_var2] <- lapply(alldata_c1[,categor_var2], factor)
alldata_c1[categor_var2] <- lapply(alldata_c1[,categor_var2], as.integer)

#Factor the rest for DesigntreatmentZ
categor_var3 <- colnames(alldata_c1[,which(lapply(alldata_c1, class) == "character")])

alldata_c1[categor_var3] <- lapply(alldata_c1[,categor_var3], factor)

#Impute zeros into remaining NA's for numeric variables
for (col in c("LotFrontage", "GarageArea", "BsmtFinSF1",
"BsmtFinSF2", "BsmtUnfSF", "TotalSF", "MasVnrArea","BathrmCnt")) {
    alldata_c1[is.na(alldata_c1[, col]), col] = 0
}


#final check
missmap(alldata_c1, col = c('dodgerblue', 'black'), main = 'Final Missingness Map', y.cex = 0.3, x.cex = 0.7, legend= FALSE)


#Check the missing data after changing NA's to Nones for categoricals
missing_data_final <- sapply(alldata_c1, function(x) { sum(is.na(x)) })
missing_data_final <- sort(missing_data_final[missing_data_final > 0], decreasing = TRUE)
missing_data_final


#extract features with high skewness and apply log transformation
as.data.frame(describe(alldata_c1[num_var])[,c('mean', 'skew', 'kurtosis')])
skewed_var <- sapply(alldata_c1[(num_var)], function(x){skewness(x, na.rm = TRUE)})
skewed_var <- skewed_var[skewed_var > .50]

alldata_c2 <- lapply(alldata_c1[,c("LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "LowQualFinSF", "GrLivArea", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "WoodDeckSF", "PoolArea", "MiscVal", "TotalSF", "TotalPorchSF")], function(x) {log1p(x)})



#replace old cols with new cols
alldata_c1$LotArea <- alldata_c2$LotArea
alldata_c1$MasVnrArea <- alldata_c2$MasVnrArea
alldata_c1$BsmtFinSF1 <- alldata_c2$BsmtFinSF1
alldata_c1$BsmtFinSF2 <- alldata_c2$BsmtFinSF2
alldata_c1$BsmtUnfSF <- alldata_c2$BsmtUnfSF
alldata_c1$LowQualFinSF <- alldata_c2$LowQualFinSF
alldata_c1$GrLivArea <- alldata_c2$GrLivArea
alldata_c1$KitchenAbvGr <- alldata_c2$KitchenAbvGr
alldata_c1$TotRmsAbvGrd <- alldata_c2$TotRmsAbvGrd
alldata_c1$Fireplaces <- alldata_c2$Fireplaces
alldata_c1$WoodDeckSF <- alldata_c2$WoodDeckSF
alldata_c1$PoolArea <- alldata_c2$PoolArea
alldata_c1$MiscVal <- alldata_c2$MiscVal
alldata_c1$TotalSF <- alldata_c2$TotalSF
alldata_c1$TotalPorchSF <- alldata_c2$TotalPorchSF



#check of skewness change
as.data.frame(describe(alldata_c1[num_var])[,c('mean', 'sd', 'skew', 'kurtosis')])


## Scaling variables for ensemble
alldata_c1[num_var] <- scale(alldata_c1[num_var])


#Split data into training and test sets for one hot encoding
training <- alldata_c1[1:ntrain[1],]
testing  <- alldata_c1[1:ntest[1],]
# dim(training)
# dim(testing)

#grab features
features <- setdiff(names(alldata_c1), training_target)

treatplan <- designTreatmentsZ(training, minFraction = 0.01, rareCount = 0, features, verbose = FALSE)
train_treated <- prepare(treatplan, dframe = training, codeRestriction = c("clean", "lev"))
test_treated  <- prepare(treatplan, dframe = testing, codeRestriction = c("clean", "lev"))

# dim(train_treated)
# dim(test_treated)

#Add target variable back in and rename
train_treated <- cbind(train_treated, training_set$SalePrice)
colnames(train_treated)[colnames(train_treated)=="training_set$SalePrice"] <- "SalePrice"

#Training/Validation & Test from treated training set
set.seed(4242)

split <- sample.split(train_treated$SalePrice, SplitRatio = 0.8)
Trainingf <- subset(train_treated, split == TRUE)
Validationf <- subset(train_treated, split == FALSE)

dim(Trainingf)
dim(Validationf)
invisible(gc())



# Modeing

#Lasso - Regularized Regression
set.seed(123)

cv_lasso = cv.glmnet(as.matrix(Trainingf[, -153]), Trainingf[, 153])

preds <- predict(cv_lasso, newx = as.matrix(Validationf[, -153]), s = "lambda.min")
rmse(Validationf$SalePrice, preds)

#SVM
svm_model<-svm(SalePrice~., data=Trainingf, cost = 2)

svm_pred <- predict(svm_model, newdata = as.matrix(Validationf[,-153]))
rmse(Validationf$SalePrice, svm_pred)


#Prepare for Xgboost
model_train <- xgb.DMatrix(data = data.matrix(Trainingf), label = Trainingf[,153])
model_val <- xgb.DMatrix(data = data.matrix(Validationf), label = Validationf[,153])


#Testing
control <- trainControl(method = "cv",
number = 5,
savePredictions = "final",
allowParallel = TRUE)


grid = expand.grid(nrounds = c(1000,1200,1500),
max_depth = c(6,8,10),
eta = c(0.025, 0.01),
gamma = c(0.1),
colsample_bytree = c(1),
min_child_weight = c(1),
subsample = c(0.8))



grid1 = expand.grid(nrounds = 1000,
max_depth = seq(4,8,by = 1),
eta = 0.025,
gamma = 0.1,
colsample_bytree = 1.0,
subsample = 1.0,
min_child_weight = 4)

