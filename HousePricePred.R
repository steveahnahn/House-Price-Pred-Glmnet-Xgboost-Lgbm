training_set <- read.csv("train.csv", stringsAsFactors = F)
test_set <- read.csv("test.csv", stringsAsFactors = F)

library(glmnet)    #glmnet
library(ggplot2)   #visualization
library(dplyr)     #data wrangling
library(xgboost)   #xgboost 
library(mice)      #missing data imputation
library(Amelia)    #missmap fx
library(psych)     #descriptives
library(corrplot)  #correlation
library(reshape2)  #melt fx
library(caret)     #ML
library(timeDate)  #skewness

#Understanding the data
cat('We have a ', dim(training_set)[1],' by ', dim(training_set)[2],' dataset')

head(training_set)
str(training_set, list.len = 10)

#Drop ID features from both training & testing
train_ID <- training_set$Id
test_ID <- test_set$Id
training_set$Id <- NULL
test_set$Id <- NULL

cat("Training set dimensions: ", dim(training_set)[1], ' by ', dim(training_set)[2], '\n')
cat("Test set dimensions: ", dim(test_set)[1], ' by ', dim(test_set)[2])

#Checking dist of target variable
describe(training_set$SalePrice)
quantile(training_set$SalePrice)
ggplot(training_set, aes(x = SalePrice)) + geom_density(fill = 'dodgerblue') + ggtitle("Distribution of Sale Price") + labs(subtitle = ('Skewed Right'))
qqnorm(training_set$SalePrice)

#log transformation
training_set$SalePrice <- log(training_set$SalePrice)

#Recheck
describe(training_set$SalePrice)
ggplot(training_set, aes(x = SalePrice)) + geom_density(fill = 'dodgerblue') + ggtitle("Distribution of log(SalePrice") + labs(subtitle = 'Approx Normal Dist')
qqnorm(training_set$SalePrice)

#Overall Quality ~ Sale Price
ggplot(training_set, aes(x = OverallQual,  y = SalePrice)) + geom_jitter()+ geom_smooth(method = "glm")+
  ggtitle("Overall Quality - Log of Sale Price") + xlab("Overall Quality") + ylab("Log of Sale Price") + theme_minimal()


ggplot(training_set, aes(x = YearBuilt, y = SalePrice)) + geom_jitter() + 
  geom_smooth(method = "loess") + ggtitle("YearBuilt - Log of Sale Price") + 
  xlab("Year Built") + ylab("Log of Sale Price") + theme_minimal()



ggplot(training_set, aes(x = as.factor(Neighborhood), y = SalePrice)) +
  geom_boxplot(position = "dodge", outlier.color = 'dodgerblue' ) + 
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, 
                                   size = 9, hjust = 1)) +
  ggtitle("Log Sale Price - Neighborhood Boxplot") + xlab("Neighborhood") + ylab("Log of Sale Price")


#drop target variable saleprice to rbind to test_set for feature cleaning
ntrain <- dim(training_set)
ntest <- dim(test_set)
training_target <- training_set$SalePrice
train_set_feat <- training_set[,-80]

all_data <- rbind(train_set_feat, test_set)

cat("Combined data dimensions are ", dim(all_data)[1], ' by ', dim(all_data)[2])


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

missmap(all_data, col = c('dodgerblue', 'black'), main = 'Missingness Map', y.cex = 0.3, x.cex = 0.7, legend= FALSE)

missing_data_col <- sapply(all_data, function(x) { sum(is.na(x)) })
missing_data_col <- sort(missing_data_col[missing_data_col > 0], decreasing = TRUE)
missing_data_col

#Change NA's to "None" for categorical variables
NA_to_None_vec <- c("Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType","GarageFinish","GarageQual" ,"GarageCond","PoolQC","Fence","MiscFeature", "MSZoning", "Utilities", "Functional", "Electrical", "Exterior1st", "Exterior2nd", "KitchenQual", "SaleType", "GarageYrBlt" )

None <- function(dataset, var){
  levels(dataset[,var]) <- c(levels(dataset[,var]), "None")
  dataset[,var][is.na(dataset[,var])] <- "None"
  return(dataset[,var])
}
alldata_c1 <-all_data
for (i in 1:length(NA_to_None_vec)){
  alldata_c1[,NA_to_None_vec[i]] <- None(all_data,NA_to_None_vec[i]) 
}


missmap(alldata_c1, col = c('dodgerblue', 'black'), main = 'Altered Missingness Map', y.cex = 0.3, x.cex = 0.7, legend= FALSE)

#Check the missing data after changing NA's to Nones for categoricals
missing_data_after <- sapply(alldata_c1, function(x) { sum(is.na(x)) })
missing_data_after <- sort(missing_data_after[missing_data_after > 0], decreasing = TRUE)
missing_data_after


# Feature Engineering and Cleaning
# First we will deal with the highly correlated variables GarageArea and GarageCars to remove dependency by simply removing one. Next we can create a TotalSF by adding the separated components that are highly correlated as well. 

alldata_c1$GarageCars <- NULL #high corr

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

#Changing numeric variables to that are really categoricals
char_var <- c("MSSubClass", "OverallQual", "OverallCond", "MoSold", "GarageYrBlt")
alldata_c1[char_var] <- lapply(alldata_c1[char_var], as.character.default)

alldata_c1 <- as.data.frame(alldata_c1)

#Factorize categorical variables that are ordinal
factors <- c("BsmtFinType1", "BsmtFinType2", "Functional", "Fence", "BsmtExposure", "GarageFinish", "LandSlope", "LotShape", "PavedDrive", "Street", "Alley", "CentralAir", "MSSubClass","OverallQual", "OverallCond", "MoSold", "FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual", "GarageYrBlt")

alldata_c1[factors] <- lapply(alldata_c1[factors], factor)

#Check the classes of your variables
#sapply(all_data_cleaned, class)

cat('Our new dataset is ', dim(alldata_c1)[1], ' by ', dim(alldata_c1)[2])

#Impute zeros into remaining NA's for numeric variables
for (col in c("LotFrontage", "GarageArea", "BsmtFinSF1", 
              "BsmtFinSF2", "BsmtUnfSF", "TotalSF", "MasVnrArea", "MasVnrType","BathrmCnt")) {
  alldata_c1[is.na(alldata_c1[, col]), col] = 0
}

#final check
missmap(alldata_c1, col = c('dodgerblue', 'black'), main = 'Final Missingness Map', y.cex = 0.3, x.cex = 0.7, legend= FALSE)

#grab the names of numeric features 
num_var<- alldata_c1[sapply(alldata_c1, is.numeric)]
num_var <- names(num_var)

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

alldata_c1[num_var] <- scale(alldata_c1[num_var])

#Split data into training and test sets
training <- alldata_c1[1:ntrain[1],]
testing  <- alldata_c1[1:ntest[1],]
dim(training)
dim(testing)


#Add target variable back in and rename
training <- cbind(training, training_set$SalePrice)
colnames(training)[colnames(training)=="training_set$SalePrice"] <- "SalePrice"

set.seed(4242)
inTrain <- createDataPartition(y = training$SalePrice, p = 0.7, list = FALSE)
Trainingf <- training[inTrain, ]
Validationf <- training[-inTrain, ]

dim(Trainingf)
dim(Validationf)
invisible(gc())

#Prepare for Modeling


