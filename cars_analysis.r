# The goal is to build a model that both predicts futre car sales prices and
# to see how much each independent variable is affecting the price/find what variables that affect the price the most. 
# A kombination of prediction and inference.
# Regression problem

# Import libraries
library("readxl")
library(ggplot2)
library(caret)
library(dplyr)
library(corrplot)
library(leaps)
library(pxweb)
library(jsonlite)
library(httr)
library(glmnet)
library(car)


# Data about number of cars in traffic in Sweden from SCB

# For interactive API use the following code. 
#SCBdata <- pxweb_interactive("https://api.scb.se/OV0104/v1/doris/sv/ssd/START/TK/TK1001/TK1001A/PersBilarA")
#df_SCBdata <- SCBdata$data
#head(df_SCBdata)


SCBdata <-
  pxweb_get(url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/TK/TK1001/TK1001A/PersBilarA",
            query = "query.json")
df_SCBdata <- as.data.frame(SCBdata, column.name.type = "text", variable.value.type = "text")
head(df_SCBdata)
str(df_SCBdata)
df_SCBdata$år <- as.numeric(as.character(df_SCBdata$år))

ggplot(df_SCBdata, aes(x = år, y = Antal)) +
  geom_line() +
  labs(title = "Number of cars in Sweden, 2002-2024",
       x = "Year",
       y = "Number of cars") +
  theme_minimal()

# The number of cars are increasing every year. 
# An automatic way for someone who wants to buy a new car to predict and understand car prices 
# can be helpful and a service that many can benefit from. 


# Excel file with data about cars for sale on Blocket.se
file <- "cars_data.xlsx"
bildata <- read_excel(file)


# EDA / DATA PRE-PROCESSING
dim(bildata)
head(bildata)
str(bildata)
summary(bildata)

bildata$bränsle <- as.factor(bildata$bränsle)
bildata$växellåda <- as.factor(bildata$växellåda)
bildata$biltyp <- as.factor(bildata$biltyp)
bildata$drivning <- as.factor(bildata$drivning)
bildata$märke <- as.factor(bildata$märke)
str(bildata)

# 0-values
colSums(bildata[sapply(bildata, is.numeric)] == 0)
# All cars with 0-values for miltal are from recent years, so 0-values seems legit.
# If log-transformation needs to be done, add a small number to avoid 0-values.


# NAN-values
colSums(is.na(bildata))
# NAN-values are handled after split into train and test data.


# Split the data into train(80%) and test(20%) sets.
set.seed(123)

trainIndex <- createDataPartition(bildata$pris, p = 0.8, list = FALSE)
trainData <- bildata[trainIndex, ]
testData <- bildata[-trainIndex, ]


# NAN-values
mode_value <- names(sort(table(trainData$drivning), decreasing = TRUE))[1]
trainData$drivning[is.na(trainData$drivning)] <- mode_value
testData$drivning[is.na(testData$drivning)] <- mode_value

# Explore numerical variables
num_vars <- names(trainData)[sapply(trainData, is.numeric)]

par(mfrow = c(2, 2))
for (var in num_vars) {
  hist(trainData[[var]], 
       main = paste("Histogram av", var), 
       xlab = var, 
       col = "lightblue")
}
par(mfrow = c(2, 2))
for (var in num_vars) {
  boxplot(trainData[[var]], 
          main = paste("Boxplot av", var), 
          col = "lightblue", 
          horizontal = TRUE)
}
par(mfrow = c(2, 2))
for (var in num_vars) {
  if (var != "pris") { 
    plot(trainData[[var]], trainData$pris, 
         main = paste("Pris vs", var), 
         xlab = var, 
         ylab = "Pris", 
         col = "lightblue", 
         pch = 16)
    lines(loess.smooth(trainData[[var]], trainData$pris), col = "red", lwd = 2)
  }
}

# The variables are not completely normally distributed. 
# The correlation between the response variable and the features seems relatively linear. 


# Outliers 
# Looking at miltal and hästkrafter as identified in boxplots. 
Q1 <- quantile(trainData$hästkrafter, 0.25)
Q3 <- quantile(trainData$hästkrafter, 0.75)
IQR <- Q3 - Q1
outlier_rows <- which(trainData$hästkrafter < (Q1 - 1.5 * IQR) | trainData$hästkrafter > (Q3 + 1.5 * IQR))
trainData[outlier_rows, ]

Q1 <- quantile(trainData$miltal, 0.25)
Q3 <- quantile(trainData$miltal, 0.75)
IQR <- Q3 - Q1
outlier_rows <- which(trainData$miltal < (Q1 - 1.5 * IQR) | trainData$miltal > (Q3 + 1.5 * IQR))
trainData[outlier_rows, ]

# Analysis of outliers shows that they are legit. If they seem to have a big affect on model performance, they will be handled then. 


# Correlation between variables
cor_matrix <- cor(trainData[, sapply(trainData, is.numeric)], use = "complete.obs")
par(mfrow = c(1, 1))
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.cex = 0.8)
print(cor_matrix)

# Correlation matrix shows that miltal and hästkrafter shows the highest correlation with pris. 
# There is a strong correlation between miltal and modellår (not surprising).  


# Explore categorical variables
summary(trainData[, sapply(trainData, is.factor)])

cat_vars <- names(trainData)[sapply(trainData, is.factor)]  
par(mfrow = c(3, 2))
for (var in cat_vars) {
  counts <- table(trainData[[var]])
  barplot(counts, 
          main = paste("Fördelning av", var),
          col = "lightblue",
          las = 2)
}

par(mfrow = c(3, 2))
for (var in cat_vars) {
  boxplot(trainData$pris ~ trainData[[var]], 
          main = paste("Pris vs", var), 
          xlab = var, 
          ylab = "Pris", 
          col = "lightblue",
          las = 2)
}

# Create dummy-variables
trainData <- model.matrix(~ . + 0, data = trainData) %>% as.data.frame()
trainData <- trainData[, !grepl("bränslebensin", colnames(trainData))] # Remove redundant dummy variable to ensure avoiding multicollinearity in the model.

testData <- model.matrix(~ . + 0, data = testData) %>% as.data.frame()
testData <- testData[, !grepl("bränslebensin", colnames(testData))] # Remove redundant dummy variable to ensure avoiding multicollinearity in the model.


# Variable standardization / scaling
num_pred <- setdiff(num_vars, "pris")

trainData[num_pred] <- scale(trainData[num_pred])
testData[num_pred] <- scale(testData[num_pred])


# Create linear regression model for EDA
lm_model <- lm(pris ~ ., data = trainData)

par(mfrow=c(2,2))
plot(lm_model)
summary(lm_model)

# Log-transformation of price
lm_model <- lm(log(pris) ~ ., data = trainData)

par(mfrow=c(2,2))
plot(lm_model)
summary(lm_model)

# Check Multicollinearity
vif(lm_model)
# High VIF values indicated multicollinearity, suggesting the removal of biltyp. 
# However, this led to worse predictive performance, making the removal unjustified.
# Lasso and Ridge will be included in model selection. 
# trainData_no_biltyp <- trainData %>% select(-biltyp) # NOTE! Needs to be done before creating dummy-variables above! 
# trainData_no_biltyp <- model.matrix(~ . + 0, data = trainData_no_biltyp) %>% as.data.frame()
# trainData_no_biltyp <- trainData_no_biltyp[, !grepl("bränslebensin", colnames(trainData_no_biltyp))] # Remove redundant dummy variable to ensure avoiding multicollinearity in the model.


# Identify and remove high leverage points and outliers.
# High leverage points
lev <- hatvalues(lm_model)
high_lev <- which(lev > (2 * mean(lev)))
# Outliers
stud_resid <- rstudent(lm_model)
outliers <- which(abs(stud_resid) > 3)
# Is any high leverage points and outliers the same? No. 
problem_obs <- intersect(high_lev, outliers)
print(problem_obs)

trainData_no_highlev_outl <- trainData[-c(high_lev, outliers), ]
lm_no_highlev_outl <- lm(log(pris) ~ ., data = trainData_no_highlev_outl)
par(mfrow=c(2,2))
plot(lm_no_highlev_outl)
summary(lm_no_highlev_outl)
# Lower RSE
# QQ-plot follows line better = Residuals follow normality better
# Fewer extreme leverage points
# Conclusion: Removing high leverage points and outliers improved model stability and assumptions.
# Analysis of high lev points and outliers show that they are legit.
# High risk to remove. Overfitting to the training data?
# Lasso and Ridge will be included in model selection.


# Residuals vs Fitted in the plot has a pattern that suggest a systematic error.
# Transformation (log, poly) of variables (miltal, hästkrafter, modellår) didn't improve it significantly.
# When prices were collected from Blocket prices were filtered between 150 000 - 500 000 SEK.
# This might be why the pattern is what it is, we need better data to fix.
# Log-transformation of price will be kept. 


# Coefficient analysis

# Log -> %
co <- coef(lm_model)
exp_co <- exp(co) - 1
exp_co_percent <- round(exp_co * 100, 2)
names(exp_co_percent) <- names(co)
print(exp_co_percent)
# Confidence interval
conf_int <- confint(lm_model)
exp_conf_int <- exp(conf_int)
colnames(exp_conf_int) <- c("Lower", "Upper")
# Plot
coef_df <- data.frame(
  Variable = rownames(exp_conf_int),
  Estimate = exp(coef(lm_model)), 
  Lower = exp_conf_int[, "Lower"],
  Upper = exp_conf_int[, "Upper"]
)
coef_df <- coef_df[-1, ]
ggplot(coef_df, aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_point(color = "blue", size = 3) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "black") +
  coord_flip() +
  labs(title = "Coefficients with Confidence Intervals",
       x = "Variable", y = "Effect size (%)") +
  theme_minimal()



# Numeric variables
# Modellår (+18.49%) - Newer cars are more expensive. 
# Hästkrafter (+18.33%) - More horsepower increases price. 
# Miltal (-9.15%) - Higher milage increases price. 

# Categorical variables
# Bränsle. Reference category: Bensin. Diesel (+10.80%) & El (+8.77%) are more expensive. Miljöbränsle/Hybrid  (+1.77%) a bit more expensive, but not statistically significant. 
# Växellåda. Reference category: Automat. Manuell (-10.72%) is cheaper. 
# Bilmärke. Reference category: Audi. Renault (-15.62%), Volkswagen (-7.03%), Ford (-9.77%), Toyota (-3.42%) are cheaper. Volvo (+0.46%) a bit more expensive, but not statistically significant. 
# Biltyp. Reference category: Cab. Kombibilar (-24.69%), Halvkombi (-26.17%), Suv (-19.05%), Coupé (-11.04%), Sedan (-20.16%) much cheaper. 
# Drivning. Reference category: Fyrhjulsdrift. Tvåhjulsdrift (-6.49%) is cheaper. 



# MODELS

# K-fold cross validation for evaluation of models
trainControl <- trainControl(method = "cv", number = 10)


# Linear regression - Model 1
lm_model <- train(log(pris) ~ ., data = trainData, method = "lm", trControl = trainControl)
summary(lm_model)
# RMSE
lm_model_rmse <- lm_model$results$RMSE


# Linear regression - Model 2, with best subset selection
regfit.full <- regsubsets(log(pris) ~ ., data = bildata, nvmax = 20)
reg.summary <- summary(regfit.full)

# Find "elbows". If not elbws, basically all variables (14) is included
find_elbow <- function(values) {
  diffs <- diff(values)
  elbow <- which.min(diffs[1:(length(diffs) - 1)] - diffs[2:length(diffs)]) + 1
  return(elbow)
}
rss_elbow <- find_elbow(reg.summary$rss)
adjr2_elbow <- find_elbow(reg.summary$adjr2)
cp_elbow <- find_elbow(reg.summary$cp)
bic_elbow <- find_elbow(reg.summary$bic)

par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Antal variabler", ylab = "RSS", type = "l", main = "RSS")
points(rss_elbow, reg.summary$rss[rss_elbow], col = "red", cex = 2, pch = 20)
plot(reg.summary$adjr2, xlab = "Antal variabler", ylab = "Justerat R²", type = "l", main = "Justerat R²")
points(adjr2_elbow, reg.summary$adjr2[adjr2_elbow], col = "red", cex = 2, pch = 20)
plot(reg.summary$cp, xlab = "Antal variabler", ylab = "Cp", type = "l", main = "Cp")
points(cp_elbow, reg.summary$cp[cp_elbow], col = "red", cex = 2, pch = 20)
plot(reg.summary$bic, xlab = "Antal variabler", ylab = "BIC", type = "l", main = "BIC")
points(bic_elbow, reg.summary$bic[bic_elbow], col = "red", cex = 2, pch = 20)

chosen_vars <- data.frame(
  Mått = c("RSS", "Justerat R²", "Cp", "BIC"),
  Antal_Variabler = c(rss_elbow, adjr2_elbow, cp_elbow, bic_elbow)
)
best_model_size <- as.integer(names(sort(table(chosen_vars$Antal_Variabler), decreasing = TRUE)[1]))
cat('Optimal number of variables according to "elbows":', best_model_size, "\n")
best_vars <- coef(regfit.full, best_model_size)
# Modellår and Hästkrafter

best_vars_names <- setdiff(names(coef(regfit.full, best_model_size)), "(Intercept)")
best_subset_lm_model <- train(log(pris) ~ ., data = trainData[, c("pris", best_vars_names)], method = "lm", trControl = trainControl)
summary(best_subset_lm_model)

# RMSE
best_subset_lm_model_rmse <- best_subset_lm_model$results$RMSE


# LASSO och RIDGE
X <- model.matrix(log(pris) ~ ., data = trainData)[, -1]
y <- log(trainData$pris)

lambda_grid_lasso <- 10^seq(log10(0.00001), log10(0.0001), length = 100)
lambda_grid_ridge <- 10^seq(log10(0.001), log10(1), length = 100)

lasso_model <- train(
  x = X, y = y,
  method = "glmnet",
  trControl = trainControl,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid_lasso)
)

ridge_model <- train(
  x = X, y = y,
  method = "glmnet",
  trControl = trainControl,
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid_ridge) 
)

best_lasso <- lasso_model$finalModel
best_ridge <- ridge_model$finalModel

# RMSE
lasso_rmse <- min(lasso_model$results$RMSE)
ridge_rmse <- min(ridge_model$results$RMSE)


# MODEL SELECTION

# RMSE
lm_rmse <- min(lm_model$results$RMSE)
lm_best_sel_rmse <- min(best_subset_lm_model$results$RMSE)
lasso_rmse <- min(lasso_model$results$RMSE)
ridge_rmse <- min(ridge_model$results$RMSE)
# R²
lm_r2 <- max(lm_model$results$Rsquared)
lm_best_selection <- max(best_subset_lm_model$results$Rsquared)
lasso_r2 <- max(lasso_model$results$Rsquared)
ridge_r2 <- max(ridge_model$results$Rsquared)

comparison <- data.frame(
  Model = c("Linear Regression", "Linear Regression with Best subset selection", "Lasso", "Ridge"),
  RMSE = c(lm_rmse, lm_best_sel_rmse, lasso_rmse, ridge_rmse),
  R2 = c(lm_r2, lm_best_selection, lasso_r2, ridge_r2)
)
print(comparison)

# Lasso is evaluated to the best model. 


# MODEL EVALUATION 
best_lambda <- lasso_model$bestTune$lambda
lasso_model_final <- glmnet(X, y, alpha = 1, lambda = best_lambda)

X_test <- model.matrix(log(pris) ~ ., data = testData)[, -1]
y_test <- log(testData$pris)

lasso_pred <- predict(lasso_model_final, newx = X_test, s = best_lambda)

# RMSE
lasso_rmse_test <- sqrt(mean((y_test  - lasso_pred)^2))
print(lasso_rmse_test)



# Look into a more non-linear model could make predictions better. 