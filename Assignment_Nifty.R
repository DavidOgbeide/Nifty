if (!require(mlba)) {
  library(devtools)
  install_github("gedeck/mlba/mlba", force=TRUE)
}

library(dplyr)
library(ggplot2)
library(tidyverse)
library(psych)
library(rstatix)
library(caret)
library(corrplot)


nifty.df <- read.csv("C:/Users/drdav/Downloads/Nifty.csv")
head(nifty.df)

# Exploratory Data Analysis
#Checking for missing values count

summary(nifty.df)

# Listing the columns with missing values

missing_columns <- colnames(nifty.df)[colSums(is.na(nifty.df)) > 0]
print(missing_columns)

# replacing missing values with the median of the colums
for (column in missing_columns) {
  median_value <- median(nifty.df[[column]], na.rm = TRUE)
  nifty.df[[column]][is.na(nifty.df[[column]])] <- median_value
}
head(nifty.df)
summary(nifty.df)

# Checking the distribution
hist(nifty.df)

# Check data types
column_types <- sapply(nifty.df, class)
print(column_types)

# Boxplot for each numeric column

numeric_columns <- c("DJI", "Nifty", "Vix", "SP500", "Brent_Crude", "USD_INR", "CBOE_10Y")

for (column in numeric_columns) {
  boxplot(nifty.df[[column]], main=column)
}

numeric_columns <- c("DJI", "Nifty", "Vix", "SP500", "Brent_Crude", "USD_INR", "CBOE_10Y")

boxplot(nifty.df[, numeric_columns], main="Boxplot of Numeric Columns")

boxplot(nifty.df$DJI, main="Boxplot of DJI")
boxplot(nifty.df$Nifty, main="Boxplot of Nifty")
boxplot(nifty.df$Vix, main="Boxplot of Vix")
boxplot(nifty.df$SP500, main="Boxplot of SP500")
boxplot(nifty.df$Brent_Crude, main="Boxplot of Brent_Crude")
boxplot(nifty.df$USD_INR, main="Boxplot of USD_INR")
boxplot(nifty.df$CBOE_10Y, main="Boxplot of CBOE_10Y")


# Define the quantile thresholds (e.g., 1% and 99%)
lower_quantile <- 0.01
upper_quantile <- 0.99

# Winsorize the columns: to handle outliers in a dataset by reducing the impact of extreme values 
# without entirely removing them. The essence of winsorizing lies in adjusting extreme values by 
# replacing them with less extreme values, typically the nearest values that are not considered outliers.
# The key essence of winsorizing is to make extreme values less influential in statistical analysis or 
# modeling without completely discarding them. This technique helps in making the data more robust against 
# the influence of outliers while preserving the overall distribution and pattern of the dataset.


nifty.df_clean <- nifty.df %>%
  mutate(DJI = pmin(quantile(DJI, probs = upper_quantile), pmax(DJI, quantile(DJI, probs = lower_quantile))),
         Nifty = pmin(quantile(Nifty, probs = upper_quantile), pmax(Nifty, quantile(Nifty, probs = lower_quantile))),
         Vix = pmin(quantile(Vix, probs = upper_quantile), pmax(Vix, quantile(Vix, probs = lower_quantile))),
         SP500 = pmin(quantile(SP500, probs = upper_quantile), pmax(SP500, quantile(SP500, probs = lower_quantile)))
         # Add other columns as needed
  )

# Data Partition
# set.seed(123): Sets a seed for reproducibility. It ensures that if you run this code multiple times, 
# you'll get the same random samples, making the process reproducible

# this code randomly selects approximately 60% of the rows from the nifty.df dataset for the training 
# set and assigns the remaining rows as the holdout dataset, facilitating the partitioning of data for 
# training and evaluating models or conducting further analysis/validation on unseen data. 

set.seed(123)
train.rows <- sample(rownames(nifty.df), 966*0.6)
train.data <- nifty.df[train.rows, ]
holdout.rows <- setdiff(rownames(nifty.df), train.rows)
holdout.data <- nifty.df[holdout.rows, ]


# When you apply scale() to nifty.df, it standardizes each column, making the mean of each column zero 
# and the standard deviation one.

train.data <- data.frame(scale(train.data))
holdout.data <- data.frame(scale(holdout.data))
scaled_data <- rbind(train.data, holdout.data)

# Fitting a linear model
Reg <- lm(Nifty~., data = train.data)
summary(Reg)

preprocessParams <- preProcess(nifty.df, method = c("range"))
scaled_data <- predict(preprocessParams, nifty.df)


preprocessParams <- preProcess(nifty.df, method = c("center", "scale"))
scaled_data <- predict(preprocessParams, nifty.df)

cor(nifty.df)
cor(scaled_data)
correlation_matrix <- cor(scaled_data)
summary(correlation_matrix)

corrplot(correlation_matrix, method = "color")

colnames(scaled_data)
# 'Nifty' is the dependent variable
model <- lm(Nifty ~ ., data = scaled_data[, -2])  # Exclude 'Nifty' from independent variables
summary(model)

Reg.backward <-step(Reg, direction = "backward")
summary(Reg.backward)
hist(Reg.backward$residuals)
plot(Reg.backward$residuals~Reg.backward$fitted.values)

colnames(scaled_data)
model <- lm(Nifty ~ ., data = scaled_data[, colnames(scaled_data) != "Nifty"])
model <- lm(Nifty ~ ., data = scaled_data)
model <- lm(Nifty ~ ., data = scaled_data[, colnames(scaled_data) != "Nifty"])
summary(scaled_data)

model <- lm(Nifty ~ ., data = scaled_data[, colnames(scaled_data) != "Nifty"])

predictors <- colnames(scaled_data)[colnames(scaled_data) != "Nifty"]
model <- lm(Nifty ~ ., data = scaled_data[, predictors])
colnames(scaled_data)
set.seed(111)






# Check the structure of scaled_data
str(scaled_data)

# Verify the presence of 'Nifty' column
summary(scaled_data)

# Create the model with explicit column names
predictors <- colnames(scaled_data)[colnames(scaled_data) != "Nifty"]
model1 <- lm(scaled_data$Nifty ~ ., data = scaled_data[, predictors])
summary(model1)


library(stats)

# Assuming 'scaled_data' has been prepared and contains only numeric columns
k <- 3  # Number of clusters
kmeans_result <- kmeans(scaled_data, centers = k)

# Cluster centers
kmeans_result$centers

# Cluster assignments for each data point
kmeans_result$cluster

# Within-cluster sum of squares
kmeans_result$tot.withinss

# Perform PCA
pca_result <- prcomp(scaled_data)

# Plot the clusters in the space of the first two principal components
plot(pca_result$x[, 1], pca_result$x[, 2], col = kmeans_result$cluster, 
     main = "K-means Clustering (PCA Space)", xlab = "PC1", ylab = "PC2")


hist(pca_result$x[, 1], col = kmeans_result$cluster, xlab = "PC1", main = "Histogram of PC1 by Cluster")
legend("topright", legend = unique(kmeans_result$cluster), col = unique(kmeans_result$cluster), fill = unique(kmeans_result$cluster))

plot(pca_result$x[, 1], pca_result$x[, 2], 
     col = kmeans_result$cluster,
     main = "PCA: PC1 vs PC2", 
     xlab = "PC1", ylab = "PC2")
legend("topright", legend = unique(kmeans_result$cluster), col = unique(kmeans_result$cluster), pch = 1)


# Histogram for PC1
hist(pca_result$x[, 1], 
     main = "Histogram of PC1",
     xlab = "PC1",
     col = "skyblue",
     border = "black")

# Histogram for PC2
hist(pca_result$x[, 2], 
     main = "Histogram of PC2",
     xlab = "PC2",
     col = "lightgreen",
     border = "black")
