# Loading data
input <- read.csv(file.choose())
View(input)


## Removing Unnecessary columns
data <- input[, -1]
attach(data)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(data)


# Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Box plot Representation

boxplot(Alcohol, col = "dodgerblue4",main = "Alcohol")
boxplot(Malic, col = "dodgerblue4",main = "Malic")
boxplot(Alcalinity, col = "dodgerblue4",main = "Alcalinity")
boxplot(Magnesium, col = "dodgerblue4",main = "Magnesium")
boxplot(Phenols, col = "dodgerblue4",main = "Phenols")
boxplot(Proanthocyanins, col = "dodgerblue4",main = "Proanthocyanins")


# Histogram Representation

hist(Alcohol, col = "blue",main = "Alcohol")
hist(Malic, col = "blue",main = "Malic")
hist(Alcalinity, col = "blue",main = "Alcalinity")
hist(Magnesium, col = "blue",main = "Magnesium")
hist(Phenols, col = "blue",main = "Phenols")
hist(Proanthocyanins, col = "blue",main = "Proanthocyanins")


########## DIMENSION REDUCTION (PCA) ANALYSIS #############

pcaObj <- princomp(data, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
pcaObj$scores[, 1:3]

# Top 3 pca scores 
final <- cbind(input[, 1], pcaObj$scores[, 1:3])
View(final)


# Scatter diagram
plot(final)

#### HIERARCHICAL CLUSTERING ####

# Loading the dataset
input <- read.csv(file.choose())
View(input)


# Removing Unnecessary columns
Data <- input[, -1]
attach(Data)

summary(Data)

# Normalize the data
normalized_data <- scale(Data[, 1:13]) 

summary(normalized_data)

# Distance matrix
d <- dist(normalized_data, method = "euclidean") 

fit <- hclust(d, method = "ward.D2")

# Display dendrogram
plot(fit) 
plot(fit, hang = -1)

groups <- cutree(fit, k =14)# Cut tree into 14 clusters

rect.hclust(fit, k =14, border = "red")

cluster <- as.matrix(groups)

final <- data.frame(cluster, Data)

aggregate(Data[, 1:11], by = list(final$cluster), FUN = mean)

library(readr)
write_csv(final, "Wine_hierarachial_R.csv")

getwd()

#### K MEANS CLUSTERING ####

# Importing the dataset
input <- read.csv(file.choose())
View(input)


## Removing Unnecessary Columns
Data <- input[, -1]
attach(Data)
str(Data)

summary(Data)

# Normalize the data
normalized_data <- scale(Data[, 1:13]) # As we already removed "Type" column so all columns need to normalize

summary(normalized_data)

# Elbow curve to decide the k value
twss <- NULL
for (i in 2:12) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:12, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# 4 Cluster Solution
fit <- kmeans(normalized_data, 4) 
str(fit)
fit$cluster
final <- data.frame(fit$cluster, Data) # Append cluster membership

A <- aggregate(Data[, 1:13], by = list(fit$cluster), FUN = mean)

library(readr)
write_csv(A, "Wine_kmeans_R.csv")

getwd()
