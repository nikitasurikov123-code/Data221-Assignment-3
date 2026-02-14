import pandas as pd
data_frame = pd.read_csv("crime.csv")
statistics = (data_frame["ViolentCrimesPerPop"])
mean = statistics.mean()
median = statistics.median()
STD = statistics.std()
minimum_value = statistics.min()
maximum_value = statistics.max()
print("Here are the results: ")
print("Mean: " ,mean)
print("Median: " ,median)
print("Standard deviation: " ,STD)
print("Minimum value: " ,minimum_value)
print("Maximum value: " ,maximum_value)
#Compare the mean and median. Does the distribution look symmetric or skewed? Explain briefly.
#The mean and median are almost the same, but still slightly different. If a mean > median, that means
#that a graph is right skewed, since mean is a little greater than the median, the graph could be
#said to be right skewed.

#If there are extreme values (very large or very small), which statistic is more affected: mean or median? Explain why
#The statistic more affected by outliers, or, extreme values is the mean. This is because the mean is calculated
#using every statistic in the dataset. So if there's 10 points, 9 being 2-4, and the last one being 100, the
#mean would be very skewed, while the median would stay almost the same, because it accounts for number of data points
#halfway below or halfway above it, in the middle, so if you had 10 points being 2-4 and another being 100, counting
#just 9 points without the 10th would make the median around 3. When you add the 10th point, it would move to 3.1-3.5,
#so outliers don't move it by much.

import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv("crime.csv")
statistics = (data_frame["ViolentCrimesPerPop"])

plt.figure()
plt.hist(statistics, bins = 50)
plt.title("Histogram of Violent crimes per population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("N/A")
plt.show()

plt.figure()
plt.boxplot(statistics)
plt.title("Boxplot of Violent crimes per population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("N/A")
plt.show()

#The histogram is right-skewed. Most populations are
#clustered around 0-0.4, where we see the most data points, and they trail off and get less and less as they
#go towards the right. Since it's spread out a lot, that indicates that violent crimes are very varied acros
#different communities. Since as crime increases, frequency decreases, that indicates that there are
#fewer areas with very high violent crime rates.

#The median is shown by the orange line inside the box. The median is somewhere between 0.37-0.39. This means
#that half of the populations have violent crime rates below 0.39, and half of them hav violent crime rates
#above 0.39. Since the median is more on the bottom of the boxplot, the data can be said to be right skewed,
#which is consistent with the histogram. The median indicates that most areas have low to moderate crime rates.

#There are no individual point outside the bounds of the upper fence (UF) or lower fence (LF), which means that
#are no obvious outliers. Even though on the histogram, there appears to be an outlier, if you look at the
#boxplot, it is still within the range of the UF, so there is no outlier, as all data falls within the range
#of the UF. The lower end also doesn't have any outliers, and ends at 0, as you can't have negative crime in this sense.
#Overall, the boxplot does not suggest outliers, as everything falls withing the range of the UF and LF.

import pandas as pd
from sklearn.model_selection import train_test_split
kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")

X = kidney_disease_data_frame.drop(columns=["classification"])
y = kidney_disease_data_frame["classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#We shouldn't train and test on the same data because the model will only memorize the same data, instead of different data
#leading to it not learning patters as it is only trained on one thing, so if you give it new data, it won't perform
#as well as something that was trained on many different things. This is also known as overfitting. Overall, the model
#will not learn pattern recognition because there's no generalization.

#The purpose of the testing set is to simulate new data so that we can see how well it can generalize something, so
#we can see how good it is at pattern recognition.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")

X = kidney_disease_data_frame.drop(columns=["classification"])
y = kidney_disease_data_frame["classification"]
#https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
#this turned it into true/false or 1/0 variables so that it could be used because it would have an error before
X = pd.get_dummies(X)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

num_neighbors = 5
knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
knn_model.fit(X_train, y_train)
y_predict = knn_model.predict(X_test)
#makes sure that "ckd" is considered positive because it wouldn't run without it
confusion_m = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, pos_label="ckd")
recall = recall_score(y_test, y_predict, pos_label="ckd")
F1_score = f1_score(y_test, y_predict, pos_label="ckd")

print("Confusion Matrix: " , confusion_m)
print("Accuracy: " , accuracy)
print("Precision: " , precision)
print("Recall: " , recall)
print("F1 Score: " , F1_score)

#In the context of kidney disease prediction, true positive means that ckd was predicted, which means that the patient
#has ckd. A true negative means that no ckd was predicted, which means that the patient doesn't have ckd.
#A false positive means that ckd was predicted when the patient didn't have ckd. Finally, a false negative means that
#ckd was not predicted for the patient, but they actually do have ckd.

#Accuracy isn't always enough because it treats every error the same, so if you get a false negative, it is treated
#as costly as a false positive, when in reality, a false negative is very costly, and way worse than a false positive.

#The most important metric is recall because it provides the number of how many people the model caught that actually have
#ckd If there is a higher recall, that means that less people were missed that were sick, which means fewer false negatives.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")

X = kidney_disease_data_frame.drop(columns=["classification"])
y = kidney_disease_data_frame["classification"]
#https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
X = pd.get_dummies(X)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

k_values = [1,3,5,7,9]
accuracy_results = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_results.append([k,accuracy])

results = pd.DataFrame(accuracy_results, columns=["k_value","accuracy"])
print("best K value: ", results)

#A lower K means there's a greater chance of overfitting, and a higher K means a greater chance of underfitting.

#The lower the k value, the bigger change of overfitting because the model is making predicitons using fewer points.
#This can make it memorize outliers, which is not good if you want it to learn a pattern, as an outlier would disrupt it.
#Since its overfitted, it has low bias and high variance, and if it's introduced to new data, then it won't function as expected

#If a K value is too large, it will cause underfitting, which means that it's too generalized and relies too much on
#lots of neighbors. This means that it can start predicting generalizations according to the majority class among the neighbors,
#instead of the correct predicted class for a minority.

