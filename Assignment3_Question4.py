import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
data_frame = pd.read_csv("kidney_disease.csv")

X = data_frame.drop(columns=["classification"])
y = data_frame["classification"]
#https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
#this turned it into true/false or 1/0 variables so that it could be used because it would have an error before
X = pd.get_dummies(X)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

num_neighbors = 5
knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
knn_model.fit(X_train, y_train)
y_predict = knn_model.predict(X_test)

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