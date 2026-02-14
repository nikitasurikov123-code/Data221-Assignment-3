import pandas as pd
from sklearn.model_selection import train_test_split
data_frame = pd.read_csv("kidney_disease.csv")

X = data_frame.drop(columns=["classification"])
y = data_frame["classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#We shouldn't train and test on the same data because the model will only memorize the same data, instead of different data
#leading to it not learning patters as it is only trained on one thing, so if you give it new data, it won't perform
#as well as something that was trained on many different things. This is also known as overfitting. Overall, the model
#will not learn pattern recognition because there's no generalization.

#The purpose of the testing set is to simulate new data so that we can see how well it can generalize something, so
#we can see how good it is at pattern recognition.