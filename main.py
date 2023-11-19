import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()
X,y = data.data, data.target

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=1234)


from logistic_regression import LogisticRegression

regression = LogisticRegression(learning_rate=0.001)
regression.fit(X_train, y_train)
predicted = regression.predict(X_test)
