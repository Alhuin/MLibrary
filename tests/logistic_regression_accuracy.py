import parent_import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from MLibrary.logistic_regression import LogisticRegression
from MLibrary.utils import accuracy


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
regressor = LogisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print(f"Logistic Regression classification accuracy: {accuracy(y_test, predictions)}")
