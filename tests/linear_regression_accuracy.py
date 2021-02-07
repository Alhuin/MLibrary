import parent_import
# import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from MLibrary.linear_regression import LinearRegression
from MLibrary.utils import mse

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
y_pred_line = regressor.predict(X)

# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8, 6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
# plt.show()

print(f"Linear Regression classification MSE: {mse(y_test, predictions)}")
