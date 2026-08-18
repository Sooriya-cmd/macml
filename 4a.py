import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
train = pd.read_csv("c:\\Users\\Admin\\Downloads\\train (2).csv")
X_train = train.iloc[:, :-1]
Y_train = train.iloc[:, -1]
model = LinearRegression()
model.fit(X_train, Y_train)
test = pd.read_csv("C:\\Users\\Admin\\Downloads\\test.csv")
X_test = test.iloc[:, :-1]
Y_test = test.iloc[:, -1]
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", Y_pred)
print("Mean Squared Error:", mse)