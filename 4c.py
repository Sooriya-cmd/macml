import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data = pd.read_csv("C:\\Users\\Admin\\Documents\\sem5\\ml\\heart.csv")
print("Dataset:")
print(data.head())
data = pd.get_dummies(data, drop_first=True)
data = data.dropna()
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
Y_pred_linear = linear_model.predict(X_test)
nonlinear_model = RandomForestRegressor(n_estimators=100, random_state=42)
nonlinear_model.fit(X_train, Y_train)
Y_pred_nonlinear = nonlinear_model.predict(X_test)
def evaluate_model(name, Y_test, Y_pred):
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f"\n{name}")
    print("MSE :", round(mse, 4))
    print("RMSE :", round(rmse, 4))
    print("MAE :", round(mae, 4))
    print("R2 :", round(r2, 4))
evaluate_model("Linear Model", Y_test, Y_pred_linear)
evaluate_model("Random Forest Regressor", Y_test, Y_pred_nonlinear)