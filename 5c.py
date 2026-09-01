import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv(r"C:\Users\Admin\Documents\sem5\ml\iris_200_95_accuracy.csv")

X = data.drop("species", axis=1)
Y = data["species"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = SVC()
model.fit(x_train, y_train)

Y_Pred = model.predict(x_test)

for actual, predicted in zip(y_test, Y_Pred):
    if actual == predicted:
        print("Corrected prediction")
    else:
        print("wrong prediction")
    
    print("Actual species :", actual)
    print("Predicted species :", predicted)
    print()

accuracy = (y_test == Y_Pred).mean()
print("Accuracy:", accuracy * 100, "%")