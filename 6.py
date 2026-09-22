import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"C:\Users\Admin\Documents\sem5\ml\iris_200_95_accuracy.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("k-Nearest Neighbor Classification")

correct = 0
wrong = 0

for i in range(len(y_test)):
    actual = y_test.iloc[i]
    predicted = y_pred[i]

    if actual == predicted:
        print(f"Correct Prediction: Actual = {actual}, Predicted = {predicted}")
        correct += 1
    else:
        print(f"Wrong Prediction: Actual = {actual}, Predicted = {predicted}")
        wrong += 1

print("Total test samples :", len(y_test))
print("Correct predictions:", correct)
print("Wrong predictions:", wrong)
print("Accuracy:", knn.score(X_test, y_test))