import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_csv(r"C:\Users\Admin\Documents\sem5\ml\heart.csv")

data = pd.get_dummies(data, drop_first=True)
data = data.dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results[name] = [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred),
        roc_auc_score(y_test, prob)
    ]

    print("\n", name)
    print("Accuracy:", results[name][0])
    print("Precision:", results[name][1])
    print("Recall:", results[name][2])
    print("F1-Score:", results[name][3])
    print("ROC-AUC:", results[name][4])

best = max(results, key=lambda x: results[x][0])
print("\nBest Model:", best)

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

for i, metric in enumerate(metrics):
    values = [results[name][i] for name in models]
    plt.figure()
    plt.scatter(models.keys(), values)
    plt.title(metric)
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.show()