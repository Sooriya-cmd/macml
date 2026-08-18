import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data =pd.read_csv("C:\\Users\\Admin\\Downloads\\logistic regression dataset-Social_Network_Ads.csv")
X = data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Predicted values :",y_pred)
print("Accuracy :",accuracy_score(y_test,y_pred))