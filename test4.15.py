from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
model =  KNeighborsClassifier (n_neighbors=3)
model.fit(X_train, y_train)
y_predict = model.predict (X_test)
accuracy = accuracy_score(y_test,y_predict)
print("准确率：",accuracy)





