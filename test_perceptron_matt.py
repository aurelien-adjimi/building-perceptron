from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import perceptron_matt as perceptron

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = perceptron.Perceptron()
model.fit(X_train, y_train)

print('Train accuracy:', f'{model.score(X_train, y_train):.2f}')
print('Test accuracy:', f'{model.score(X_test, y_test):.2f}')

y_pred = model.predict(X_test)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))