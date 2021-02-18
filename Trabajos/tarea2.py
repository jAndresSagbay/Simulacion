import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


boston = datasets.load_boston()

X = boston.data[:, np.newaxis, 5]
y = boston.target
plt.scatter(X, y)
plt.title('Datos')
plt.show()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lr = linear_model.LinearRegression()

lr.fit(X_train, y_train)

Y_pred = lr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresión Lineal Simple')
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()

print('Valor de la pendiente o coeficiente "a":')
print(lr.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr.intercept_)
print()
print('La ecuación del modelo es igual a:')
print('y = ', lr.coef_, 'x ', lr.intercept_)

print()
print('Precisión del modelo:')
print(lr.score(X_train, y_train))