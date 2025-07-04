import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('valeurs_trimestrielles.csv', sep=';')
print("L'entête du dataframe: \n\n", dataset.head())
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
#Afficher le nuage de point
plt.scatter(X, Y, c='r')
plt.xlabel('Les périodes')
plt.ylabel('Les indices')
plt.show()
#Diviser le dataset en training set & test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print("X_train.shape: ", x_train.shape)
print("X_test.shape : ", x_test.shape)
#Construction du modéle
model = LinearRegression()
model.fit(x_train, y_train)
#Faire des nouvelles prédictions
y_pred = model.predict(x_test)
print(y_pred)
#Faire des prédictions dans le future
prediction = model.predict([[98], [99], [100]])
print('\n\n PREDICTION : \n\n', prediction)
#Visualisation le résultat
plt.scatter(x_test, y_test, c='r')
plt.plot(x_train, model.predict(x_train), c='blue')
plt.title('Evolution des loyers')
plt.xlabel('Trimestre')
plt.ylabel('Loyer')
plt.show()