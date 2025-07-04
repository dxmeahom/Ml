import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = sns.load_dataset('iris')
print(data.head())

#Tracer chaque caractéristique  en fonction du type
plt.xlabel('Features')
plt.ylabel('Type')
pltX = data.loc[:, 'sepal_length']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, c='b', label='sepal_length')

pltX = data.loc[:, 'sepal_width']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, c='r', label='sepal_width')

pltX = data.loc[:, 'petal_length']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, c='y', label='petal_length')

pltX = data.loc[:, 'petal_width']
pltY = data.loc[:, 'species']
plt.scatter(pltX, pltY, c='g', label='petal_width')
plt.legend()
plt.show()
#Préparation des données d'enrainement
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
# split du jeu de données
# split en 70% train et 30% test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print("X_train.shape: ", x_train.shape)
print("X_test.shape : ", x_test.shape)

# entraînement du modèle
model = LogisticRegression()
model.fit(x_train, y_train)
# tester le modèle en utilisant la collection de test
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)
# calculer l'accuracy du modèle en comparant les espèces prédites par le modèle 'y_pred'avec les vraies réponses 'y_test'
print('Accuracy', accuracy_score(y_test, y_pred))

# faire une prédiction sur de nouvelles données
sample = [[5, 5, 3, 2], [2, 4, 3, 5]]
preds = model.predict(sample)
print("Prédictions:", preds)

