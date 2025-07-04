import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("Iris.csv")
print("L'entête du dataframe: \n\n", df.head())

# sélectionner les observations des colonnes 1 à 4 dans X et de la colonne 5 dans Y
X = df.iloc[:, 1:5]
Y = df.iloc[:, 5]
print("\nL'entête de X: \n\n", X.head())
print("\nL'entête de Y: \n\n", Y.head())


# Partie 2: Visualisation des données avec Matplotlib
# récuperer les valeurs uniques de Y (les 3 différentes espèces)
unique_y = np.unique(Y)

# initialiser une figure avec 10 pouces de large et 6 pouces de hauteur
plt.figure(figsize=(10, 6))

# créer un objet axe pour contenir nos données
axe = plt.subplot()

# ici par exemple, nous souhaitons avoir sur l'axe des abscices SepalLength, et sur les y PetalLength
data_x_col = 0
data_y_col = 2
axe.set_xlabel(X.columns.values[data_x_col])
axe.set_ylabel(X.columns.values[data_y_col])


# tracer un nuage de points pour chacune des espèces,
# en sélectionnant seulement les observations de l'espèce en question X[Y == i]
for i in unique_y:
    plt.scatter(X[Y == i].iloc[:, data_x_col], X[Y == i].iloc[:, data_y_col], label=i)

# afficher une légende correspondant à chaque nuage de points
plt.legend()

# afficher la figure
plt.show()
# Partie 3: Apprentissage automatique avec Sklearn
# split en 70% train et 30% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print("X_train.shape: ", X_train.shape)
print("X_test.shape : ", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)


# entraînement du modèle k plus
# considérer les 3 plus proches voisins (pour un meilleur fonctionnement, prendre un nombre impair)
classifier_knn = KNeighborsClassifier(n_neighbors=3)

# entraîner le modèle
classifier_knn.fit(X_train, y_train)


# tester le modèle en utilisant la collection de test
y_pred = classifier_knn.predict(X_test)


# calculer l'accuracy du modèle en comparant les espèces prédites par le modèle 'y_pred'avec les vraies réponses 'y_test'
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# faire une prédiction sur de nouvelles données
sample = [[6, 5, 3, 2], [3, 4, 3, 5]]
preds = classifier_knn.predict(sample)
print("Prédictions:", preds)



