import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


#1 Importation du DataSet
dataset = datasets.load_iris()
#print(dataset)
print(dataset.data)
print(dataset.target_names)
#conversion dataset à un DataFrame
df = pd.DataFrame(dataset.data)
print(df.head())
#Rennomer les columns du DataFrame
df.columns = ["Longeur_Sepal", "Largeur_Sepal", "Longeur_Petal", "Largeur_Petal"]
print(df)

#Idenifier le nombre de cluster  à utiliser dans l'exemple(ELbow Method)
inertia = [] #Liste qui vas contenir les couts de notre model
K_range = range(1, 11)
for k in K_range:
    model = KMeans(n_clusters=k).fit(df)
    inertia.append(model.inertia_)
plt.plot(K_range, inertia)
plt.title('LA Méthode ELBLOW')
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele')
plt.show()


#Application de Kmeans
model = KMeans(n_clusters=3)
model.fit(df)

#Visualisation du resultat
couleurs = np.array(["Red", "green", "blue"])
plt.scatter(df.Longeur_Petal, df.Largeur_Petal, c=couleurs[model.labels_])
plt.xlabel('Longeur_Petal')
plt.ylabel('Largeur_Petal')
plt.show()