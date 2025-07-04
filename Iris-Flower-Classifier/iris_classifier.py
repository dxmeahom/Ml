from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()

# Convert to a pandas DataFrame for easier handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head())

import matplotlib.pyplot as plt

# Map species numbers to names for easier plotting
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_map)

# Make a scatter plot of petal length vs petal width, colored by species
plt.figure(figsize=(8, 6))
for species in df['species_name'].unique():
    subset = df[df['species_name'] == species]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], label=species)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Petal Length vs Width by Species')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Features (X) are the measurements, labels (y) are the species
X = df[iris.feature_names]
y = df['species']

# Split the data: 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the KNN model (let's use 3 neighbors)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict the species for the test set
y_pred = knn.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Let's predict the species for some new, made-up flower measurements
# Each list is: [sepal length, sepal width, petal length, petal width]
new_samples = [
    [5.1, 3.5, 1.4, 0.2],  # likely setosa
    [6.0, 2.2, 4.0, 1.0],  # likely versicolor
    [6.9, 3.1, 5.4, 2.1]   # likely virginica
]

predictions = knn.predict(new_samples)

# Map the predicted numbers back to species names
predicted_species = [species_map[p] for p in predictions]

for i, sample in enumerate(new_samples):
    print(f"Sample {sample} is predicted to be: {predicted_species[i]}")