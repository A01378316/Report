#Kathia Bejarano Zamora
#A01378316
#Perceptron Iris

#***********************IMPORTACIÓN*******************************************************
import numpy as np
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve


#*********************************PREPARACIÓN DE MIS DATOS*******************************
# Cargar el conjunto de datos IRIS
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

# Dividir los datos en conjuntos de entrenamiento, prueba y validación
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#*********************************MODELO PERCEPTRÓN*******************************
# Crear un objeto Perceptrón y entrenarlo en el conjunto de entrenamiento
perceptron = Perceptron(max_iter=100, alpha=0.01)
perceptron.fit(X_train, y_train)

# Realizar predicciones en los datos de entrenamiento, prueba y validación
y_train_pred = perceptron.predict(X_train)
y_test_pred = perceptron.predict(X_test)
y_val_pred = perceptron.predict(X_val)

# Calcular el accuracy en los datos de entrenamiento, prueba y validación
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
accuracy_val = accuracy_score(y_val, y_val_pred)

# Realizar una validación cruzada para obtener predicciones en múltiples conjuntos de prueba
y_pred_cv = cross_val_predict(perceptron, X, y, cv=3)

# Calcular la varianza de las predicciones
variance = np.var(y_pred_cv)

# Calcular el sesgo (bias) a partir de la fórmula
bias_cv = np.mean(y_pred_cv) - np.mean(y)
bias_train = np.mean(y_train_pred) - np.mean(y_train)  # Calcular el sesgo en el conjunto de entrenamiento
bias_val = np.mean(y_val_pred) - np.mean(y_val)
bias_test = np.mean(y_test_pred) - np.mean(y_test)

# Calcular la varianza en los conjuntos de entrenamiento, prueba y validación
variance_train = np.var(y_train_pred)
variance_test = np.var(y_test_pred)
variance_val = np.var(y_val_pred)

# Crear subplots para mostrar las seis gráficas en una pantalla
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))


#*********************************VISUALIZACIÓN*******************************
# Gráfica 1: Sesgo en el conjunto de prueba
axes[0, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap=plt.cm.Paired)
axes[0, 0].set_xlabel('Característica 1')
axes[0, 0].set_ylabel('Característica 2')
axes[0, 0].set_title(f'Sesgo en Conjunto de Prueba: {bias_test:.2f}')

# Gráfica 2: Sesgo en el conjunto de validación
axes[0, 1].scatter(X_val[:, 0], X_val[:, 1], c=y_val_pred, cmap=plt.cm.Paired)
axes[0, 1].set_xlabel('Característica 1')
axes[0, 1].set_ylabel('Característica 2')
axes[0, 1].set_title(f'Sesgo en Conjunto de Validación: {bias_val:.2f}')

# Gráfica 3: Sesgo en el conjunto de entrenamiento
axes[0, 2].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap=plt.cm.Paired)
axes[0, 2].set_xlabel('Característica 1')
axes[0, 2].set_ylabel('Característica 2')
axes[0, 2].set_title(f'Sesgo en Conjunto de Entrenamiento: {bias_train:.2f}')

# Gráfica 4: Varianza en el conjunto de prueba (gráfico de barras con valor)
labels = ['Entrenamiento', 'Prueba', 'Validación']
variances = [variance_train, variance_test, variance_val]
axes[1, 0].bar(labels, variances, color=['blue', 'green', 'red'])
axes[1, 0].set_ylabel('Varianza')
axes[1, 0].set_title('Varianza en Diferentes Conjuntos de Datos')

# Agregar valores de varianza en las barras
for bar, variance_value in zip(axes[1, 0].patches, variances):
    axes[1, 0].annotate(f'{variance_value:.2f}', (bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01),
                        fontsize=12)

# Gráfico 5: Accuracy en Conjuntos de Entrenamiento y Prueba (gráfico de barras)
conjuntos = ['Entrenamiento', 'Prueba', 'Validación']
precisiones = [accuracy_train, accuracy_test, accuracy_val]
axes[1, 1].bar(conjuntos, precisiones, color=['blue', 'green', 'red'])
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Accuracy en Conjuntos de Entrenamiento, Prueba y Validación')

# Gráfico 6: Curva de Aprendizaje
train_sizes, train_scores, test_scores = learning_curve(perceptron, X_train, y_train, cv=3, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

axes[1, 2].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
axes[1, 2].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
axes[1, 2].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
axes[1, 2].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación")
axes[1, 2].set_xlabel('Tamaño del Conjunto de Entrenamiento')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].set_title('Curva de Aprendizaje')
axes[1, 2].legend(loc="best")

# Agregar títulos generales a las gráficas
plt.suptitle('Sesgo, Varianza, Accuracy y Curva de Aprendizaje en Diferentes Conjuntos de Datos')

plt.tight_layout()

# Mostrar las gráficas
plt.show()
