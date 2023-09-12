#Kathia Bejarano Zamora
#A01378316
#Perceptron Iris

#***********************IMPORTACIÓN*******************************************************
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#*********************************PREPARACIÓN DE MIS DATOS*******************************
# Cargar el conjunto de datos importante mencionar primer mejora importaremos 
# Iris, este set de datos que se encuentra en sklearna
iris = load_iris()
X = iris.data
#Lo clasificamos binario para poder trabajar con mis set
y = (iris.target == 0).astype(int)  

# Divido los datos en conjuntos para probar validar y ejecutae
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creo el Perceptrón primero en los datos de entrenamiento
set = Perceptron(max_iter=100, alpha = .01)
set.fit(X_train, y_train)

# Realizo las predicciones en los datos de entrenamiento y prueba (importante ya etsoy usando funiciones predefinidas)
y_train_pred = set.predict(X_train)
y_test_pred = set.predict(X_test)

# Calculo el accuracy en los datos de entrenamiento y prueba
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

# Realizo una validación cruzada para obtener predicciones en múltiples conjuntos de prueba 
# (aquí viene otra mejora en comparaciín de la última entrega)
y_pred_cv = cross_val_predict(set, X, y, cv=3) 

# Calculo la varianza de las predicciones
#Este calculo se realiza a partir de una función para más fácil
variance = np.var(y_pred_cv)

# Calculo el sesgo (bias) a partir de mi formula
bias_cv = np.mean(y_pred_cv) - np.mean(y)

# Calculo el accuracy promedio
accuracy_cv = accuracy_score(y, y_pred_cv)

#****************************VISUALIZACIÓN****************************************************
#Voy a dividir en tres porque voy a desplegar 3 grafiquitas
# Creo una gráfica de barras para mostrar las métricas y evaluarlas
metricas = ['Varianza', 'Sesgo', 'Precisión Promedio']
valores = [variance, bias_cv, accuracy_cv]
# Gráfico de barraS
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.bar(metricas, valores, color=['blue', 'purple', 'pink'])
plt.xlabel('Métricas de Evaluación')
plt.ylabel('Valores')
plt.title('Métricas de Evaluación del Perceptrón')

# Gráfico de dispersión para visualizar las predicciones en función de los datos originales

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_cv, cmap=plt.cm.Paired)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Predicciones en función de los Datos Originales')

# Gráfico de barras para comparar la precisión entre el conjunto de prueba y entrenamiento
#Esta gráfica igual nos ayudará a determinar si esta overfitt fitt u underfitt
plt.subplot(1, 3, 3)
conjuntos = ['Entrenamiento', 'Prueba']
precisiones = [accuracy_train, accuracy_test]
plt.bar(conjuntos, precisiones, color=['blue', 'green'])
plt.xlabel('Conjunto de Datos')
plt.ylabel('Accuracy')
plt.title('Accuracy en Conjuntos de Entrenamiento y Prueba')

plt.tight_layout()
plt.show()
