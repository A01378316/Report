
#Kathia Bejarano Zamora
#A01378316
#Perceptron Iris en teoría mejorado

#***********************IMPORTACIÓN*******************************************************
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#*********************************PREPARACIÓN DE MIS DATOS*******************************
# Cargar el conjunto de datos Iris de ejemplo
#Este conjunto sirve mucho debido a su composición pero simplicidad al mismo tiempo
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

#la parte nueva del código incluye lo que se puede conocer como bootstrap
#Vamos a hacer muchas replicas para entrenar más 
n_replicas = 100

# Almaceno las predicciones de cada réplica
y_pred_replicas = []

# Creo y entreno un Perceptrón para cada réplica
for _ in range(n_replicas):
    # Creo un conjunto de entrenamiento
    indices_bootstrap = np.random.choice(len(X), len(X), replace=True)
    X_bootstrap = X[indices_bootstrap]
    y_bootstrap = y[indices_bootstrap]
    
    set = Perceptron(random_state=42,max_iter = 1000, alpha =.001) #Random state mejora
    set.fit(X_bootstrap, y_bootstrap)
    
    # Realizo predicciones en el conjunto original
    y_pred = set.predict(X)
    y_pred_replicas.append(y_pred)

# Calculo la varianza de las predicciones 
variance_bootstrap = np.var(y_pred_replicas, axis=0)
print("LA VARIANZAAAAAAA")
print(variance_bootstrap)
print(np.mean(variance_bootstrap))

# Calculo el sesgo (bias) promedio de todo
bias_bootstrap = np.mean(y_pred_replicas, axis=0) - np.mean(y)

# Calculo la precisión promedio 
accuracy_bootstrap = [accuracy_score(y, y_pred) for y_pred in y_pred_replicas]
accuracy_avg_bootstrap = np.mean(accuracy_bootstrap)

#*******************************VISUALIZACIÓN************************
# Creo una gráfica de barras para mostrar las métricas
metricas_bootstrap = ['Varianza', 'Sesgo', 'Precisión Promedio']
valores_bootstrap = [np.mean(variance_bootstrap), np.mean(bias_bootstrap), accuracy_avg_bootstrap]

# Gráfico de barras
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(metricas_bootstrap, valores_bootstrap, color=['blue', 'green', 'pink'])
plt.xlabel('Métricas de Evaluación (Bootstrap)')
plt.ylabel('Valores')
plt.title('Métricas de Evaluación del Perceptrón (Bootstrap)')

# Gráfico de dispersión para visualizar las predicciones en función de los datos originales
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=np.mean(y_pred_replicas, axis=0), cmap=plt.cm.Paired)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Predicciones en función de los Datos Originales (Bootstrap)')

plt.tight_layout()
plt.show()
