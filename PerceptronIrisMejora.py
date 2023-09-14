import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

# Cargar el conjunto de datos Iris de ejemplo
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

# Parte nueva del código que incluye lo que se conoce como bootstrap
n_replicas = 100
y_pred_replicas = []

for _ in range(n_replicas):
    indices_bootstrap = np.random.choice(len(X), len(X), replace=True)
    X_bootstrap = X[indices_bootstrap]
    y_bootstrap = y[indices_bootstrap]
    
    perceptron = Perceptron(random_state=42, max_iter=1000, alpha=0.001)
    perceptron.fit(X_bootstrap, y_bootstrap)
    
    y_pred = perceptron.predict(X)
    y_pred_replicas.append(y_pred)

# Calculo la varianza de las predicciones 
variance_bootstrap = np.var(y_pred_replicas, axis=0)

# Definir una cuadrícula de hiperparámetros para buscar
param_grid = {
    'max_iter': [100, 500, 1000],
    'alpha': [0.001, 0.01, 0.1]
}

# Crear un objeto Perceptrón
perceptron = Perceptron()

# Realizar una búsqueda en cuadrícula con validación cruzada
grid_search = GridSearchCV(perceptron, param_grid, cv=3)
grid_search.fit(X, y)

# Obtener los mejores hiperparámetros
best_max_iter = grid_search.best_params_['max_iter']
best_alpha = grid_search.best_params_['alpha']

# Crear un objeto Perceptrón con los mejores hiperparámetros
best_perceptron = Perceptron(max_iter=best_max_iter, alpha=best_alpha)
best_perceptron.fit(X, y)

# Calcular la curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(best_perceptron, X, y, cv=3)

# Calcular las medias y desviaciones estándar de las puntuaciones
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Crear la gráfica de la curva de aprendizaje mejorada
plt.figure(figsize=(10, 6))
plt.title("Curva de Aprendizaje")
plt.xlabel("Tamaño del Conjunto de Entrenamiento")
plt.ylabel("Accuracy")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación")

plt.legend(loc="best")

# Visualización de resultados
# (Puedes usar los mismos gráficos mejorados que proporcionaste originalmente)

# Creo una gráfica de barras mejorada para mostrar las métricas
metricas_bootstrap = ['Varianza', 'Sesgo', 'Precisión Promedio']
valores_bootstrap = [np.mean(variance_bootstrap), np.mean(bias_bootstrap), accuracy_avg_bootstrap]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Creo una gráfica de barras mejorada para mostrar las métricas
metricas_bootstrap = ['Varianza', 'Sesgo', 'Precisión Promedio']
valores_bootstrap = [np.mean(variance_bootstrap), np.mean(bias_bootstrap), accuracy_avg_bootstrap]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico de barras mejorado
axes[0, 0].bar(metricas_bootstrap, valores_bootstrap, color=['blue', 'green', 'pink'], alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 0].set_xlabel('Métricas de Evaluación (Bootstrap)')
axes[0, 0].set_ylabel('Valores')
axes[0, 0].set_title('Métricas de Evaluación del Perceptrón (Bootstrap)')

# Gráfico de dispersión mejorado para visualizar las predicciones en función de los datos originales
scatter = axes[0, 1].scatter(X[:, 0], X[:, 1], c=np.mean(y_pred_replicas, axis=0), cmap=plt.cm.Paired, edgecolor='black', linewidth=0.5)
axes[0, 1].set_xlabel('Característica 1')
axes[0, 1].set_ylabel('Característica 2')
axes[0, 1].set_title('Predicciones en función de los Datos Originales (Bootstrap)')

# Agregar una barra de colores
cbar = fig.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('Predicciones Promedio', rotation=270, labelpad=20)

# Gráfico de contorno mejorado para mostrar la frontera de decisión
axes[1, 0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='black', linewidth=0.5)
axes[1, 0].contourf(X[:, 0], X[:, 1], np.mean(y_pred_replicas, axis=0), cmap=plt.cm.Paired, alpha=0.6)
axes[1, 0].set_xlabel('Característica 1')
axes[1, 0].set_ylabel('Característica 2')
axes[1, 0].set_title('Frontera de Decisión del Perceptrón (Bootstrap)')

# Gráfico de dispersión mejorado para mostrar la distribución de clases originales
axes[1, 1].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='x', s=200, edgecolors='black', linewidths=1.5)
axes[1, 1].set_xlabel('Característica 1')
axes[1, 1].set_ylabel('Característica 2')
axes[1, 1].set_title('Distribución de Clases Originales')

# Curva de Aprendizaje
# ...

# Curva de Aprendizaje
train_sizes, train_scores, test_scores = learning_curve(best_perceptron, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación")
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Accuracy')
plt.title('Curva de Aprendizaje')
plt.legend(loc="best")

plt.tight_layout()

# Mostrar la gráfica de la curva de aprendizaje
plt.show()
