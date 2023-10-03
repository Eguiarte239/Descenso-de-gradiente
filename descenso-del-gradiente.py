import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar los datos desde el archivo CSV
df = pd.read_csv('salary.csv')

# Extraer las columnas YearsExperience y Salary como arreglos NumPy
years_experience = np.array(df['YearsExperience'])
salary = np.array(df['Salary'])

# Definir la función de pérdida basada en tus datos
def loss_function(theta0, theta1):
    hypothesis = theta0 + theta1 * years_experience
    loss = np.mean((hypothesis - salary) ** 2)
    return loss

# Inicializar los parámetros theta0 y theta1
theta0 = np.float64(0.0)
theta1 = np.float64(0.0)

# Hiperparámetros del descenso del gradiente
learning_rate = 0.01
num_iterations = 100

# Almacenar los valores para la gráfica
theta0_history = [theta0]
theta1_history = [theta1]
loss_history = [loss_function(theta0, theta1)]

# Realizar el descenso del gradiente
for _ in range(num_iterations):
    gradient0 = np.mean(theta0 + theta1 * years_experience - salary)
    gradient1 = np.mean((theta0 + theta1 * years_experience - salary) * years_experience)
    
    theta0 -= learning_rate * gradient0
    theta1 -= learning_rate * gradient1
    
    loss = loss_function(theta0, theta1)
    
    theta0_history.append(theta0)
    theta1_history.append(theta1)
    loss_history.append(loss)

# Imprimir los parámetros óptimos
print("Parámetros óptimos (theta0, theta1):", (theta0, theta1))

# Graficar la función de pérdida
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

theta0_vals = np.linspace(-1, 1, 100)
theta1_vals = np.linspace(-1, 1, 100)
theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
loss_grid = np.zeros_like(theta0_grid, dtype=float)

for i in range(100):
    for j in range(100):
        loss_grid[i, j] = loss_function(theta0_grid[i, j], theta1_grid[i, j])

ax.plot_surface(theta0_grid, theta1_grid, loss_grid, cmap='viridis')
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Loss')

plt.title('Función de Pérdida')
plt.show()

# Graficar la trayectoria del descenso del gradiente
plt.figure()
plt.contour(theta0_vals, theta1_vals, loss_grid, levels=50, cmap='viridis')
plt.scatter(theta0_history, theta1_history, c='red', marker='x', label='Descenso del Gradiente')
plt.xlabel('Theta0')
plt.ylabel('Theta1')
plt.title('Trayectoria del Descenso del Gradiente')
plt.legend()
plt.show()