import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Завантаження даних про діабет
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розподіл на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення та навчання лінійної моделі
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Прогнозування
y_pred = regr.predict(X_test)

# Вивід коефіцієнтів
print("Коефіцієнти регресії:", regr.coef_)
print("Вільний член (intercept):", regr.intercept_)

# Метрики якості
print("R2 score:", r2_score(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# Побудова графіка
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Виміряно (y_test)')
ax.set_ylabel('Передбачено (y_pred)')
ax.set_title('Лінійна регресія: передбачене vs реальне')
plt.grid(True)
plt.show()
