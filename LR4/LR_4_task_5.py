import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Вхідні дані для варіанта 10
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 4 + np.sin(X).ravel() + np.random.uniform(-0.6, 0.6, m)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Поліноміальна регресія (ступінь 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Перевірка ознак після трансформації
print("Перші 5 трансформованих X_poly:")
print(X_poly[:5])

# Навчання на поліноміальних ознаках
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Коефіцієнти поліноміальної моделі
intercept = poly_reg.intercept_
coef = poly_reg.coef_
print("\nКоефіцієнти поліноміальної моделі:")
print(f"intercept = {intercept:.2f}, coef = {coef}")

# Побудова графіків
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_lin_pred, color='green', linewidth=2, label='Лінійна регресія')
plt.plot(X, y_poly_pred, color='red', linewidth=2, label='Поліноміальна регресія (ступінь 2)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Лінійна vs Поліноміальна регресія')
plt.grid(True)
plt.show()
