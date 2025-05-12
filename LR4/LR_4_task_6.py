import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, title):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error', random_state=42
    )

    train_errors_mean = -np.mean(train_scores, axis=1)
    val_errors_mean = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors_mean, "o-", label="Навчальна помилка")
    plt.plot(train_sizes, val_errors_mean, "o-", label="Перевірочна помилка")
    plt.title(title)
    plt.xlabel("Кількість навчальних прикладів")
    plt.ylabel("Середньоквадратична помилка")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Генерація даних
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 4 + np.sin(X).ravel() + np.random.uniform(-0.6, 0.6, m)

# Лінійна регресія
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання для лінійної регресії")

# Поліноміальна регресія (ступінь 2)
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_features_2.fit_transform(X)
poly_reg_2 = LinearRegression()
plot_learning_curves(poly_reg_2, X_poly_2, y, "Криві навчання для поліноміальної регресії (ступінь 2)")

# Поліноміальна регресія (ступінь 10)
poly_features_10 = PolynomialFeatures(degree=10, include_bias=False)
X_poly_10 = poly_features_10.fit_transform(X)
poly_reg_10 = LinearRegression()
plot_learning_curves(poly_reg_10, X_poly_10, y, "Криві навчання для поліноміальної регресії (ступінь 10)")
