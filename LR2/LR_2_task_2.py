import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Завантаження та обробка даних
df = pd.read_csv(r'D:\income_data.txt', header=None, skipinitialspace=True, na_values=' ?')
df.dropna(inplace=True)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Кодування категоріальних ознак
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

y = LabelEncoder().fit_transform(y)

# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Функція для тренування та оцінки моделі
def evaluate_svm(kernel_type):
    print(f"\n🔹 SVM з ядром: {kernel_type}")
    clf = SVC(kernel=kernel_type)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# 3. Дослідження різних ядер
kernels = ['poly', 'rbf', 'sigmoid']
for kernel in kernels:
    evaluate_svm(kernel)