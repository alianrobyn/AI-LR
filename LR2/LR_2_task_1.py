import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Завантаження даних
input_file = r'D:\income_data.txt'
df = pd.read_csv(input_file, header=None, na_values=' ?', skipinitialspace=True)

# 2. Видалення пропусків
df.dropna(inplace=True)

# 3. Розділяємо X (ознаки) і y (мітки)
X = df.iloc[:, :-1]  # всі колонки, крім останньої
y = df.iloc[:, -1]   # остання колонка — цільова

# 4. Кодування категоріальних ознак
label_encoders = []
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders.append((col, le))

# Кодуємо цільову змінну (<=50K → 0, >50K → 1)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# 5. Масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Створення та тренування SVM-класифікатора
svm_clf = SVC(kernel='rbf')  # Можна змінити kernel на 'linear', 'poly', тощо
svm_clf.fit(X_train, y_train)

# 8. Прогнозування та оцінка
y_pred = svm_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_encoder.classes_))