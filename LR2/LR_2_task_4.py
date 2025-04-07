import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Класифікатори
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Завантаження та підготовка даних
df = pd.read_csv(r'D:\income_data.txt', header=None, skipinitialspace=True, na_values=' ?')
df.dropna(inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Кодування категорій
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

y = LabelEncoder().fit_transform(y)

# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Класифікатори
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree (CART)": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF Kernel)": SVC(kernel='rbf')
}

# 3. Тренування та оцінка
results = {}

for name, model in models.items():
    print(f"\n🔹 {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# 4. Візуалізація результатів
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.ylabel('Accuracy')
    plt.title('Порівняння якості класифікаторів')
    plt.xticks(rotation=45)
    plt.ylim(0.7, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

