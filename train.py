import numpy as np
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Fix randomness (so results stay same every time)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 2. Load dataset
data = load_breast_cancer()
X = data.data   # features
y = data.target # labels

# 3. Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

# 4. Scale features (important!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create model
model = LogisticRegression(max_iter=500, random_state=SEED)

# 6. Train model
model.fit(X_train, y_train)

# 7. Predict
predictions = model.predict(X_test)

# 8. Evaluate
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("F1 Score:", f1)

# 9. Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))