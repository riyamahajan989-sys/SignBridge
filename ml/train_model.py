import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("ml/dataset/sign_data.csv")

print("Dataset shape:", data.shape)
print("Number of samples:", len(data))
print("Number of signs:", data["label"].nunique())

# -----------------------------
# 2. Separate features and label
# -----------------------------
X = data.drop("label", axis=1)
y = data["label"]

# -----------------------------
# 3. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 4. Create KNN model
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# -----------------------------
# 5. Train model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 6. Test model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Save trained model
# -----------------------------
with open("ml/model/sign_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")