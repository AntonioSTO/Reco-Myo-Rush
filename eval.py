import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

data = np.load("features.npz", allow_pickle=True)
X, y = data["X"], data["y"]

model, scaler, le = joblib.load("model.pkl")

X = scaler.transform(X)
y = le.transform(y)

yp = model.predict(X)

print("Accuracy:", accuracy_score(y, yp))
print(classification_report(y, yp))
