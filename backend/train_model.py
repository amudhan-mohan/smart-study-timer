'''import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, homogeneity_score
import pickle
import os

# Dummy data and labels for simulation
X = np.array([
    [50, 10, 1],
    [45, 15, 2],
    [60, 10, 0],
    [30, 20, 5],
    [35, 25, 4],
    [20, 30, 6],
])
y_true = np.array([0, 0, 0, 2, 1, 2])

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

y_pred = gmm.predict(X)

log_likelihood = gmm.score(X) * len(X)
ari = adjusted_rand_score(y_true, y_pred)
homogeneity = homogeneity_score(y_true, y_pred)

model_path = os.path.join("backend", "gmm_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(gmm, f)

print("Model trained and saved to:", model_path)
print(f"Log Likelihood (Total): {log_likelihood:.2f}")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Homogeneity Score: {homogeneity:.4f}")'''

import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import pandas as pd
import os

# Simulated training data: [focus_time, break_time, distractions]
X = np.array([
    [50, 10, 1],
    [45, 15, 2],
    [60, 10, 0],
    [30, 20, 5],
    [35, 25, 4],
    [20, 30, 6],
])

# Fit the GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
y_pred = gmm.predict(X)

# Analyze the clusters (to know which label means what)
df = pd.DataFrame(X, columns=["focus", "break", "distractions"])
df["cluster"] = y_pred
print("Cluster Summary:")
print(df.groupby("cluster").mean())

# Save the model
model_path = os.path.join("backend", "gmm_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(gmm, f)

print("\nModel saved to:", model_path)