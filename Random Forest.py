import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 6)  
V_true = 0.9 + 0.2 * np.random.rand(n_samples)  
theta_true = -np.pi/6 + (np.pi/3) * np.random.rand(n_samples)  

df = pd.DataFrame(X, columns=[f"meas_{i+1}" for i in range(X.shape[1])])
df['V'] = V_true
df['theta'] = theta_true

features = df.drop(['V', 'theta'], axis=1)
target_V = df['V']
target_theta = df['theta']


X_train, X_test, y_train_V, y_test_V = train_test_split(features, target_V, test_size=0.2, random_state=1)
_, _, y_train_theta, y_test_theta = train_test_split(features, target_theta, test_size=0.2, random_state=1)

rf_v = RandomForestRegressor(n_estimators=100, random_state=0)
rf_v.fit(X_train, y_train_V)
V_pred = rf_v.predict(X_test)

rf_theta = RandomForestRegressor(n_estimators=100, random_state=0)
rf_theta.fit(X_train, y_train_theta)
theta_pred = rf_theta.predict(X_test)


print("Voltage Prediction:")
print("R²:", r2_score(y_test_V, V_pred))
print("MSE:", mean_squared_error(y_test_V, V_pred))

print("\nPhase Angle Prediction:")
print("R²:", r2_score(y_test_theta, theta_pred))
print("MSE:", mean_squared_error(y_test_theta, theta_pred))

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.scatter(y_test_V, V_pred, alpha=0.6)
plt.xlabel("True V"); plt.ylabel("Predicted V")
plt.title("Voltage Prediction")

plt.subplot(1, 2, 2)
plt.scatter(y_test_theta, theta_pred, alpha=0.6)
plt.xlabel("True θ"); plt.ylabel("Predicted θ")
plt.title("Angle Prediction")

plt.tight_layout()
plt.show()

