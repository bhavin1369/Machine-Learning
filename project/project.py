import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]   

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

results = {}

# Training and evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = [rmse, r2]

# Results table
results_df = pd.DataFrame(results, index=["RMSE", "R2 Score"]).T
print(results_df)

# Plot RMSE comparison
results_df["RMSE"].plot(kind="bar")
plt.title("RMSE Comparison of ML Models")
plt.ylabel("RMSE")
plt.show()

# Actual vs Predicted for best model
best_model = RandomForestRegressor(n_estimators=100)
best_model.fit(X_train, y_train)
y_best = best_model.predict(X_test)

plt.scatter(y_test, y_best)
plt.xlabel("Actual S11")
plt.ylabel("Predicted S11")
plt.title("Actual vs Predicted S11 (Random Forest)")
plt.show()
