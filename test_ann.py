"""Thực hiện tuning params cho thuật toán MLPRegressor/MLPClassifier
trên bộ dữ liệu Boston Housing dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


if __name__ == "__main__":
    df = pd.read_csv(
        "C:/Users/Barbara Kieu/Downloads/data_mining/resource/BostonHousing.csv"
    )
    X = df.iloc[:, :-1]  # all columns except target
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(score)

# Tuning hồi quy, classifier
# Xác định lưới tham số để tuning
param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01],
}

# Create an MLPClassifier instance
mlp = MLPClassifier(max_iter=500)

# GridSearchCV để tìm bộ tham số tốt nhất
grid_search = GridSearchCV(mlp, param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluate the Model on Test Data
y_pred = (model.predict(X_test) > 0.5).astype(np.int32)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot
# plt.plot(best_model.loss_curve_)
# plt.title("Loss Curve")
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.show()
