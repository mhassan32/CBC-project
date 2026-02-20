import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


# Load Data

df = pd.read_csv("cbc_dataset.csv")
df.columns = df.columns.str.lower()


# Data Cleaning

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=['number']).columns
 

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)



for col in numeric_cols:
    df = df[df[col] >= 0]


# Feature Selection


target_cols = ['hb', 'rbc', 'wbc', 'platelets', 'lymp', 'mono', 'hct', 'mcv', 
               'mch', 'mchc', 'rdw', 'pdw', 'mpv', 'pct']

X = df.drop(['diagnosis'] + target_cols, axis=1)  
y = df[target_cols]                               


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


#  Feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# Train Multi-output Linear Regression

lr_model = MultiOutputRegressor(LinearRegression())
lr_model.fit(X_train_scaled, y_train)


# Predictions

y_pred = lr_model.predict(X_test_scaled)
y_pred = pd.DataFrame(y_pred, columns=y.columns)

# Evaluation

print("Linear Regression Metrics for all numeric targets:\n")
metrics = []

for col in target_cols:
    mse = mean_squared_error(y_test[col], y_pred[col])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test[col], y_pred[col])
    metrics.append([col, mse, rmse, r2])
    print(f"{col}: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")

metrics_df = pd.DataFrame(metrics, columns=['Target', 'MSE', 'RMSE', 'R2'])
print("\nSummary Table:\n", metrics_df)

# Visualization
# Example scatter for hb


plt.figure(figsize=(8,6))
plt.scatter(y_test['hb'], y_pred['hb'], alpha=0.5)
plt.plot([y_test['hb'].min(), y_test['hb'].max()],
         [y_test['hb'].min(), y_test['hb'].max()],
         'r--', lw=2)
plt.xlabel("Actual Hb")
plt.ylabel("Predicted Hb")
plt.title("Linear Regression: Actual vs Predicted Hb")
plt.show()