#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

#Load and prepare the data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# View structure
print(df.head())
print(df.describe())

#EDA- Exploratory Data Analysis
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']])
plt.show()

#Feature Selection and splitting
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

#Build Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Intercept term
print("\nIntercept:", model.intercept_)

comparison_df = pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_pred[:10]})
print(comparison_df)

#Evaluate the model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#Visualize predictions
plt.scatter(y_test, y_pred)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')  # perfect prediction line
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Housing Prices")
plt.show()
