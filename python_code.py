# Part 1: Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Part 2: Load the California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Display structure
print("First 5 rows of the dataset:")

print(df.head())

print("\n Dataset Summary:")
print(df.describe())

print("\n Missing Values Check:")
print(df.isnull().sum())

# Save original dataset
df.to_csv("california_housing_raw.csv", index=False)

# Part 3: Preprocessing & Data Cleaning

# Optional: Remove outliers (for demonstration)
df = df[df['MedHouseVal'] < 5]  # Cap the top price values

# Scale the data (optional for linear models, useful for others)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('MedHouseVal', axis=1))

X_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
X_scaled['MedHouseVal'] = df['MedHouseVal'].reset_index(drop=True)


# Part 4: Feature Engineering

df['RoomsPerHousehold'] = df['AveRooms'] / df['HouseAge']
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['HouseAge']

# Save cleaned dataset
df.to_csv("california_housing_cleaned.csv", index=False)

print("\n Feature Engineering Completed")
print("New Columns:", ['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold'])


# Part 5: Exploratory Data Analysis (EDA)
sns.set(style="whitegrid")
eda_output_dir = "eda_outputs"
os.makedirs(eda_output_dir, exist_ok=True)  # Create folder if it doesn't exist

print("\n Starting EDA...")

# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/correlation_heatmap.png")
plt.close()

# 2. Distribution of Median House Value
plt.figure(figsize=(8, 5))
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title("Distribution of Median House Value")
plt.xlabel("House Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/price_distribution.png")
plt.close()

# 3. Median Income vs Median House Value
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal')
plt.title("Median Income vs House Value")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/income_vs_price.png")
plt.close()

# 4. House Age vs Median House Value
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=pd.cut(df['HouseAge'], bins=10), y='MedHouseVal')
plt.title("House Age vs House Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/age_vs_price.png")
plt.close()

# 5. Latitude vs Median House Value
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Latitude', y='MedHouseVal', hue='MedInc', palette='viridis')
plt.title("Latitude vs House Value (colored by Income)")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/latitude_vs_price.png")
plt.close()

# 6. Longitude vs Median House Value
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Longitude', y='MedHouseVal', hue='MedInc', palette='cool')
plt.title("Longitude vs House Value (colored by Income)")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/longitude_vs_price.png")
plt.close()

# 7. Rooms per Household vs Median House Value
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='RoomsPerHousehold', y='MedHouseVal')
plt.title("Rooms per Household vs House Value")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/rooms_per_household_vs_price.png")
plt.close()

# 8. Population per Household vs Median House Value
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='PopulationPerHousehold', y='MedHouseVal')
plt.title("Population per Household vs House Value")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/population_per_household_vs_price.png")
plt.close()

# 9. Pairplot of Selected Features
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'Population', 'MedHouseVal']])
plt.savefig(f"{eda_output_dir}/pairplot_features.png")
plt.close()

# 10. KDE Plot for Income vs Price
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='MedInc', y='MedHouseVal', cmap="Reds", fill=True)
plt.title("Income vs House Value Density Plot")
plt.tight_layout()
plt.savefig(f"{eda_output_dir}/kde_income_vs_price.png")
plt.close()

print(" EDA visualizations saved in folder:", eda_output_dir)


# Part 6: Model Building & Evaluation

print("\n Starting Model Training and Evaluation...")

# Feature and target selection
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100)
}

# Results storage
results = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n {name} Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    results.append({
        'Model': name,
        'R2 Score': r2,
        'RMSE': rmse,
        'MAE': mae
    })


# Save evaluation results
results_df = pd.DataFrame(results)
results_df.to_csv("model_evaluation_results.csv", index=False)

print("\n Model evaluation results saved to model_evaluation_results.csv")

# Plot Actual vs Predicted (Random Forest as final choice)
final_model = models['Random Forest']
y_final_pred = final_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_final_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted: Random Forest")
plt.tight_layout()
plt.savefig("eda_outputs/actual_vs_predicted_rf.png")
plt.close()

print(" Actual vs Predicted plot saved as actual_vs_predicted_rf.png")



