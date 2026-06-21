# Week 3 Assignment 1
# Exploratory Data Analysis (EDA) and Machine Learning
# Agricultural Yield Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("agricultural_yield_dataset.csv")

# =============================
# Q1. Dataset Overview
# =============================
print("Q1")
print("Rows and Columns :", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nFirst 10 Records:")
print(df.head(10))

# =============================
# Q2. Data Types and Missing Values
# =============================
print("\nQ2")
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# =============================
# Q3. Descriptive Statistics
# =============================
print("\nQ3")
print(df.describe())

num_cols = df.select_dtypes(include=['int64', 'float64'])

print("\nFeature with Highest Mean:")
print(num_cols.mean().idxmax())

print("\nFeature with Highest Standard Deviation:")
print(num_cols.std().idxmax())

# =============================
# Q4. Distribution Analysis
# =============================
print("\nQ4")

df["rainfall_mm"].hist()
plt.title("Rainfall Distribution")
plt.show()

df["temperature_c"].hist()
plt.title("Temperature Distribution")
plt.show()

df["fertilizer_kg"].hist()
plt.title("Fertilizer Distribution")
plt.show()

df["yield_ton_per_hectare"].hist()
plt.title("Yield Distribution")
plt.show()

# =============================
# Q5. Crop Type Analysis
# =============================
print("\nQ5")

print(df["crop_type"].value_counts())

sns.countplot(x="crop_type", data=df)
plt.title("Crop Type Count")
plt.show()

print("\nMost Frequent Crop:")
print(df["crop_type"].value_counts().idxmax())

# =============================
# Q6. Soil Type Analysis
# =============================
print("\nQ6")

print(df["soil_type"].value_counts())

sns.countplot(x="soil_type", data=df)
plt.title("Soil Type Count")
plt.show()

print("\nMost Common Soil Type:")
print(df["soil_type"].value_counts().idxmax())

# =============================
# Q7. Yield Distribution
# =============================
print("\nQ7")

df["yield_ton_per_hectare"].hist()
plt.title("Yield Distribution")
plt.show()

# Observe histogram and write:
# 1. Whether it looks normal
# 2. Whether outliers are present

# =============================
# Q8. Scatter Plot Analysis
# =============================
print("\nQ8")

plt.scatter(df["rainfall_mm"], df["yield_ton_per_hectare"])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.scatter(df["fertilizer_kg"], df["yield_ton_per_hectare"])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()

# Compare both plots and write which looks stronger

# =============================
# Q9. Correlation Analysis
# =============================
print("\nQ9")

corr_matrix = num_cols.corr()

print(corr_matrix)

sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()

yield_corr = corr_matrix["yield_ton_per_hectare"].sort_values(ascending=False)

print("\nTop Correlations with Yield:")
print(yield_corr)

# =============================
# Q10. Group-Based Analysis
# =============================
print("\nQ10")

crop_yield = df.groupby("crop_type")["yield_ton_per_hectare"].mean()
print("\nAverage Yield by Crop:")
print(crop_yield)

soil_yield = df.groupby("soil_type")["yield_ton_per_hectare"].mean()
print("\nAverage Yield by Soil:")
print(soil_yield)

print("\nCrop with Highest Average Yield:")
print(crop_yield.idxmax())

print("\nSoil Type with Highest Average Yield:")
print(soil_yield.idxmax())

# =============================
# Q11. Feature Encoding
# =============================
print("\nQ11")

categorical_cols = df.select_dtypes(include=["object"]).columns

print("\nCategorical Columns:")
print(categorical_cols)

df_encoded = pd.get_dummies(df, columns=categorical_cols)

print("\nFirst 5 Rows After Encoding:")
print(df_encoded.head())

# =============================
# Q12. Feature Selection
# =============================
print("\nQ12")

X = df_encoded.drop("yield_ton_per_hectare", axis=1)
y = df_encoded["yield_ton_per_hectare"]

print("\nTarget Variable:")
print("yield_ton_per_hectare")

# =============================
# Q13. Train-Test Split
# =============================
print("\nQ13")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("X_train Shape :", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape :", y_test.shape)

# =============================
# Q14. Linear Regression Model
# =============================
print("\nQ14")

model = LinearRegression()

model.fit(X_train, y_train)

print("\nIntercept:")
print(model.intercept_)

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nCoefficients:")
print(coeff_df)

highest_feature = coeff_df.loc[
    coeff_df["Coefficient"].idxmax(),
    "Feature"
]

print("\nFeature with Highest Positive Coefficient:")
print(highest_feature)
