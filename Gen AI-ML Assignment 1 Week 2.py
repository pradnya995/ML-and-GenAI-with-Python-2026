# Netflix User Analytics

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Q1 - Load dataset and display first 5 records

data = pd.read_csv("netflix_users.csv")
print("First 5 Records")
print(data.head())

# Q2 - Number of rows and columns

print("\nRows and Columns")
print(data.shape)

# Q3 - Column names

print("\nColumn Names")
print(data.columns)

# Q4 - Numerical and Categorical Features

print("\nNumerical Features")
print(data.select_dtypes(include=['int64','float64']).columns)

print("\nCategorical Features")
print(data.select_dtypes(include=['object', 'str']).columns)

# Q5 - Missing Values

print("\nMissing Values")
print(data.isnull().sum())

# Q6 - Average Age

print("\nAverage Age")
print(data["Age"].mean())

# Q7 - Average Watch Hours Per Week

print("\nAverage Watch Hours")
print(data["WatchHoursPerWeek"].mean())

# Q8 - Average Monthly Spending

print("\nAverage Monthly Spend")
print(data["MonthlySpend"].mean())

# Q9 - Users in Each Subscription Category

print("\nSubscription Counts")
print(data["SubscriptionType"].value_counts())

# Q10 - Percentage of Renewed Users

renewed = (data["SubscriptionRenewed"] == "Yes").sum()
total = len(data)

print("\nRenewal Percentage")
print((renewed / total) * 100)

# Q11 - Convert Categorical Data to Numerical

le = LabelEncoder()

data["Gender"] = le.fit_transform(data["Gender"])
data["SubscriptionType"] = le.fit_transform(data["SubscriptionType"])
data["FavoriteGenre"] = le.fit_transform(data["FavoriteGenre"])
data["SubscriptionRenewed"] = le.fit_transform(data["SubscriptionRenewed"])

# Q12 - Define X and y for Renewal Prediction

X = data.drop(["UserID", "SubscriptionRenewed"], axis=1)
y = data["SubscriptionRenewed"]

# Q13 - Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Q14 - Decision Tree Model

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Q15 - Accuracy of Decision Tree

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy")
print(accuracy_score(y_test, dt_pred))

# Q16 - Confusion Matrix

print("\nDecision Tree Confusion Matrix")
print(confusion_matrix(y_test, dt_pred))

# Q17 - KNN with K = 5

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

# Q18 - Compare Accuracy

dt_acc = accuracy_score(y_test, dt_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy")
print(knn_acc)

print("\nDecision Tree Accuracy")
print(dt_acc)

if knn_acc > dt_acc:
    print("KNN performed better")
else:
    print("Decision Tree performed better")

# Q19 - Linear Regression for Monthly Spending

X2 = data.drop(["UserID", "MonthlySpend"], axis=1)
y2 = data["MonthlySpend"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X2_train, y2_train)

# Q20 - Predict Monthly Spending for New User

new_user_data = [[25, 1, 2, 15, 3, 1, 10, 1]]
new_user_df = pd.DataFrame(new_user_data, columns=X2_train.columns)
spend = lr.predict(new_user_df)

print("\nPredicted Monthly Spending")
print(spend[0])
