# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the data
data = pd.read_csv("Data.csv")

# Initial data exploration
print(data.head())
print(data.shape)
print(data.columns.tolist())
print(data.info())
print(data.describe())

# Separate numerical and categorical columns
Num_col = [col for col in data.columns if data[col].dtypes != "object"]
Cat_col = [col for col in data.columns if data[col].dtypes == "object"]

# Handling missing values
print(data.isnull().sum())
data["Age"] = data["Age"].replace(np.nan, data["Age"].median())
data["Embarked"] = data["Embarked"].replace(np.nan, "S")
print(data.isnull().sum())

# Drop unnecessary columns
data1 = data.drop(["Cabin", "Name", "Ticket"], axis=1)

# Convert data types and encode categorical variables
data1["Age"] = data1["Age"].astype(int)
data1["Sex"] = data1["Sex"].apply(lambda x: 1 if x == "male" else 0)
data1["Fare"] = data1["Fare"].astype(int)
data1['Age'] = pd.cut(x=data1['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100], labels=['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elder'])

# Encoding Embarked column
le = LabelEncoder()
data1["Embarked"] = le.fit_transform(data1["Embarked"])

# Mapping Age groups to numerical values
age_mapping = {'Infant': 0, 'Teen': 1, '20s': 2, '30s': 3, '40s': 4, '50s': 5, 'Elder': 6}
data1['Age'] = data1['Age'].map(age_mapping)

# Drop rows with NaN values in Age after mapping
data1 = data1.dropna(subset=['Age'])

# Exploratory Data Analysis
plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(2, 4, figsize=(20, 20))
sns.countplot(x="Survived", data=data1, ax=ax[0, 0])
sns.countplot(x='Pclass', data=data1, ax=ax[0, 1])
sns.countplot(x='Sex', data=data1, ax=ax[0, 2])
sns.countplot(x='Embarked', data=data1, ax=ax[1, 0])
sns.histplot(x='Fare', data=data1, bins=10, ax=ax[1, 1])
sns.countplot(x='SibSp', data=data1, ax=ax[1, 2])
sns.countplot(x='Parch', data=data1, ax=ax[1, 3])
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data1.corr(), annot=True, cmap='Blues')
plt.title("Correlation Matrix")
plt.show()

# Splitting data into training and testing sets
x = data1.drop("Survived", axis=1)
y = data1["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)

# Standard Scaler for data normalization
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)

# Model 1: Decision Tree Classifier
modelDT = DecisionTreeClassifier()
modelDT.fit(x_train_scaled, y_train)
train_pred = modelDT.predict(x_train_scaled)
print("Decision Tree Training Accuracy:", accuracy_score(train_pred, y_train))

# Model 2: K-Nearest Neighbors Classifier
modelKNN = KNeighborsClassifier(n_neighbors=3)
modelKNN.fit(x_train_scaled, y_train)
train_pred_knn = modelKNN.predict(x_train_scaled)
print("KNN Training Accuracy:", accuracy_score(train_pred_knn, y_train))

# Model 3: Logistic Regression
modelLR = LogisticRegression()
modelLR.fit(x_train_scaled, y_train)
train_pred_lr = modelLR.predict(x_train_scaled)
print("Logistic Regression Training Accuracy:", accuracy_score(train_pred_lr, y_train))

# Model 4: Support Vector Machine
modelSVM = SVC()
modelSVM.fit(x_train_scaled, y_train)
train_pred_svm = modelSVM.predict(x_train_scaled)
print("SVM Training Accuracy:", accuracy_score(train_pred_svm, y_train))

# Testing accuracy and confusion matrix for each model
models = [modelDT, modelKNN, modelLR, modelSVM]
model_names = ['Decision Tree', 'KNN', 'Logistic Regression', 'SVM']

for model, name in zip(models, model_names):
    prediction = model.predict(x_test_scaled)
    print(f"{name} Test Accuracy: {accuracy_score(prediction, y_test)}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, prediction), annot=True, cmap='Blues')
    plt.ylabel('Predicted Values')
    plt.xlabel('Actual Values')
    plt.title(f'Confusion Matrix of {name} Algorithm')
    plt.show()
