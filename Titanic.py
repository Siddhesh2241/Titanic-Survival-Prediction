import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Data.csv")
data.head()
data.shape
data.columns.tolist()
Num_col = [col for col in data.columns if data[col].dtypes != "object"] 
Cat_col = [col for col in data.columns if data[col].dtypes == "object"]
data.info()
data.describe()
data.duplicated()
data.nunique()
data.isnull().sum()
data["Age"] = data["Age"].replace(np.nan,data["Age"].median(axis=0))
data["Embarked"] = data["Embarked"].replace(np.nan,"S")

data.isnull().sum()
data1 = data.drop(["Cabin","Name","Ticket"],axis =1)
data1.isnull().sum()
data1["Age"] = data["Age"].astype(int)
data1["Sex"] = data1["Sex"].apply(lambda x : 1 if x == "male" else 0)
data1["Fare"] = data["Fare"].astype(int)
data1.dtypes

plt.figure(figsize=(20,20))
fig,ax=plt.subplots(2,4,figsize=(20,20))
sns.countplot(x="Survived",data=data1,ax=ax[0,0])
sns.countplot(x = 'Pclass', data = data1, ax=ax[0,1])
sns.countplot(x = 'Sex', data = data1, ax=ax[0,2])
sns.countplot(x = 'Age', data = data1, ax=ax[0,3])
sns.countplot(x = 'Embarked', data = data1, ax=ax[1,0])
sns.histplot(x = 'Fare', data= data1, bins=10, ax=ax[1,1])
sns.countplot(x = 'SibSp', data = data1, ax=ax[1,2])
sns.countplot(x = 'Parch', data = data1, ax=ax[1,3])

## Ensure the Age column is numeric and handle non-numeric entries
data1['Age'] = pd.to_numeric(data1['Age'], errors='coerce')
# creating age groups - young (0-18), adult(18-30), middle aged(30-50), old (50-100)
data1['Age'] = pd.cut(x=data1['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100], labels = ['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elder'])

plt.figure(figsize=(6,6))
sns.countplot(x = 'Age', data = data1)
plt.show()

plt.figure(figsize=(20,20))
fig,ax=plt.subplots(2,3,figsize=(20,20))
sns.countplot(x = 'Sex', data = data1, hue = 'Survived', ax= ax[0,0])
sns.countplot(x = 'Age', data = data1, hue = 'Survived', ax=ax[0,1])
sns.countplot(x= "Pclass",data = data1, hue="Survived", ax=ax[0,2])
sns.countplot(x = 'Parch', data = data1, hue = 'Survived', ax=ax[1,0])
sns.scatterplot(x = 'SibSp', y = 'Parch', data = data1, hue = 'Survived', ax=ax[1,1])
sns.pointplot(x = 'Pclass', y = 'Survived', data = data1, ax=ax[1,2])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data1["Embarked"] = le.fit_transform(data1["Embarked"])

data1.dtypes

# Here could not convert string to float, so we do mapping

age_mapping = {
    'infant': 0,
    'teen': 1,
    '20s': 2,
    '30s': 3,
    '40s': 4,
    '50s': 5,
    'elder': 6}
data1['Age'] = data1['Age'].map(age_mapping)
data1.dropna(subset=['Age'], axis= 0)

plt.figure(figsize=(12,6))
sns.heatmap(data1.corr(),annot = True,color="Blue")
plt.title("Corelation of matrix")
plt.show()

x = data1.drop("Survived",axis=1)
y = data1["Survived"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state=10)

from sklearn.tree import DecisionTreeClassifier
modelDT = DecisionTreeClassifier()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
x_scale = scale.fit_transform(x_train)

modelDT.fit(x_train,y_train)

train_pred = modelDT.predict(x_train)
from sklearn.metrics import accuracy_score
accuracy_score(train_pred,y_train)

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors = 3)
scale = StandardScaler()
x_scale = scale.fit_transform(x_train)
np.isnan(x_scale).sum() # These checks ensure there are no hidden NaNs
np.isinf(x_scale).sum()
# # Replace NaNs or Infinities with zero
x_train_scaled = np.nan_to_num(x_scale, nan=0.0, posinf=0.0, neginf=0.0) 

modelKNN.fit(x_train_scaled,y_train)
Train_pred = modelKNN.predict(x_train_scaled)
accuracy_score(Train_pred,y_train)

from sklearn.linear_model import LogisticRegression
modelLR = LogisticRegression()
scale = StandardScaler()
x_scale = scale.fit_transform(x_train)

np.isnan(x_scale).sum() # These checks ensure there are no hidden NaNs

np.isinf(x_scale).sum()

# # Replace NaNs or Infinities with zero
x_train_scaled = np.nan_to_num(x_scale, nan=0.0, posinf=0.0, neginf=0.0) 

modelLR.fit(x_train_scaled,y_train)

train_pred = modelLR.predict(x_train_scaled)
accuracy_score(train_pred,y_train)

from sklearn.svm import SVC
modelSVM = SVC()
scale = StandardScaler()
x_scale = scale.fit_transform(x_train)
np.isnan(x_scale).sum() # These checks ensure there are no hidden NaNs
# # Replace NaNs or Infinities with zero
x_train_scaled = np.nan_to_num(x_scale, nan=0.0, posinf=0.0, neginf=0.0) 
modelSVM.fit(x_train_scaled,y_train)
train_pred = modelSVM.predict(x_train_scaled)
accuracy_score(train_pred,y_train)

Prediction = modelDT.predict(x_test)
accuracy_score(Prediction,y_test)

from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(Prediction,y_test),annot= True, cmap = 'Blues')
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.title('confusion matrix of Decesion tree Algorithm')
plt.show()

## KNeighborsClassifier

scale = StandardScaler()
x_scale = scale.fit_transform(x_test)
x_test_scaled = np.nan_to_num(x_scale, nan=0.0, posinf=0.0, neginf=0.0) 
Prediction = modelKNN.predict(x_test_scaled)
accuracy_score(Prediction,y_test)
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(Prediction,y_test),annot= True, cmap = 'Blues')
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.title('confusion matrix of KNeighbors Algorithm')
plt.show()

# Logistics Regression
scale = StandardScaler()
x_scale = scale.fit_transform(x_test)
x_test_scaled = np.nan_to_num(x_scale, nan=0.0, posinf=0.0, neginf=0.0)

Prediction = modelLR.predict(x_test_scaled)
accuracy_score(Prediction,y_test)

from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(Prediction,y_test),annot= True, cmap = 'Blues')
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.title('confusion matrix of logistics Algorithm')
plt.show()

# Support vector machine

scale = StandardScaler()
x_scale = scale.fit_transform(x_test)

x_test_scaled = np.nan_to_num(x_scale, nan=0.0, posinf=0.0, neginf=0.0)

Prediction = modelSVM.predict(x_test_scaled)

accuracy_score(Prediction,y_test)

from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(Prediction,y_test),annot= True, cmap = 'Blues')
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.title('confusion matrix of SVM Algorithm')
plt.show()


