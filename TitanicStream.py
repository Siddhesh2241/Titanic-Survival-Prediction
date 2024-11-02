import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(filepath):
    """Load data from a CSV file."""
    data = pd.read_csv(filepath)
    return data


def display_basic_info(data):
    """Display basic information about the dataset."""
    st.write("Data Head:", data.head())
    st.write("Data Shape:", data.shape)
    st.write("Data Columns:", data.columns.tolist())
    st.write("Data Info:")
    st.write(data.info())
    st.write("Data Description:", data.describe())
    st.write("Duplicated Rows:", data.duplicated().sum())
    st.write("Number of Unique Values per Column:", data.nunique())
    st.write("Missing Values:", data.isnull().sum())


def preprocess_data(data):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Embarked"].fillna("S", inplace=True)
    data = data.drop(["Cabin", "Name", "Ticket"], axis=1)

    # Convert columns to correct data types
    data["Age"] = data["Age"].astype(int)
    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "male" else 0)
    data["Fare"] = data["Fare"].astype(int)

    # Encode categorical features
    le = LabelEncoder()
    data["Embarked"] = le.fit_transform(data["Embarked"])

    # Map age groups
    age_mapping = {'Infant': 0, 'Teen': 1, '20s': 2, '30s': 3, '40s': 4, '50s': 5, 'Elder': 6}
    data['Age'] = pd.cut(data['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100],
                         labels=['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elder']).map(age_mapping)
    data.dropna(subset=['Age'], inplace=True)
    return data


def plot_data(data):
    """Plot various data visualizations."""
    fig, ax = plt.subplots(2, 4, figsize=(20, 20))
    sns.countplot(x="Survived", data=data, ax=ax[0, 0])
    sns.countplot(x='Pclass', data=data, ax=ax[0, 1])
    sns.countplot(x='Sex', data=data, ax=ax[0, 2])
    sns.countplot(x='Age', data=data, ax=ax[0, 3])
    sns.countplot(x='Embarked', data=data, ax=ax[1, 0])
    sns.histplot(x='Fare', data=data, bins=10, ax=ax[1, 1])
    sns.countplot(x='SibSp', data=data, ax=ax[1, 2])
    sns.countplot(x='Parch', data=data, ax=ax[1, 3])
    st.pyplot(fig)

    plt.figure(figsize=(6, 6))
    sns.countplot(x='Age', data=data)
    st.pyplot(plt.gcf())

    fig, ax = plt.subplots(2, 3, figsize=(20, 20))
    sns.countplot(x='Sex', data=data, hue='Survived', ax=ax[0, 0])
    sns.countplot(x='Age', data=data, hue='Survived', ax=ax[0, 1])
    sns.countplot(x="Pclass", data=data, hue="Survived", ax=ax[0, 2])
    sns.countplot(x='Parch', data=data, hue='Survived', ax=ax[1, 0])
    sns.scatterplot(x='SibSp', y='Parch', data=data, hue='Survived', ax=ax[1, 1])
    sns.pointplot(x='Pclass', y='Survived', data=data, ax=ax[1, 2])
    st.pyplot(fig)


def plot_correlation_heatmap(data):
    """Plot the correlation heatmap of the dataset."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.corr(), annot=True, cmap='Blues')
    plt.title("Correlation Matrix")
    st.pyplot(plt.gcf())


def scale_data(x_train, x_test):
    """Scale the training and testing data."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train_scaled = np.nan_to_num(x_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_scaled = np.nan_to_num(x_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_scaled, x_test_scaled


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    """Train and evaluate the model, displaying accuracy and confusion matrix."""
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    st.write(f"{model_name} Train Accuracy:", accuracy_score(train_pred, y_train))
    st.write(f"{model_name} Test Accuracy:", accuracy_score(test_pred, y_test))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(test_pred, y_test), annot=True, cmap='Blues')
    plt.ylabel('Predicted Values')
    plt.xlabel('Actual Values')
    plt.title(f'Confusion Matrix of {model_name}')
    st.pyplot(plt.gcf())


def main():
    st.title("Titanic Data Analysis and Model Evaluation")

    # Load and display data
    data = load_data("Data.csv")
    display_basic_info(data)

    # Preprocess the data
    data = preprocess_data(data)
    st.write("Data Types After Preprocessing:", data.dtypes)

    # Plot data
    plot_data(data)
    plot_correlation_heatmap(data)

    # Prepare data for modeling
    x = data.drop("Survived", axis=1)
    y = data["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    # Train and evaluate models
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC()
    }

    for model_name, model in models.items():
        train_and_evaluate_model(model, x_train_scaled, y_train, x_test_scaled, y_test, model_name)


if __name__ == "__main__":
    main()
