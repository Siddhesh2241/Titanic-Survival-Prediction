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

st.set_page_config(
    page_title="Titanic Survival Prediction using Supervised Classification Algorithm",
    page_icon=":ship:",
    layout="wide"
)

def load_data():
    st.subheader("Upload CSV file :file_folder:")
    upload_file = st.file_uploader("Upload file", type="csv")

    if upload_file is not None:
        data = pd.read_csv(upload_file)
        st.dataframe(data)
        return data

def Analyse_data(data):
    # Exploratory Data Analysis (EDA)
    st.subheader("First five rows of the dataset")
    st.write(data.head())

    st.subheader("Columns in dataset:")
    st.write(data.columns.tolist())

    st.subheader("Numerical columns in dataset:")
    st.write([col for col in data.columns if data[col].dtypes != "object"])

    st.subheader("Categorical columns in dataset:")
    st.write([col for col in data.columns if data[col].dtypes == "object"])

    st.subheader("Information of the CSV file:")
    st.write(data.info())

    st.subheader("Statistical data of the dataset:")
    st.write(data.describe())

    st.subheader("Check for duplicates in the dataset:")
    st.write(data.duplicated().sum())

    st.subheader("Unique values in the dataset:")
    st.write(data.nunique())

    st.subheader("Check for null values in the dataset:")
    st.write(data.isnull().sum())

def preprocess_data(data):
    st.subheader("Filling null values of columns")
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Embarked"].fillna("S", inplace=True)

    st.write("Data after Filling Missing values", data)

    st.subheader("Again check missing values")
    st.write(data.isnull().sum())

    st.subheader("Delete Unnecessary column")
    columns_to_drop = st.multiselect("Select the columns you want to drop:", options=data.columns.tolist())

    if st.button("Submit and drop columns"):
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
            st.write(f"### DataFrame after dropping columns: {columns_to_drop}")
            st.write(data)
        else:
            st.write("No columns dropped.")

    st.subheader("Convert Age and Fare columns to int data types")
    try:
        data["Age"] = data["Age"].astype(int)
        data["Fare"] = data["Fare"].astype(int)
    except ValueError:
        st.error("Conversion failed: Ensure no missing values remain in Age and Fare columns.")

    st.write(data[["Age", "Fare"]].dtypes)

    # Encode categorical columns
    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "male" else 0)

    st.subheader("Encode categorical features")
    le = LabelEncoder()
    data["Embarked"] = le.fit_transform(data["Embarked"])
    st.write("Data after encoding", data["Embarked"])

    st.subheader("Map age groups")
    age_mapping = {'Infant': 0, 'Teen': 1, '20s': 2, '30s': 3, '40s': 4, '50s': 5, 'Elder': 6}
    data['Age'] = pd.cut(data['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100],
                         labels=['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elder']).map(age_mapping)
    data.dropna(subset=['Age'], inplace=True)

    # One-hot encode remaining non-numeric columns
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    if non_numeric_cols.any():
        data = pd.get_dummies(data, columns=non_numeric_cols)

    return data

def plot_data(data):
    st.subheader("Basic EDA Analysis")
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

    st.subheader("Relation between features")
    fig, ax = plt.subplots(2, 3, figsize=(20, 20))
    sns.countplot(x='Sex', data=data, hue='Survived', ax=ax[0, 0])
    sns.countplot(x='Age', data=data, hue='Survived', ax=ax[0, 1])
    sns.countplot(x="Pclass", data=data, hue="Survived", ax=ax[0, 2])
    sns.countplot(x='Parch', data=data, hue='Survived', ax=ax[1, 0])
    sns.scatterplot(x='SibSp', y='Parch', data=data, hue='Survived', ax=ax[1, 1])
    sns.pointplot(x='Pclass', y='Survived', data=data, ax=ax[1, 2])
    st.pyplot(fig)

def plot_correlation_heatmap(data):
    st.subheader("Plot the correlation heatmap of the dataset")
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='Blues')
    plt.title("Correlation Matrix")
    st.pyplot(plt.gcf())


def scale_data(x_train, x_test):
    st.subheader("Scale the training and testing data.")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train_scaled = np.nan_to_num(x_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_scaled = np.nan_to_num(x_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_scaled, x_test_scaled


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    st.subheader(f"Train and evaluate the {model_name} model")
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

    st.title("Titanic Survival Prediction :ship:")
    st.markdown("Titanic survival prediction using supervised classification algorithms")

    st.sidebar.header("Classification algorithms")
    Algos = st.sidebar.selectbox(
        "Choose algorithm", 
        options=["Decision Tree", "K-Nearest Neighbors", "Logistic Regression", "Support Vector Machine"], 
        index=0
    )
    
    n_neighbors = 5  # Default value
    if Algos == "K-Nearest Neighbors":
        n_neighbors = st.sidebar.slider("Select n_neighbors:", min_value=1, max_value=12, value=5, step=2)

    data = load_data()

    if data is not None:
        Analyse_data(data)
       
        data = preprocess_data(data)
        st.write("Data Types After Preprocessing:", data.dtypes)
        
        plot_data(data)
        #plot_correlation_heatmap(data)

        # Prepare data for modeling
        x = data.drop("Survived", axis=1)
        y = data["Survived"]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)
        x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
        
       
        # Mapping of algorithms
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=n_neighbors),
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC()
        }

        # Train and evaluate the selected algorithm only
        selected_model = models[Algos]

        train_and_evaluate_model(selected_model, x_train_scaled, y_train, x_test_scaled, y_test, Algos)

        

if __name__ == "__main__":
    main()
