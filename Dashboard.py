import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the data
data_path = 'notebook\data\Data.csv'
data = pd.read_csv(data_path)

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna('Unknown', inplace=True)

# Add age group column for analysis
def categorize_age(age):
    if age < 18:
        return 'Child'
    elif age < 60:
        return 'Adult'
    else:
        return 'Elderly'

data['AgeGroup'] = data['Age'].apply(categorize_age)

# Streamlit App
def main():
    st.title("Titanic Survival Analysis Dashboard")

    # Sidebar Filters
    st.sidebar.header("Filters")
    selected_class = st.sidebar.multiselect("Select Passenger Class:", options=data['Pclass'].unique(), default=data['Pclass'].unique())
    selected_gender = st.sidebar.multiselect("Select Gender:", options=data['Sex'].unique(), default=data['Sex'].unique())

    # Filtered Data
    filtered_data = data[(data['Pclass'].isin(selected_class)) & (data['Sex'].isin(selected_gender))]

    # Metrics at the top
    st.metric(label="Count of Passenger IDs", value=filtered_data['PassengerId'].count())
    st.metric(label="Average of Age", value=round(filtered_data['Age'].mean(), 2))
    st.metric(label="Count of Fare", value=round(filtered_data['Fare'].sum(), 2))

    # Subplots with specific types for each cell
    fig = make_subplots(
        rows=3, cols=3, 
        subplot_titles=(
            "Survived by Age", "Survived by Pclass", "Passenger List",
            "Survived by Embarked", "Age by Age Group", 
            "Survived by Gender", "Fare Distribution", "Age Distribution"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.1,  # Add vertical spacing
        horizontal_spacing=0.1  # Add horizontal spacing
    )

    # Plot 1: Survived by Age (Bar plot)
    survived_by_age = filtered_data.groupby(['Survived', 'Age']).size().reset_index(name='Count')
    fig.add_trace(go.Bar(x=survived_by_age['Survived'], y=survived_by_age['Count'], 
                         name="Survived by Age", marker_color='skyblue'), row=1, col=1)

    # Plot 2: Survived by Pclass (Bar plot)
    class_survival = filtered_data.groupby('Pclass')['Survived'].mean()
    fig.add_trace(go.Bar(x=class_survival.index, y=class_survival.values, name="Survived by Pclass", marker_color='skyblue'), row=1, col=2)

    # Plot 3: Passenger List (Table)
    st.markdown("### Passenger List")
    st.dataframe(filtered_data[['Name', 'Sex', 'Age', 'Pclass']])

    # Plot 4: Survived by Embarked (Bar plot)
    embarked_survival = filtered_data.groupby('Embarked')['Survived'].mean()
    fig.add_trace(go.Bar(x=embarked_survival.index, y=embarked_survival.values, name="Survived by Embarked", marker_color='lightcoral'), row=2, col=1)

    # Plot 5: Age by Age Group (Bar plot)
    age_group_counts = filtered_data['AgeGroup'].value_counts()
    fig.add_trace(go.Bar(x=age_group_counts.index, y=age_group_counts.values, name="Age by Age Group", marker_color='lightgreen'), row=2, col=2)

    # Plot 6: Survived by Gender (Bar plot)
    gender_survival = filtered_data.groupby('Sex')['Survived'].mean()
    fig.add_trace(go.Bar(x=gender_survival.index, y=gender_survival.values, name="Survived by Gender", marker_color='lightblue'), row=3, col=1)

    # Plot 7: Fare Distribution (Histogram)
    fig.add_trace(go.Histogram(x=filtered_data['Fare'], name="Fare Distribution", marker_color='orange'), row=3, col=2)

    # Plot 8: Age Distribution (Histogram)
    fig.add_trace(go.Histogram(x=filtered_data['Age'], name="Age Distribution", marker_color='yellowgreen'), row=3, col=3)

    # Update Layout for better visual spacing
    fig.update_layout(
        height=1200, 
        width=1400, 
        showlegend=True, 
        title_text="Titanic Data Insights",
        bargap=0.2,  # Space between bars for better clarity
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),  # Remove gridlines for a cleaner look
        yaxis=dict(showgrid=True, gridcolor='lightgrey')  # Light grey gridlines for easy reference
    )

    
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
