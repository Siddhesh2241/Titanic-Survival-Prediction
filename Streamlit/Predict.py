import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from config.insert_data import insert_titanic_data

# Set up Streamlit page configuration with an icon
st.set_page_config(page_title="🚢 Titanic Survival Prediction", layout="centered", page_icon="🚢")

# Title and description with emojis
st.title("🚢 Titanic Survival Prediction App")
st.write("Will you survive the Titanic disaster? Enter passenger details to find out! 🧭")

# Add a background color and style the layout
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ffcccb;
        color: black;
        font-size: 20px;
    }
    div.stButton > button:hover {
        background-color: #ff6961;
        color: white;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Form for passenger details input with icons for each field
with st.form("prediction_form"):
    st.header("🔍 Passenger Details")
    Pclass = st.selectbox("🚂 Passenger Class (Pclass)", [1, 2, 3], help="Choose the class of the ticket.")
    Age = st.slider("👤 Age", min_value=0, max_value=100, value=25, help="Select the age of the passenger.")
    SibSp = st.number_input("👫 Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, step=1, help="Enter the number of siblings or spouses aboard.")
    Parch = st.number_input("👨‍👩‍👧‍👦 Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, step=1, help="Enter the number of parents or children aboard.")
    Fare = st.number_input("💸 Fare", min_value=0.0, format="%.2f", help="Enter the fare amount.")
    Sex = st.selectbox("⚧ Gender", ["male", "female"], help="Select the passenger's gender.")
    Embarked = st.selectbox("🌍 Port of Embarkation (Embarked)", ["S", "C", "Q"], help="Select the port where the passenger embarked.")

    submit_button = st.form_submit_button("🔍 Predict Survival")

# Process and display prediction
if submit_button:
    # Prepare data for prediction
    data = CustomData(
        Pclass=Pclass,
        Age=Age,
        SibSp=SibSp,
        Parch=Parch,
        Fare=Fare,
        Sex=Sex,
        Embarked=Embarked
    )

    pred_df = data.get_data_as_data_frame()

    # Predict survival
    predict_pipeline = PredictPipeline()
    try:
        Prediction = predict_pipeline.predict(pred_df)
        result_text = "🟢 Survived" if Prediction[0] == 1 else "🔴 Did not survive"
        
        # Display result with icon
        st.markdown(f"## Prediction: {result_text}")

        # Insert data into database
        insert_titanic_data(int(Pclass), int(Age), int(SibSp), int(Parch), float(Fare), Sex, Embarked, int(Prediction[0]))

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

# About section with more emojis and a footer style
st.markdown("---")
st.write("### ℹ️ About this App")
st.write(
    "This app uses a machine learning model to predict if a Titanic passenger would survive based on demographic and ticket information. "
    "It takes into account details like passenger class, age, family aboard, and more. 🧑‍🏫"
)
st.markdown("<hr style='border-top: 2px solid #eee;'>", unsafe_allow_html=True)
st.write("📝 Created with Streamlit | Made for fun Titanic Survival Predictions 🚢")
