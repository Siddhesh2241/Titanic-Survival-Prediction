from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from config.Insert_data import insert_student_data

application = Flask(__name__)

app = application

# Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictedata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        # Retrieve form data

        PassengerId =float(request.form.get('PassengerId'))
        Pclass = float(request.form.get('Pclass'))
        Age = float(request.form.get('Age'))
        lunch = float(request.form.get('lunch'))
        SibSp = float(request.form.get('SibSp'))
        Parch = float(request.form.get('Parch'))
        Fare = float(request.form.get('Fare'))
        Sex = request.form.get("Sex")
        Embarked = request.form.get("Embarked")

        # Use the data to make a prediction

        data =  CustomData(
            PassengerId =float(request.form.get('PassengerId')),
            Pclass = float(request.form.get('Pclass')),
            Age = float(request.form.get('Age')),
            lunch = float(request.form.get('lunch')),
            SibSp = float(request.form.get('SibSp')),
            Parch = float(request.form.get('Parch')),
            Fare = float(request.form.get('Fare')),
            Sex = request.form.get("Sex"),
            Embarked = request.form.get("Embarked"),

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results =  predict_pipeline.predict(pred_df)

        insert_student_data(int(PassengerId), int(Pclass), int(Age), int(lunch), 
                            int(SibSp), int(Parch), int(Fare),Sex, Embarked, int(results[0]))
        return render_template("home.html",results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)