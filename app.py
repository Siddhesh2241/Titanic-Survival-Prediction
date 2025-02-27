from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from config.insert_data import insert_titanic_data

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

        #PassengerId =int(request.form.get('PassengerId'))
        Pclass = int(request.form.get('Pclass'))
        Age = int(request.form.get('Age'))
        SibSp = int(request.form.get('SibSp'))
        Parch = int(request.form.get('Parch'))
        Fare = float(request.form.get('Fare'))
        Sex = request.form.get("Sex")
        Embarked = request.form.get("Embarked")

        # Use the data to make a prediction

        data =  CustomData(
            #PassengerId =int(request.form.get('PassengerId')),
            Pclass = int(request.form.get('Pclass')),
            Age = int(request.form.get('Age')),
            SibSp = int(request.form.get('SibSp')),
            Parch = int(request.form.get('Parch')),
            Fare = float(request.form.get('Fare')),
            Sex = request.form.get("Sex"),
            Embarked = request.form.get("Embarked")

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        Predicition =  predict_pipeline.predict(pred_df)

        results = "Survived" if Predicition[0] == 1 else "Did not survived"
        insert_titanic_data(int(Pclass), int(Age),  
                            int(SibSp), int(Parch), float(Fare),Sex, Embarked, int(Predicition[0]))
        return render_template("home.html",results=results)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)