import sys
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def Predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
class CustomData:
        def __init__(self,
                    PassengerId:int,
                    Pclass:int,
                    Age:int,
                    SibSp:int,
                    Parch:int,
                    Fare:int,
                    Sex:str,
                    Embarked:str  ):
        
                    self.PassengerId = PassengerId
                    self.Pclass = Pclass
                    self.Age = Age
                    self.SibSp = SibSp
                    self.Parch = Parch
                    self.Fare = Fare
                    self.Sex = Sex
                    self.Embarked = Embarked

        def get_data_as_data_frame(self):
             try:
                  custom_data_input_dict = {
                       "PassengerId" : [self.PassengerId],
                       "Pclass": [self.Pclass],
                       "Age":[self.Age],
                       "SibSp":[self.SibSp],
                       "Parch":[self.Parch],
                       "Fare":[self.Fare],
                       "Sex": [self.Sex],
                       "Embarked": [self.Embarked]
                    }

                  return pd.DataFrame(custom_data_input_dict)

             except Exception as e:
                  raise CustomException(e,sys)