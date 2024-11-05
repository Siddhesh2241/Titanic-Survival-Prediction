import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Decision Tree":DecisionTreeClassifier(),
                "KNeighbors":KNeighborsClassifier(),
                "LogisticRegression":LogisticRegression(),
                "Support Vector Machine": SVC(),
                "RandomForestClassifier": RandomForestClassifier(),
                "BaggingClassifier":BaggingClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(algorithm="SAMME"),
                "XGBClassifier":XGBClassifier()
            }

            parms = {

                "Decision Tree": {
                'max_depth': [None, 10, 20, 30, 40, 50],
            
                 },
                "KNeighbors": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
                 },
                "LogisticRegression": {
               
                'C': [0.01, 0.1, 1, 10, 100]
              
                 },
               "Support Vector Machine": {
               'C': [0.1, 1, 10, 100],
               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma': ['scale', 'auto']
                },
               "RandomForestClassifier": {
               'n_estimators': [100, 200, 300, 400],
               'max_depth': [None, 10, 20, 30]
            
                },
               "BaggingClassifier": {
               'n_estimators': [10, 50, 100],
                },
               "AdaBoostClassifier": {
               'n_estimators': [50, 100, 150],
               'learning_rate': [0.01, 0.1, 1.0, 1.5]
                },
               "XGBClassifier": {
               'n_estimators': [50, 100, 200],
                }
              }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,
                                                y_test=y_test,models=models,param=parms)
            
            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            
            predicted = best_model.predict(X_test)

            Accuracy = accuracy_score(y_test,predicted)

            return Accuracy,best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)