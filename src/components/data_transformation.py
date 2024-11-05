import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer # it is used to create pipeline
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
         This function is responsible for data transformation
        '''
        try:
            # Define pipelines for numerical and categorical transformations
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine pipelines into a single preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Drop unnecessary columns
            columns_to_drop = ["Name", "Ticket", "Cabin"]
            train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
            test_df = test_df.drop(columns=columns_to_drop, errors="ignore")

            logging.info("Dropped columns: Name, Ticket, Cabin")
            logging.info(f"Train DataFrame columns after dropping: {train_df.columns.tolist()}")
            logging.info(f"Test DataFrame columns after dropping: {test_df.columns.tolist()}")

            # Define target column and identify feature columns
            target_column_name = "Survived"
            numerical_columns = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]

            # Dynamically identify categorical columns (non-numeric) after dropping columns
            categorical_columns = [col for col in train_df.columns if train_df[col].dtype == "object" and col not in [target_column_name]]

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            # Separate input and target features for train and test data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate transformed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
