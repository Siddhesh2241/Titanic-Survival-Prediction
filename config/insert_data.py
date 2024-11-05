import sys
from config.database_conn import Create_connection

from src.exception import CustomException
from src.logger import logging

def insert_titanic_data(Pclass, Age, SibSp, Parch, Fare, Sex, Embarked,Predicition):
    connection = Create_connection()
    if connection:
        try:
          cursor = connection.cursor()
          insert_query = """
            INSERT INTO inter_titanicdata (Pclass, Age, SibSp, Parch, Fare, Sex, Embarked,Predicition)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
          """
          data_tuple = (Pclass, Age, SibSp, Parch, Fare, Sex, Embarked,Predicition)
          cursor.execute(insert_query, data_tuple)
          connection.commit()
          cursor.close()
          connection.close()

          logging.info("Data added into database succesfully")
        
        except Exception as e:
           raise CustomException(e,sys)