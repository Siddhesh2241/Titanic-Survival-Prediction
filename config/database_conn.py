import sys
import mysql.connector
from mysql.connector import Error

from src.exception import CustomException
from src.logger import logging


def Create_connection():
    """Creates and returns a connection to the MySQL database."""

    try:
        connection = mysql.connector.connect(
            host = "127.0.0.1",
            port = "3306",
            username = "root",
            password = "siddhesh2241",
            database = "new_titanic"
        )

        logging.info("DataBase connected succesfully")

        return connection
    
    except Exception as e:
        raise CustomException(e,sys)
        