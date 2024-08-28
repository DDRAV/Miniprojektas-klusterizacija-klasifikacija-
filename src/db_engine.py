"""
Database file to connect, execute commands and close connection
"""

import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

load_dotenv()
class DBEngine:
    def __init__(self):
        self.connection = self.connect()
        self.cursor = self.connection.cursor()

    @staticmethod
    def connect():
        """
        Prisijungiam prie duomenu bazes pagal info env faile
        :return: connection duomenu bazes objektas
        """
        try:
            connection = psycopg2.connect(
                dbname=os.getenv('DATABASE_NAME'),
                user=os.getenv('DB_USERNAME'),
                password=os.getenv('PASSWORD'),
                host=os.getenv('HOST'),
                port=os.getenv('PORT')
            )
            print("PostgreSQL connection is opened")
            return connection
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)
            return None

    def execute_sql(self, query, params=None):
        """
        Paleidziam komanda SQL.
        :param query: Komanda kuria norime paleist
        :param params: Opcionalus parametrai
        :return: Komandos rezultatas: tuple
        """
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            return self.cursor.fetchall()
        except (Exception, psycopg2.Error) as error:
            print("Error while executing query", error)
            self.connection.rollback()
            return None

    def disconnect(self):
        """
        Uždaro duomenų bazės ryšį
        :return None
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
                print('PostgreSQL connection is closed')
        except (Exception, psycopg2.Error) as error:
            print("Error while closing PostgreSQL connection", error)

    def __del__(self):
        self.disconnect()

if __name__ == "__main__":
    db = DBEngine()
    if db.connection:
        result = db.execute_sql(
            "INSERT INTO orders (price, loading_warehouse, loading_time, unloading_warehouse, unloading_time, manager_id, customer_id, carrier_id) VALUES"
            "('1200', 82, '08:00:00', 83, '17:00:00', '1', '101', '2'),"
            "('850', 84, '09:00:00', 85, '18:00:00', '2', '102', '3'),"
            "('950', 86, '08:30:00', 87, '16:30:00', '3', '103', '5'),"
            "('1100', 88, '10:00:00', 89, '15:00:00', '4', '104', '7'),"
            "('1300', 90, '07:00:00', 91, '19:00:00', '5', '105', '8'),"
            "('980', 92, '09:00:00', 93, '14:00:00', '6', '106', '10'),"
            "('1075', 94, '08:30:00', 95, '16:00:00', '7', '107', '12'),"
            "('1150', 96, '10:00:00', 97, '17:00:00', '8', '108', '14'),"
            "('1020', 98, '08:00:00', 99, '15:00:00', '9', '109', '16'),"
            "('1230', 100, '07:00:00', 101, '18:00:00', '10', '110', '18');"
) # Example query
        print(f"Command executed")