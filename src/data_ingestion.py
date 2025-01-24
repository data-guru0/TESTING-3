import psycopg2
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import os
from sklearn.model_selection import train_test_split
import sys
from config.databse_config import DB_CONFIG
from config.paths_config import *

logger=get_logger(__name__)

class DataIngestion:
    def __init__(self, db_params, output_dir):
        self.db_params = db_params
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def connect_to_db(self):
        """Establish connection to PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.db_params['host'],
                port=self.db_params['port'],
                dbname=self.db_params['dbname'],
                user=self.db_params['user'],
                password=self.db_params['password']
            )
            logger.info("Database connection established.")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise CustomException(str(e), sys)

    def extract_data(self):
        """Extract data from PostgreSQL and return as DataFrame"""
        try:
            conn = self.connect_to_db()
            query = "SELECT * FROM public.titanic;"
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Data extracted from table 'titanic'.")
            return df
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            raise CustomException(str(e), sys)

    def save_data(self, df):
        """Save DataFrame to CSV after splitting into train and test sets"""
        try:
            # Split data into 80-20 train-test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data to CSV
            train_output_path = os.path.join(self.output_dir, 'titanic_train.csv')
            test_output_path = os.path.join(self.output_dir, 'titanic_test.csv')

            train_df.to_csv(train_output_path, index=False)
            test_df.to_csv(test_output_path, index=False)

            logger.info(f"Data saved: Training data to {train_output_path}, Testing data to {test_output_path}.")
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            raise CustomException(str(e), sys)

    def run(self):
        """Run the data ingestion process"""
        try:
            logger.info("Data ingestion process started.")
            df = self.extract_data()
            self.save_data(df)
            logger.info("Data ingestion process completed successfully.")
        except CustomException as ce:
            logger.error(f"CustomException occurred: {ce}")
            raise ce
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise CustomException(str(e), sys)


if __name__ == "__main__":

    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion.run()