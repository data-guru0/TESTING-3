from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base_hook import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy

def load_to_sql(file_path):
    # Get PostgreSQL connection from Airflow
    conn = BaseHook.get_connection('postgres_default')  # Use the connection ID you set in the Airflow UI
    # Create SQLAlchemy engine for PostgreSQL
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@testing-3_607202-postgres-1:{conn.port}/{conn.schema}")
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    # Load data into PostgreSQL, replacing the table if it exists
    df.to_sql(name="titanic", con=engine, if_exists="replace", index=False)

# Define the DAG
with DAG(
    dag_id="extract_titanic_data",
    schedule_interval=None,  # No schedule for now
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Step 1: List Objects in GCP Bucket (you may want to list files before downloading)
    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket="my-bucket45",  # Your GCP Bucket name here
    )

    # Step 2: Download the Titanic Dataset from GCS to local filesystem
    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        bucket="my-bucket45",  # Your GCP Bucket name here
        object_name="Titanic-Dataset.csv",  # GCS object name (CSV file)
        filename="/tmp/Titanic-Dataset.csv",  # Local destination file path
    )

    # Step 3: Load Titanic Data into PostgreSQL
    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/tmp/Titanic-Dataset.csv"}  # Path to the CSV
    )

    # Set task dependencies (order of execution)
    list_files >> download_file >> load_data
