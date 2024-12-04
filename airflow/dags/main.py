from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from dataloader import load_data_from_mongodb
from trainer import train_model, evaluate_and_save_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 14),
}

with DAG('mongodb_to_pytorch', default_args=default_args, schedule_interval='@daily') as dag:
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data_from_mongodb,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={'data': "{{ ti.xcom_pull(task_ids='load_data') }}"},
    )

    evaluate_and_save_model_task = PythonOperator(
        task_id='evaluate_and_save_model',
        python_callable=evaluate_and_save_model,
        op_kwargs={'model': "{{ ti.xcom_pull(task_ids='train_model') }}"},
    )

    load_data_task >> train_model_task >> evaluate_and_save_model_task