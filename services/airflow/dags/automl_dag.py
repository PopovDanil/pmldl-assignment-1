from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


def data_engineering(**kwargs):
    from data_engineering.pipeline import run as data_engineering_run
    data_engineering_run()


def model_engineering(**kwargs):
    from model_engineering.pipeline import run as model_engineering_run
    model_engineering_run()


def deployment(**kwargs):
    from deployment.pipeline import run as deployment_run
    deployment_run()


with DAG(
    dag_id="ML_pipeline",
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1
) as dag:
    data = PythonOperator(
        task_id="data_engineering",
        python_callable=data_engineering,
    )

    model = PythonOperator(
        task_id="model_engineering",
        python_callable=model_engineering,
    )

    deploy = PythonOperator(
        task_id="deployment",
        python_callable=deployment,
    )

    data >> model >> deploy
