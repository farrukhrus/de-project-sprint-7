import os
from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import \
    SparkSubmitOperator

os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf"
os.environ["YARN_CONF_DIR"] = "/etc/hadoop/conf"
os.environ["JAVA_HOME"] = "/usr"
os.environ["SPARK_HOME"] = "/usr/lib/spark"
os.environ["PYTHONPATH"] = "/usr/local/lib/python3.8"


default_args = {"start_date": datetime(2023, 6, 7), "owner": "airflow"}
dag = DAG("datamarts_dag", schedule_interval=None, default_args=default_args)


t1 = SparkSubmitOperator(
    task_id="zones_dm",
    dag=dag,
    application="/lessons/zones_dm.py",
    conn_id="yarn_spark",
    application_args=[
        "2022-06-15",
        "30",
        "/user/farrukhrus/data/geo/events",
        "/user/farrukhrus/data/cities_tz.csv",
        "/user/farrukhrus/data/analytics/zone_dm",
    ],
    conf={"spark.driver.maxResultSize": "20g"},
    num_executors=2,
    executor_memory="4g",
    executor_cores=2,
)

t2 = SparkSubmitOperator(
    task_id="users_dm",
    dag=dag,
    application="/lessons/users_dm.py",
    conn_id="yarn_spark",
    application_args=[
        "2022-06-15",
        "30",
        "/user/farrukhrus/data/geo/events",
        "/user/farrukhrus/data/cities_tz.csv",
        "/user/farrukhrus/data/analytics/users_dm",
    ],
    conf={"spark.driver.maxResultSize": "20g"},
    num_executors=2,
    executor_memory="4g",
    executor_cores=2,
)

t3 = SparkSubmitOperator(
    task_id="recommendations_dm",
    dag=dag,
    application="/lessons/recommendations_dm.py",
    conn_id="yarn_spark",
    application_args=[
        "2022-06-15",
        "30",
        "/user/farrukhrus/data/geo/events",
        "/user/farrukhrus/data/cities_tz.csv",
        "/user/farrukhrus/data/analytics/recomm_dm",
    ],
    conf={"spark.driver.maxResultSize": "20g"},
    num_executors=2,
    executor_memory="4g",
    executor_cores=2,
)

t1 >> t2 >> t3