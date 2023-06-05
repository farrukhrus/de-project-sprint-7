import datetime
import sys
import os

import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.window import Window

def main():
    
    os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
    os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
    os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
    
    #date = sys.argv[1]
    #days_cnt = sys.argv[2]
    #events_base_path = sys.argv[3]
    #au_cities_path = sys.argv[4]
    #output_path = sys.argv[5]
    date="2022-05-31"
    num_days=30
    events_base_path="/user/farrukhrus/data/geo/events"
    #au_cities_path="/user/farrukhrus/data/geo.csv"
    au_cities_path="/user/farrukhrus/data/cities_tz.csv"
    output_path="/user/farrukhrus/data/analytics/recomm_dm"
    
    conf = SparkConf().setAppName(f"usersd")
    sc = SparkContext(conf=conf)
    sql = SQLContext(sc)

    to_date = F.to_date(F.lit(date), "yyyy-MM-dd")
    events = sql.read.parquet(events_base_path)\
        .filter(F.col("date").between(F.date_sub(to_date, num_days), to_date)).sample(0.5)

    au_cities = sql.read.option("delimiter", ";").option("header", "true").option("inferSchema", "true").csv(au_cities_path)
    
    recomm_dm = calculate_recommendations_datamart(events, au_cities)
    recomm_dm.write.mode("overwrite").parquet(f"{output_path}/date={date}/depth={num_days}")

def calculate_distance(messages_df):
    window = Window().partitionBy(["user_id", "date"]).orderBy(F.desc("datetime"))
    users = (
        messages_df.select(
            "user_id", "date", "datetime", "message_lat", "message_lon", "city"
        )
        .withColumn("rn", F.row_number().over(window))
        .where("rn=1")
        .select("user_id", "date", "message_lat", "message_lon", "city")
    )

    users_joined = users.join(
        users.select(
            F.col("user_id").alias("user_right"),
            "date",
            F.col("message_lat").alias("lat_right"),
            F.col("message_lon").alias("lon_right"),
            F.col("city").alias("city_right"),
        ),
        "date",
    )
    
    radius=6371
    distances = users_joined.withColumn("distance",
            F.lit(2) * radius * F.asin(
                F.sqrt(
                    F.pow(F.sin((F.col('lat_right') - F.col('message_lat'))/F.lit(2)),2)
                    + 
                    F.cos(F.col("message_lat")) * F.cos(F.col("lat_right"))
                    * F.pow(F.sin((F.col('lon_right') - F.col('message_lon'))/F.lit(2)),2)
                )
            ),
        ).where("distance<=1 and user_id<user_right")

    return distances


def calculate_recommendations_datamart(events, au_cities):
    joined = events.select(
        F.col("event.message_id").alias("message_id"), 
        F.col("event.message_from").alias("user_id"),
        F.col("lat").alias("message_lat"),
        F.col("lon").alias("message_lon"),
        "date", F.to_timestamp("event.datetime").alias("datetime"),
    ).crossJoin(au_cities.select( "city",
            F.col("lat").alias("city_lat"),
            F.col("lng").alias("city_lon"),
            F.col("timezone").alias("city_tz"),
        )
    )

    radius=6371
    window = Window().partitionBy(["message_id"]).orderBy(F.asc("distance"))
    messages_df = joined.withColumn("distance",
            F.lit(2)* radius * F.asin(
                F.sqrt(
                    F.pow(F.sin((F.col('city_lat') - F.col('message_lat'))/F.lit(2)),2)
                    + 
                    F.cos(F.col("message_lat")) * F.cos(F.col("city_lat"))
                    * F.pow(F.sin((F.col('city_lon') - F.col('message_lon'))/F.lit(2)),2)
                )
            ),
        ).withColumn("rn", F.row_number().over(window)).where("rn=1")

    subs = (
        events.where("event.subscription_channel is not null")
        .select(F.col("event.user").alias("user_id"), "event.subscription_channel")
        .distinct()
    )
    subs_pairs = subs.join(
        subs.select(F.col("user_id").alias("user_right"), "subscription_channel"),
        "subscription_channel",
    )

    messages = (
        events.where(
            "event.message_from is not null and event.message_to is not null"
        )
        .select(
            F.col("event.message_from").alias("user_id"),
            F.col("event.message_to").alias("user_right"),
        )
        .distinct()
    )
    messages_pairs = messages.union(
        messages.select(
            F.col("user_id").alias("user_right"), F.col("user_right").alias("user_id")
        )
    )

    window = Window().partitionBy(["user_id"]).orderBy(F.desc("datetime"))
    local_time =  messages_df.select("user_id", "datetime", "city_tz")\
        .withColumn("rn", F.row_number().over(window)).where("rn=1")\
        .withColumn("local_time", F.from_utc_timestamp(F.col("datetime"), F.col("city_tz")))\
        .drop("datetime", "city_tz", "rn")

    distances_df = calculate_distance(messages_df)
    #distances_df.printSchema()
    #distances_df.show(10, True)
    
    recomm_dm = (
        distances_df.join(subs_pairs, on=["user_id", "user_right"], how="leftsemi")
        .join(messages_pairs, on=["user_id", "user_right"], how="leftanti")
        .join(local_time, on="user_id", how="left")
        .select(
            F.col("user_id").alias("user_left"),
            "user_right",
            F.col("date").alias("processed_dttm"),
            F.col("city").alias("zone_id"),
        )
    )
    recomm_dm.show(10, True)

    return recomm_dm


if __name__ == "__main__":
    main()