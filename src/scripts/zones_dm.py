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
    output_path="/user/farrukhrus/data/analytics/zone_dm"
    
    conf = SparkConf().setAppName(f"usersd")
    sc = SparkContext(conf=conf)
    sql = SQLContext(sc)

    to_date = F.to_date(F.lit(date), "yyyy-MM-dd")

    events = sql.read.parquet(events_base_path)\
        .filter(F.col("date").between(F.date_sub(to_date, num_days), to_date))\
        .sample(0.01)

    au_cities = sql.read.option("delimiter", ";")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .csv(au_cities_path)

    zones_dtmrt = calculate_zones_dm( events=events, au_cities=au_cities )
    zones_dtmrt.write.parquet(f"{output_path}/date={date}/depth={num_days}")


def calculate_zones_dm(events, au_cities):
    joined = events.select(
        "event.message_id", F.col("event.message_from").alias("user_id"),
        F.col("event_type").alias("event_type"),
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

    #messages_df.show(10, True)
    window = Window().partitionBy(["month", "city"])
    zones_dm = (
        messages_df.select("message_id", "user_id", "date", "event_type", "city")
        .withColumn("month", F.trunc(F.col("date"), "month"))
        .withColumn("week", F.trunc(F.col("date"), "week"))
        .withColumn("month_message",
            F.sum(F.when(messages_df.event_type == "message", 1).otherwise(0)).over(window),)
        .withColumn("month_reaction",
            F.sum(F.when(messages_df.event_type == "reaction", 1).otherwise(0)).over(window),)
        .withColumn("month_subscription",
            F.sum(F.when(messages_df.event_type == "subscription", 1).otherwise(0)).over(window),)
        .withColumn("month_user", F.size(F.collect_set("user_id").over(window)))
        .withColumn("message", F.when(messages_df.event_type == "message", 1).otherwise(0))
        .withColumn("reaction", F.when(messages_df.event_type == "reaction", 1).otherwise(0))
        .withColumn("subscription",F.when(messages_df.event_type == "subscription", 1).otherwise(0),)
        .groupBy(
            "month",
            "week",
            F.col("city").alias("zone_id"),
            "month_message",
            "month_reaction",
            "month_subscription",
            "month_user",
        )
        .agg(
            F.sum("message").alias("week_message"),
            F.sum("reaction").alias("week_reaction"),
            F.countDistinct("user_id").alias("week_user"),
        )
    )
    #zones_dm.printSchema()
    #zones_dm.show(10,True)
    
    return zones_dm

if __name__ == "__main__":
    main()