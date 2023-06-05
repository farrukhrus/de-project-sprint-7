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
    output_path="/user/farrukhrus/data/analytics/users_dm"
    
    conf = SparkConf().setAppName(f"usersd")
    sc = SparkContext(conf=conf)
    sql = SQLContext(sc)

    to_date = F.to_date(F.lit(date), "yyyy-MM-dd")
    events = sql.read.parquet(events_base_path)\
        .filter(F.col("date").between(F.date_sub(to_date, num_days), to_date))\
        .where("event.message_from is not null and event_type = 'message'").sample(0.01)
    #events.printSchema()
    #events.show(10, True)
    
    au_cities = sql.read.option("delimiter", ";").option("header", "true").option("inferSchema", "true").csv(au_cities_path)
    users_dm(events, au_cities).write.mode("overwrite").parquet(f"{output_path}/date={date}/depth={num_days}")

def users_dm(events, au_cities):
    
    # all users
    joined = events.select(
        "event.message_id", F.col("event.message_from").alias("user_id"),
        F.col("lat").alias("message_lat"),
        F.col("lon").alias("message_lon"),
        "date", F.to_timestamp("event.datetime").alias("datetime"),
    ).crossJoin(au_cities.select( "city",
            F.col("lat").alias("city_lat"),
            F.col("lng").alias("city_lon"),
            F.col("timezone").alias("city_tz"),
        )
    )

    # distance calc
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
    
    active_window = Window().partitionBy(["user_id"]).orderBy(F.desc("date"))
    active = messages_df.withColumn("rn_active", F.row_number().over(active_window))\
        .where("rn_active=1").select("user_id", F.col("city").alias("act_city"))

    visit_window = Window().partitionBy(["user_id"]).orderBy(F.asc("date"))
    visits = messages_df.select("user_id", "city", "date")\
        .withColumn("lag_city", F.lag("city", 1).over(visit_window))\
        .where("lag_city!=city or lag_city is null")\
        .withColumn("lead_date", F.lead("date", 1).over(visit_window))\
        .withColumn("lead_date_ifnull",
                    F.when(F.col("lead_date").isNull(), datetime.date.today())
                    .otherwise(F.col("lead_date")),)\
        .withColumn("days_in", F.datediff(F.col("lead_date_ifnull"), F.col("date")))
    
    # residents
    days_for_residency = 27
    home_window = Window().partitionBy(["user_id"]).orderBy(F.desc("days_in"), F.desc("date"))
    home_city = visits.where(f"days_in>={days_for_residency}")\
        .withColumn("rn_home", F.row_number().over(home_window))\
        .where(f"rn_home=1")\
        .select("user_id", F.col("city").alias("home_city"))

    # travelers
    travel_cities = visits.groupby("user_id")\
        .agg(F.collect_list("city").alias("travel_array"))\
        .withColumn("travel_count", F.size(F.col("travel_array")))

    # local time 
    window = Window().partitionBy(["user_id"]).orderBy(F.desc("datetime"))
    local_time =  messages_df.select("user_id", "datetime", "city_tz")\
        .withColumn("rn", F.row_number().over(window)).where("rn=1")\
        .withColumn("local_time", F.from_utc_timestamp(F.col("datetime"), F.col("city_tz")))\
        .drop("datetime", "city_tz", "rn")
    
    local_time.show(10, False)

    # final
    return active.join(home_city, on="user_id", how="left")\
        .join(travel_cities, on="user_id", how="left")\
        .join(local_time, on="user_id", how="left")


if __name__ == "__main__":
    main()