from pyspark.sql import HiveContext
from pyspark.sql import SparkSession,Row

def CreateSparkContext():
    # 构建SparkSession实例对象
    spark = SparkSession.builder \
        .appName("TestSparkSession") \
        .master("local") \
        .config("hive.metastore.uris", "thrift://hadoop1:9083") \
        .config('spark.executor.num','4')\
        .config('spark.executor.memory','32g')\
        .config("spark.executor.cores",'4')\
        .config('spark.cores.max','8')\
        .config('spark.driver.memory','32g')\
        .getOrCreate()

    # 获取SparkContext实例对象
    sc = spark.sparkContext
    return sc

sc =CreateSparkContext()
hive_context = HiveContext(sc)
category= sc.textFile('/data/lin/train_data/user_data/category.txt').map(lambda line: line.split(",")[0:3]).toDF(['products','category','channel']).registerTempTable("result_tmp")
hive_context.sql("use sparktest")
hive_context.sql("drop table if EXISTS  category_type ")
hive_context.sql("create table category_type(products string,category string ,channel string) ")
hive_context.sql('insert into table category_type as select products ,category,channel from result_tmp')
#hive_context.sql("load data '/data/lin/train_data/user_data/category.txt' overwrite into table products_user ")
"""
hive_context.sql("drop table if EXISTS  products_user  ")
hive_context.sql("create table if not exists products_user(user_id:string ,products:string,rating:string,category:string,channel:string)")
hive_context.sql("load data '/data/lin/predict_data/recommend_movie_result/test' overwrite into table products_user")
"""
result = hive_context.sql("select * from category_type limit 10")
print("the result is :{}".format(result))
print("the type is :{}".format(type(result)))