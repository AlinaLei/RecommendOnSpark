from pyspark.sql import HiveContext
from pyspark.sql import SparkSession,Row
import os

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
        .config('spark.driver.memory','32g') \
        .config("spark.sql.catalogImplementation", "hive")\
        .getOrCreate()

    # 获取SparkContext实例对象
    sc = spark.sparkContext
    return sc

sc =CreateSparkContext()

hive_context = HiveContext(sc)
hive_context.sql("use sparktest")
#hive_context.sql("select * from category_type limit 50").coalesce(1).write.csv("hdfs://172.16.3.200:8020/user/root/hivetest", mode='overwrite')
hive_context.sql("select * from category_type limit 50").createOrReplaceTempView("test_hive")
hive_context.sql("")
"""
category= sc.textFile('/data/lin/train_data/user_data/category.txt').map(lambda line: line.split(",")[0:3]).toDF(['products','category','channel']).registerTempTable("result_tmp")
result1 = hive_context.sql("select * from result_tmp limit 10")
result1.show()
hive_context.sql("use sparktest")
hive_context.sql("drop table if EXISTS  category_type ")
hive_context.sql(""create table category_type as select * from result_tmp where 1=2 "")
hive_context.sql(""insert overwrite  table category_type  select products ,category,channel from result_tmp"")
#hive_context.sql("load data '/data/lin/train_data/user_data/category.txt' overwrite into table products_user ")
result = hive_context.sql("select * from category_type limit 10")
result.show()

"""
file_name = os.listdir("/data/lin/predict_data/recommend_movie_result/test11/")
for i in file_name:
    if os.path.splitext(i)[1]==".csv":
        names=i
path = "/data/lin/predict_data/recommend_movie_result/test/"+names
print("the path is :{}".format(path))
result_tmp=sc.textFile(path).map(lambda line: line.split("|")[0:5]).toDF(['user','products','rating','category','channel']).registerTempTable("result_tmp")
hive_context = HiveContext(sc)
hive_result = hive_context.sql( """select * from result_tmp limit 10""")
hive_result.show()
hive_context.sql("""use sparktest""")
hive_context.sql("""drop table if EXISTS  recommend_result """)
hive_context.sql("""create table recommend_result as select * from result_tmp where 1=2 """)
hive_context.sql("""insert overwrite  table recommend_result  select * from result_tmp""")

