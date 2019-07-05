from pyspark.sql import SparkSession,Row
from pyspark.sql import HiveContext
import os

os.environ['SPARK_HOME']="/opt/spark"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python"  # set python version
os.environ['JAVA_HOME']='/opt/jdk1.8.0_141'

os.environ['PYTHON_HOME']="/usr/bin/python"

def CreateSparkContext():
    # 构建SparkSession实例对象
    spark = SparkSession.builder \
        .appName("TestSparkSession") \
        .master("spark://hadoop2:7077") \
        .config("hive.metastore.uris", "thrift://hadoop1:9083") \
        .config('spark.executor.num','4')\
        .config('spark.executor.memory','64g')\
        .config("spark.executor.cores",'4')\
        .config('spark.cores.max','16')\
        .config('spark.driver.memory','32g') \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON","/usr/bin/python")\
        .config("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON","/usr/bin/python")\
        .getOrCreate()

    # 获取SparkContext实例对象
    sc = spark.sparkContext
    return sc

def CreateSparkContext_tmp():
    spark = SparkSession.builder \
        .appName("SparkSessionExample") \
        .master("local") \
        .getOrCreate()

    # 获取SparkContext实例对象
    sc = spark.sparkContext
    return sc

def sc_path(pathtype,path):
    global Path
    if pathtype == "local":
        Path = "file://"+path

    else:
        Path = "hdfs://hadoop2:9000/root/hadoop/input/data/"
    print("the path is :{}".format(Path))
    return Path

class HiveOperator():
    def __init__(self, sc):
        self.sc = sc
        return HiveContext(self.sc)

    def result_to_hive(self,sql_list):
        hive_context = HiveContext(self.sc)
        return hive_context.spl(sql_list)